#!/usr/bin/env python

import os, pdb, argparse, json, torch, cv2, numpy as np, rtree, pandas, glob
from torch.autograd import Variable
from shapely.geometry import MultiPolygon, box
from ssd import build_ssd
from data import BaseTransform

transform = BaseTransform(300, (104, 117, 123))

def get_metrics(gt_boxes, pred_boxes):
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    total_overlap = 0.0

    # Create the RTree out of the ground truth boxes
    idx = rtree.index.Index()
    for i, rect in enumerate(gt_boxes):
        idx.insert(i, tuple(rect))

    gt_mp = MultiPolygon([box(*b) for b in gt_boxes])
    pred_mp = MultiPolygon([box(*b) for b in pred_boxes])

    for rect in pred_boxes:
        best_jaccard = 0.0
        best_idx = None
        best_overlap = 0.0
        for gt_idx in idx.intersection(rect):
            gt = gt_boxes[gt_idx]
            intersection = (min(rect[2], gt[2]) - max(rect[0], gt[0])) * (min(rect[3], gt[3]) - max(rect[1], gt[1]))
            rect_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            union = rect_area + gt_area - intersection
            jaccard = float(intersection) / float(union)
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_idx = gt_idx
            if intersection > best_overlap:
                best_overlap = intersection
        if best_idx is None or best_jaccard <= 0.00000000000001:
            false_positives += 1
        else:
            idx.delete(best_idx, gt_boxes[best_idx])
            true_positives += 1
        total_overlap = best_overlap
    total_jaccard = total_overlap / (gt_mp.area + pred_mp.area - total_overlap) if len(gt_boxes) > 0 else None
    false_negatives = len(gt_boxes) - true_positives
    return false_positives, false_negatives, true_positives, total_jaccard

def img_iter(test_set, data_dir):
    for sample in test_set:
        img = cv2.imread(os.path.join(data_dir, sample['image_path']))

        boxes = np.array([[r['x1'], r['y1'], r['x2'], r['y2']] for r in sample['rects']])

        for i in range(0, 201, 200):
            for j in range(0, 201, 200):
                subset = img[i:i+300, j:j+300, :]
                current_boxes = boxes.copy()
                if len(current_boxes) > 0:
                    current_boxes[:, (0, 2)] -= j
                    current_boxes[:, (1, 3)] -= i
                    current_boxes = np.clip(current_boxes, a_min = 0, a_max = 300)
                    mask = (current_boxes[:, 2] - current_boxes[:, 0] > 2) & (current_boxes[:, 3] - current_boxes[:, 1] > 2)
                    current_boxes = current_boxes[mask, :]

                transformed, _, _ = transform(subset.copy())
                yield torch.from_numpy(transformed.transpose((2,0,1))[(2,1,0),:,:]).float(), subset, current_boxes

def test_net(net, test_set, data_dir, batch_size = 16, thresh=0.5):
    dataset = img_iter(test_set, data_dir)
    results = []

    false_positives, false_negatives, true_positives, num_samples = 0,0,0,0

    [os.remove(f) for f in glob.glob('samples/*.jpg')]

    for batch_num in range(0, len(test_set), batch_size):
        inputs, originals, targets = zip(*[next(dataset) for _ in range(batch_size)])
        inputs = Variable(torch.stack(inputs, 0).cuda(), volatile=True)

        y = net(inputs)      # forward pass
        detections = y.data.cpu().numpy()

        ridx = np.random.randint(len(detections))

        for i in range(len(detections)):
            dets = detections[i, 1]
            orig = originals[i]

            dets[:, (1, 3)] = np.clip(dets[:, (1, 3)] * orig.shape[1], a_min=0, a_max = orig.shape[1])
            dets[:, (2, 4)] = np.clip(dets[:, (2, 4)] * orig.shape[0], a_min=0, a_max = orig.shape[0])

            valid_dets = dets[dets[:, 0] >= thresh, :]

            fp, fn, tp, jaccard = get_metrics(targets[i], valid_dets[:, 1:].round().astype(int))
            false_positives += fp
            false_negatives += fn
            true_positives += tp
            num_samples += len(targets[i])

            results.append({
                'false_positives' : fp,
                'false_negatives' : fn,
                'true_positives' : tp,
                'jaccard' : jaccard
            })
            if tp < fp + fn:
                orig = orig.copy()
                actual = orig.copy()
                for box in valid_dets[:, 1:].round().astype(int):
                    cv2.rectangle(orig, tuple(box[:2]), tuple(box[2:]), (0,0,255))

                for box in targets[i].astype(int):
                    cv2.rectangle(actual, tuple(box[:2]), tuple(box[2:]), (255,0,0))

                cv2.imwrite('samples/sample_%d.jpg' % (batch_num + i), np.concatenate([actual, orig], axis=1))

        print('False positivies: %d, False negatives: %d, True positivies: %d, Precision: %f, Recall: %f' % 
            (false_positives, false_negatives, true_positives, float(true_positives)/(true_positives + false_positives), float(true_positives)/(true_positives + false_negatives)))

    pandas.DataFrame(results).to_csv('results.csv', index=False)
    return false_positives, false_negatives, true_positives, num_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Weight file to use')
    parser.add_argument('--test_set', default='../data/val_data.json', help='JSON file describing the test data')
    parser.add_argument('--thresh', default=0.5, type=float, help='Confidence threshold')
    args = parser.parse_args()
    
    data_dir = os.path.dirname(os.path.abspath(args.test_set))
    test_set = json.load(open(args.test_set))
    test_set = filter(lambda x: 'AOI' not in x['image_path'], test_set)

    print('Testing on %d images' % len(test_set))

    checkpoint = torch.load(args.weights)

    print('Best loss was %f' % checkpoint['best_loss'])

    num_classes = 2 # +1 background
    net = build_ssd('test', 300, num_classes, batch_norm = False).cuda()

    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    fp, fn, tp, N = test_net(net, test_set, data_dir, thresh=args.thresh)

    precision = float(tp)/(tp + fp)
    recall = float(tp) / (tp + fn)
    stats = {
        'false_negatives' : fn,
        'false_positives' : fp,
        'true_positives' : tp,
        'num_samples' : N,
        'precision' : precision,
        'recall' : recall,
        'thresh' : args.thresh
    }
    checkpoint['stats'] = stats

    torch.save(checkpoint, args.weights)


