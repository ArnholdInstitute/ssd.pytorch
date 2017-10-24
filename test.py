#!/usr/bin/env python

from __future__ import print_function
import sys, os, argparse, torch, psycopg2, pdb, numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot, VOC_CLASSES as labelmap
from PIL import Image
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
from Dataset import RandomSampler
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--country', required=True, help='Which country to test on')
parser.add_argument('--thresh', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

class Scale:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def rgb(self, val):
        return (
            255 * (1.0 - val), # Blue
            0,
            255 * val
        )

def test_net(save_folder, net, generator, thresh, batch_size):
    while True:
        inputs, originals, meta = zip(*[next(generator) for _ in range(batch_size)])
        inputs = Variable(torch.stack(inputs, 0).cuda())

        y = net(inputs)      # forward pass
        detections = y.data.cpu().numpy()

        for i in range(len(detections)):
            dets = detections[i, 1]
            roff, coff, filename, whole_img = meta[i]
            orig = originals[i]

            dets[:, (1, 3)] = np.clip(dets[:, (1, 3)] * orig.shape[1], a_min=0, a_max = orig.shape[1])
            dets[:, (2, 4)] = np.clip(dets[:, (2, 4)] * orig.shape[2], a_min=0, a_max = orig.shape[0])

            valid_dets = dets[dets[:, 0] >= thresh, :]

            if len(valid_dets) == 0:
                continue

            scale = Scale(valid_dets[:, 0].min(), valid_dets[:, 0].max())

            for det in valid_dets:

                pdb.set_trace()


if __name__ == '__main__':
    # load net
    num_classes = 2 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.weights)['state_dict'])
    net.eval()
    print('Finished loading model!')
    # load data

    conn = psycopg2.connect(
        dbname='aigh',
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', ''),
        password=os.environ.get('DB_PASSWORD', '')
    )

    gen = RandomSampler(conn, args.country, BaseTransform(net.size, (104, 117, 123)))

    net = net.cuda()
    cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, gen, args.thresh, 32)
























