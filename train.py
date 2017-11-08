#!/usr/bin/env python

import os, torch, json, time, pdb, numpy as np, cv2
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, detection_collate, BaseTransform
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from Dataset import Dataset
from datetime import datetime
from torch.utils.data import DataLoader

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
args = parser.parse_args()

TIMESTAMP = datetime.now().isoformat()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = v2

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# train_sets = 'train'
ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # only support voc now
num_classes = 2
batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = 120000
weight_decay = 0.0005
stepvalues = (50000, 100000, 120000)
gamma = 0.1
momentum = 0.9

net = build_ssd('train', 300, num_classes, batch_norm = False).cuda()
cudnn.benchmark = True

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

checkpoint = None
if args.resume:
    checkpoint = torch.load(args.resume)
    print('Resuming training, loading {}...'.format(args.resume))
    net.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    args.start_iter = checkpoint['epoch']
    TIMESTAMP = checkpoint['timestamp'] if 'timestamp' in checkpoint else TIMESTAMP
else:
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    net.vgg.load_state_dict(vgg_weights)
    tuneable = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.SGD(tuneable, lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

def mk_checkpoint(net, optim, checkpoint_name, epoch, best_loss):
    state_dict = net.state_dict()
    for key in state_dict.keys():                                                                                                                                                                                
        state_dict[key] = state_dict[key].cpu()                                                                                                                                                                                         
    torch.save({                                                                                                                                                                                                 
        'epoch': epoch,                                                                                                                                                                                     
        'state_dict': state_dict,
        'best_loss' : best_loss,
        'timestamp' : TIMESTAMP,                                                                                                                                                                            
        'optimizer': optim},                                                                                                                                                                                     
        checkpoint_name)

POS_NEG_RATIO = 3
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, POS_NEG_RATIO, 0.5, False, args.cuda)

def check_gradients(net):
    for name, p in net.named_parameters():
        print('%s: min: %f, max: %f, median: %f, mean: %f, std: %f' % 
                (name, p.grad.min().data[0], p.grad.max().data[0], p.grad.median().data[0], 
                    p.grad.mean().data[0], p.grad.std().data[0]))

def train():
    best_loss = checkpoint['best_loss'] if checkpoint and 'best_loss' in checkpoint else float('inf')
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    train_data = json.load(open('../data/train_data.json')) + json.load(open('../data/rio_train.json'))
    dataset = Dataset('../data', train_data, transform=SSDAugmentation(ssd_dim, means)).even()

    val_data = json.load(open('../data/val_data.json')) + json.load(open('../data/rio_val.json'))
    val_dataset = Dataset('../data', val_data, transform=BaseTransform(300, (104, 117, 123)))
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, collate_fn=detection_collate)


    epoch_size = len(dataset) // args.batch_size
    step_index = 0
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, #num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)

    scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-8, verbose=True)

    net.train()
    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        # img_data = images.numpy()
        # for i, img in enumerate(img_data):
        #     img = img.transpose((1,2,0))[:,:,(2,1,0)]
        #     img = ((img - img.min()) / img.max() * 255).astype('uint8').copy()

        #     for box in targets[i].numpy():
        #         box = (box * 300).round().astype(int)
        #         cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), (0,0,255))

        #     cv2.imwrite('samples/sample_%d.jpg' % i, img)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        print('Timer: %f sec || iter %d || Loss: %.4f, loss_l: %.4f, loss_c: %.4f' % 
            ((t1 - t0), iteration, loss.data[0], loss_l.data[0], loss_c.data[0]))

        if iteration % 100 == 0 and iteration != args.start_iter:
            # validate
            test_loss = 0
            N = len(val_loader)
            net.eval()
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = Variable(inputs.cuda(), volatile=True)
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
                out = net(inputs)
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                print('[%d/%d]: loss: %f, loss_l: %f, loss_c: %f' % (i, N, loss.data[0], loss_l.data[0], loss_c.data[0]))

                test_loss += loss_l.data[0] + loss_c.data[0]

            net.train()
            scheduler.step(test_loss)

            print('test_loss = %f, best_loss = %f' % (test_loss, best_loss))
            if test_loss < best_loss:
                print('Saving state, iter: %d, best_loss = %f' % (iteration, test_loss))
                best_loss = test_loss
                mk_checkpoint(net, optimizer, 'weights/ssd300_0712_' + TIMESTAMP + '.pth', iteration, test_loss)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
