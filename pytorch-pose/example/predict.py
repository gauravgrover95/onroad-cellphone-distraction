from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import _init_paths
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets
import pose.losses as losses

from custom_mpii import CustomMpii
from PIL import Image
import paint


# get model names and dataset names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))


# init global variables
best_acc = 0
idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

customMpiiObject = None

def main(args):
    global best_acc
    global idx
    global customMpiiObject

    # idx is the index of joints used to compute accuracy
    if args.dataset in ['mpii', 'lsp']:
        idx = [1,2,3,4,5,6,11,12,15,16]
    elif args.dataset == 'coco':
        idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    else:
        print("Unknown dataset: {}".format(args.dataset))
        assert False

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    njoints = CustomMpii.njoints

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks,
                                       num_blocks=args.blocks,
                                       num_classes=njoints,
                                       resnet_layers=args.resnet_layers)

    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = losses.JointsMSELoss().to(device)

    if args.solver == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )
    else:
        print('Unknown solver: {}'.format(args.solver))
        assert False

    # optionally resume from a checkpoint
    title = args.dataset + ' ' + args.arch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    print("Loading validation dataset...")
    customMpiiObject = CustomMpii(is_train=False, **vars(args))

    loss, acc, predictions = validate(args.unit_path, model, criterion, njoints,
                                        args.debug, args.flip)
    print(predictions)

    if args.debug:
        paint.main(args.unit_path, predictions.squeeze().tolist())
    
    return


def validate(unit_path, model, criterion, num_classes, debug=False, flip=True):
    losses = AverageMeter()
    acces = AverageMeter()

    dimensions = [0,0]
    scale_factor = 0
    with Image.open(unit_path) as img:
        dimensions = map(lambda x:x/2, img.size)
        scale_factor = (img.size[1])/200.0
        print(dimensions)
        print(scale_factor)

    # Custom edits [for testing]
    # dimensions = [459.0, 335.0]
    scale_factor = 6.0969 + 3
    # predictions
    predictions = torch.Tensor(1, num_classes, 2)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        input_temp, target_temp, meta = customMpiiObject.get_image_data(
            './data/custom/images/015620151.jpg',
            dimensions,
            scale_factor)

        input = torch.zeros([1, 3, 256, 256])
        target = torch.zeros([1, 16, 64, 64])
        target_weight = torch.zeros([1, 16, 1])

        input[0, :, :, :] = input_temp.to(device, non_blocking=True)
        target[0, :, :, :] = target.to(device, non_blocking=True)
        target_weight[0, :, :] = meta['target_weight'].to(device, non_blocking=True)

        # compute output
        output = model(input)
        score_map = output[-1].cpu() if type(output) == list else output.cpu()
        # if type(output) == list:  # multiple output
        #     loss = 0
        #     for o in output:
        #         loss += criterion(o, target, target_weight)
        #     output = output[-1]
        # else:  # single output
        #     loss = criterion(output, target, target_weight)

        acc = accuracy(score_map, target.cpu(), idx)

        meta_center_temp = torch.Tensor(meta['center'])
        meta['center'] = torch.zeros([1,2])
        meta['center'][0,:] = meta_center_temp
        
        # generate predictions
        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
        # for n in range(score_map.size(0)):
        #     predictions[meta['index'][n], :, :] = preds[n, :, :]
        predictions = preds
        acces.update(acc[0], input.size(0))
        
        # measure accuracy and record loss
        
    return losses.avg, acces.avg, predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='mpii',
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: mpii)')
    parser.add_argument('--image-path', default='', type=str,
                        help='path to images')
    parser.add_argument('--anno-path', default='', type=str,
                        help='path to annotation (json)')
    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')
    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                    help='output resolution (default: 64, to gen GT)')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('--resnet-layers', default=50, type=int, metavar='N',
                        help='Number of resnet layers',
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    # Training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--scale-factor', type=float, default=0.25,
                        help='Scale factor (data aug).')
    parser.add_argument('--rot-factor', type=float, default=30,
                        help='Rotation factor (data aug).')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint/mpii/hg_s8_b1', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='checkpoint/mpii/hg_s8_b1/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')

    # Self added
    parser.add_argument('--unit-path', default='', type=str,
                        help='path to test image', required=True)
    
    main(parser.parse_args())
