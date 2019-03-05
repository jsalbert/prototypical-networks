import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import pprint


from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import AverageMeter, compute_accuracy, euclidean_dist, mkdir, weights_init_xavier, accuracy_top_k
from models.identity import Identity
from models.convnet_mini import ConvNet
from dataloaders.mini_imagenet_loader import MiniImageNet
from samplers.episodic_batch_sampler import EpisodicBatchSampler


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.append('default_convnet')

parser = argparse.ArgumentParser(description='Pytorch Prototypical Networks Testing')
parser.add_argument('--train_dir', type=str, help='path to training data (default: none)')
parser.add_argument('--test_dir', type=str, metavar='train_dir', help='path to validation data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--evaluation_name', type=str, help='Evaluation name')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--cpu', default=False, action='store_true', help='CPU mode')
parser.add_argument('--checkpoint', type=str, help='model checkpoint path')
parser.add_argument('--results_name', type=str, help='name of the results csv')
parser.add_argument('--n_episodes', default=None, type=int, help='Number of episodes to average')
parser.add_argument('--n_way', default=None, type=int, help='Number of classes per episode')
parser.add_argument('--n_support', default=None, type=int, help='Number of support samples per class')
parser.add_argument('--n_query', default=None, type=int, help='Number of query samples')


def main():
    args = parser.parse_args()

    global results_path
    results_path = os.path.join('evaluations', args.evaluation_name)
    mkdir(results_path)
    options = vars(args)
    save_options_dir = os.path.join(results_path, 'options.txt')

    with open(save_options_dir, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(options.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    printer = pprint.PrettyPrinter()
    printer.pprint(options)

    # Create model
    print("=> creating model '{}'".format(args.arch))

    if args.arch == 'default_convnet':
        model = ConvNet()
    else:
        model = models.__dict__[args.arch]()

        if args.out_dim is not None:
            lin = nn.Linear(model.fc.in_features, args.out_dim)
            weights_init_xavier(lin)
            model.fc = lin
        else:
            model.fc = Identity()

    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' )".format(args.checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

    if not args.cpu:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    model.eval()

    cudnn.benchmark = True

    # Testing data
    test_dataset = MiniImageNet('test')

    test_sampler = EpisodicBatchSampler(test_dataset.labels, args.n_episodes, args.n_way, args.n_support + args.n_query)
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler,
                             num_workers=args.workers, pin_memory=True)

    test(test_loader, model, args)


def test(test_loader, model, args):
    print('Testing...')
    losses = AverageMeter()
    accuracy = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for n_episode, batch in enumerate(test_loader, 1):
            data, _ = [_.cuda(non_blocking=True) for _ in batch]
            p = args.n_support * args.n_way
            data_support, data_query = data[:p], data[p:]

            # Compute class prototypes (n_way, output_dim)
            class_prototypes = model(data_support).reshape(args.n_support, args.n_way, -1).mean(dim=0)

            # Generate labels (n_way, n_query)
            labels = torch.arange(args.n_way).repeat(args.n_query)
            labels = labels.type(torch.cuda.LongTensor)

            # Compute loss and metrics
            logits = euclidean_dist(model(data_query), class_prototypes)
            loss = F.cross_entropy(logits, labels)
            acc = compute_accuracy(logits, labels)

            # Record loss and accuracy
            losses.update(loss.item(), data_query.size(0))
            accuracy.update(acc, data_query.size(0))

        print('Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Test Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'.format(loss=losses, accuracy=accuracy))

    return losses.avg, accuracy.avg


if __name__ == '__main__':
    main()
