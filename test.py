from models.identity import Identity
from utils import AverageMeter, accuracy_top_k, euclidean_dist, mkdir, weights_init_xavier
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pprint
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.utils.data
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Prototypical Networks Testing')
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
parser.add_argument('--subtract_mean', type=float, default=(0.5, 0.5, 0.5), nargs=3,
                    help='Substract mean of image channels when loading the data')
parser.add_argument('--subtract_std', type=float, default=(0.5, 0.5, 0.5), nargs=3,
                    help='Divide by std of image channels when loading the data')
parser.add_argument('-s', '--image_size', default=224, type=int, help='Image Size to load images')
parser.add_argument('--checkpoint', type=str, help='model checkpoint path')
parser.add_argument('--results_name', type=str, help='name of the results csv')
parser.add_argument('--save_prototypes', action='store_true', help='If true will save prototypes')
parser.add_argument('--load_prototypes', action='store_true', help='If true will load prototypes')
parser.add_argument('--out_dim', default=None, type=int, help='Output embedding dimension')


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

    global printer
    printer = pprint.PrettyPrinter()
    printer.pprint(options)
    # Create model
    print("=> creating model '{}'".format(args.arch))
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

    # Data loading code
    mean = np.array(args.subtract_mean)
    std = np.array(args.subtract_std)

    if mean[0] > 1 or mean[1] > 1 or mean[2] > 1:
        print('One or more of the subtract mean values were above 1, dividing by 255...')
        mean /= 255

    if std[0] > 1 or std[1] > 1 or std[2] > 1:
        print('One or more of the subtract std values were above 1, dividing by 255...')
        std /= 255

    print('Normalizing by mean of %.4f, %.4f, %.4f' % (mean[0], mean[1], mean[2]))
    print('Normalizing by std of %.4f, %.4f, %.4f' % (std[0], std[1], std[2]))

    # normalize = transforms.Normalize(mean=mean, std=std)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Training data
    train_directory = args.train_dir
    train_dataset = ImageFolder(
        train_directory,
        transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True, shuffle=False)

    classes = train_dataset.classes

    # Testing data
    testing_directory = args.test_dir
    test_dataset = ImageFolder(testing_directory, transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.workers,
                             pin_memory=True, shuffle=False)

    class_prototypes = compute_prototypes(train_loader, model, classes, args.load_prototypes, args.save_prototypes)
    class_prototypes = torch.stack([values for key, values in class_prototypes.items()])
    evaluation(test_loader, class_prototypes, model, classes, args)


def compute_prototypes(data_loader, model, classes, load=False, save=False):
    class_prototypes = {n: [] for n in range(len(classes))}
    if load:
        print('Loading prototypes from ', os.path.join(results_path, 'prototypes'))
        filenames = sorted(os.listdir(os.path.join(results_path, 'prototypes')))
        for i, filename in enumerate(filenames):
            if filename.endswith('.pt'):
                class_prototypes[i] = torch.load(os.path.join(results_path, 'prototypes', filename))
        if len(class_prototypes) != len(classes):
            raise ValueError('There are not the same number of prototypes as classes')
    else:
        with torch.no_grad():
            print('Computing class prototypes...')
            for data, targets in data_loader:
                data = data.cuda(non_blocking=True)
                outputs = model(data)
                for i, output in enumerate(outputs):
                    class_prototypes[targets[i].item()].append(output)

            for key, values in class_prototypes.items():
                class_prototypes[key] = torch.stack(values).mean(dim=0)
                if save:
                    mkdir(os.path.join(results_path, 'prototypes'))
                    torch.save(class_prototypes[key], os.path.join(results_path, 'prototypes',
                                                                   classes[key] + '_prototype.pt'))

    return class_prototypes


def evaluation(data_loader, class_prototypes, model, classes, args):
    queries_data = {n: [] for n in range(len(classes))}
    losses = AverageMeter()
    accuracy_1 = AverageMeter()
    accuracy_5 = AverageMeter()
    metrics_class = {}
    with torch.no_grad():
        print('Forwarding queries...')
        for data, targets in data_loader:
            data = data.cuda(non_blocking=True)
            outputs = model(data)
            for i, output in enumerate(outputs):
                queries_data[targets[i].item()].append(output)
        mean_accuracy = []
        for key, values in queries_data.items():
            if len(values) > 0:
                logits = euclidean_dist(torch.stack(values), class_prototypes)
                labels = torch.Tensor([key]).repeat(len(values)).type(torch.cuda.LongTensor)
                loss = F.cross_entropy(logits, labels).item()
                acc = accuracy_top_k(logits, labels, top_k=(1, 2, 3, 4, 5))
                acc = [a.item() for a in acc]
                metrics_class[key] = {'class_name': classes[key], 'accuracy': acc[0],
                                      'loss': loss, 'n_samples': len(values)}
                # Record loss and accuracy
                losses.update(loss, len(values))
                accuracy_1.update(acc[0], len(values))
                accuracy_5.update(acc[4], len(values))
                mean_accuracy.append(acc[0])
                # print('Class ' + classes[key])
                printer.pprint(metrics_class[key])
            else:
                metrics_class[key] = {'class_name': classes[key], 'accuracy': np.nan,
                                      'loss': np.nan, 'n_samples': 0}
                # print('Class ' + classes[key] + ' is empty')

    mean_accuracy = sum(mean_accuracy) / len(mean_accuracy)

    print('Total Weighted Top-1 Accuracy %.4f\n'
          'Total Weighted Top-5 Accuracy %.4f\n'
          'Mean Class Top-1 Accuracy %.4f\n'
          'Total Avg Loss %.4f' % (accuracy_1.avg, accuracy_5.avg, mean_accuracy, losses.avg))

    class_metrics_df = pd.DataFrame.from_dict(metrics_class, 'index')
    class_metrics_df.round(4)
    class_metrics_df.to_csv(os.path.join(results_path, args.results_name + '_individual.csv'), index=False,
                            float_format='%.4f')

    average_df = pd.DataFrame({'accuracy_top_1': [accuracy_1.avg],
                               'accuracy_top_5': [accuracy_5.avg],
                               'sensitivity': [mean_accuracy]})
    average_df.round(4)
    average_df.to_csv(os.path.join(results_path, args.results_name + '_average.csv'), index=False, float_format='%.4f')


if __name__ == '__main__':
    main()
