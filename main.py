import os
import time
import random
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from utils.model import ABLE
from utils.resnet import *
from utils.utils_algo import *
from utils.utils_loss import ClsLoss, ConLoss
from utils.mnist import load_mnist
from utils.fmnist import load_fmnist
from utils.kmnist import load_kmnist
from utils.cifar10 import load_cifar10


parser = argparse.ArgumentParser(
    prog='ABLE demo file.',
    usage='Code for "Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning" in IJCAI-ECAI 2022.',
    description='PyTorch implementation of ABLE.',
    epilog='end',
    add_help=True
)

parser.add_argument('--dataset', default='cifar10', \
                    help='dataset name', type=str, \
                    choices=['mnist', 'fmnist', 'kmnist', 'cifar10'])

parser.add_argument('--data-dir', default='./data', type=str)

parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')

parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')

parser.add_argument('--pmodel_path', default='./pmodel/cifar10.pt', type=str,
                    help='pretrained model path for generating instance dependent partial labels')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', \
                    choices=['resnet18'],
                    help='network architecture (only resnet18 used)')

parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')

parser.add_argument('--temperature', type=float, default=0.07,
                    help='temperature for loss function')

parser.add_argument('--loss_weight', default=1.0, type=float,
                    help='contrastive loss weight')

parser.add_argument('--cuda_VISIBLE_DEVICES', default='0', type=str, \
                    help='which gpu(s) can be used for distributed training')

parser.add_argument('--epochs', default=500, type=int, 
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')

parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_VISIBLE_DEVICES


def main():

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
    
    print("=> creating ABLE_model '{}'\n".format(args.arch))

    model = ABLE(args=args, base_encoder=ABLENet)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
         
    if args.dataset == 'mnist':
        train_loader, train_partialY_matrix, test_loader = load_mnist(args)

    elif args.dataset == 'fmnist':
        train_loader, train_partialY_matrix, test_loader = load_fmnist(args)

    elif args.dataset == 'kmnist':
        train_loader, train_partialY_matrix, test_loader = load_kmnist(args)

    elif args.dataset == 'cifar10':
        train_loader, train_partialY_matrix, test_loader = load_cifar10(args)
    

    tempY = train_partialY_matrix.sum(dim=1).unsqueeze(1).repeat(1, train_partialY_matrix.shape[1])
    uniform_confidence = train_partialY_matrix.float() / tempY
    uniform_confidence = uniform_confidence.cuda()
    
    loss_cls = ClsLoss(predicted_score=uniform_confidence)
    loss_con = ConLoss(predicted_score=uniform_confidence)
    
    best_acc = 0.0
    
    for epoch in range(0, args.epochs):
        
        print('\nEpoch[{}] starts...'.format(epoch))
        
        adjust_learning_rate(args, optimizer, epoch)
        
        acc_train_cls = train(args, epoch, loss_cls, loss_con, train_loader, model, optimizer)
        
        acc_test = test(args, epoch, test_loader, model)
        
        if acc_test > best_acc:
            best_acc = acc_test
        
        print('Epoch[{}] ends => Training_Acc {}, Testing_Acc {}, Best_Acc {}.\n'.format(epoch, acc_train_cls.avg, acc_test, best_acc))
        

def train(args, epoch, loss_cls, loss_con, train_loader, model, optimizer):
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    
    model.train()
    
    for idx, (images_w, images_s, labels, true_labels, index) in enumerate(train_loader): 
        
        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()
        X_tot = torch.cat([X_w, X_s], dim=0)
        batch_size = args.batch_size
        
        cls_out, features = model(args=args, images=X_tot, partial_Y=Y, is_eval=False)
        cls_out_w = cls_out[0 : batch_size, :]
        
        cls_loss = loss_cls(cls_out_w, index)
        con_loss, new_target = loss_con(args, cls_out, features, Y, index)
        loss = cls_loss + args.loss_weight * con_loss

        loss_cls.update_target(batch_index=index, updated_confidence=new_target)
        loss_con.update_target(batch_index=index, updated_confidence=new_target)
        
        acc = accuracy(cls_out_w, Y_true)[0]
        acc_cls.update(acc[0])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return acc_cls


def test(args, epoch, test_loader, model):    
    with torch.no_grad():      
        model.eval()   
        top1_acc = AverageMeter("Top1")
        
        for batch_idx, (images, labels) in enumerate(test_loader): 
            images, labels = images.cuda(), labels.cuda()
            outputs = model(args=args, img_w=images, is_eval=True)    
            acc1, _ = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])

    return top1_acc.avg


if __name__ == '__main__':
    main()
