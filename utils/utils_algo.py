import math
import pickle
import numpy as np
import copy
from scipy.special import comb
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_instancedependent_candidate_labels(model, train_X, train_Y):
    with torch.no_grad():
        k = int(torch.max(train_Y) - torch.min(train_Y) + 1)
        n = train_Y.shape[0]
        model = model.cuda()
        train_Y = torch.nn.functional.one_hot(train_Y, num_classes=k)
        avg_C = 0
        partialY_list = []
        rate, batch_size = 0.4, 2000
        step = math.ceil(n / batch_size)
        
        for i in range(0, step):
            
            b_end = min((i + 1) * batch_size, n)
    
            train_X_part = train_X[i * batch_size : b_end].cuda()
        
            outputs = model(train_X_part)
            
            train_p_Y = train_Y[i * batch_size : b_end].clone().detach()

            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(train_p_Y == 1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0
            
            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()
            
            train_p_Y[torch.where(z == 1)] = 1.0
            partialY_list.append(train_p_Y)

        partialY = torch.cat(partialY_list, dim=0).float()

        assert partialY.shape[0] == train_X.shape[0]
        
    avg_C = torch.sum(partialY) / partialY.size(0)
    
    return partialY



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


