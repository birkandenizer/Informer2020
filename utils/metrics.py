import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import  f1_score

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe

def classification_metric(pred, true):
    acc = accuracy_score(pred, true)
    f1 = f1_score(pred, true)

    return acc, f1

#torchvision implementation sigmoid_focal_loss()

#alternative facebook implementation
#https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

#source https://github.com/ashawkey/FocalLoss.pytorch
#https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075
class FocalLoss(nn.Module):
    '''
    Multi-class Focal loss implementation
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        print(f'size of input: {input.size()}, size of target: {target.size()}')
        logpt = F.log_softmax(input, dim=1)
        print(f'size of logpt: {logpt.size()}, size of target: {target.size()}')
        pt = torch.exp(logpt)
        print(f'size of pt: {pt.size()}, size of target: {target.size()}')
        logpt = (1-pt)**self.gamma * logpt
        print(f'size of logpt: {logpt.size()}, size of target: {target.size()}')
        loss = F.nll_loss(logpt, target, self.weight)
        print(f'size of loss: {loss.size()}, size of target: {target.size()}')
        return loss