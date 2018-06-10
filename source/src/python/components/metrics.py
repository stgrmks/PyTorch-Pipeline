_author__ = 'MSteger'

import torch
from sklearn import metrics
from torch import nn

def accuracy_score(y, yHat):
    yHat = yHat.max(dim = 1)[1]
    accuracy = torch.sum(y == yHat).double() / yHat.size(0)
    return accuracy.data.tolist() * float(1.)

def sk_accuracy_score(y, yHat):
    y = y.detach().cpu().numpy()
    yHat = yHat.max(dim = 1)[1].detach().cpu().numpy()
    return metrics.accuracy_score(y, yHat)

def sk_precision_score(y, yHat, **params):
    y = y.detach().cpu().numpy()
    yHat = yHat.max(dim = 1)[1].detach().cpu().numpy()
    return metrics.precision_score(y, yHat, **params)

def sk_f1_score(y, yHat, **params):
    y = y.detach().cpu().numpy()
    yHat = yHat.max(dim = 1)[1].detach().cpu().numpy()
    return metrics.f1_score(y, yHat, **params)

def sk_log_loss(y, yHat, **params):
    y = y.detach().cpu().numpy()
    yHat = yHat.max(dim = 1)[1].detach().cpu().numpy()
    return metrics.log_loss(y, yHat, **params)

def log_loss(y, yHat):
    ll = nn.CrossEntropyLoss().forward(yHat, y)
    return ll.item()

def fbeta_score(y, yHat, beta = 2, eps = 1e-9):
    y, yHat = y.float(), yHat.max(dim = 1)[1].float()
    TP = (y*yHat).sum(dim = 0)
    prec = TP.div(yHat.sum(dim = 0)).add(eps)
    rec = TP.div(y.sum(dim = 0)).add(eps)
    return torch.mean((prec*rec).div(prec.mul(beta**2) + rec + eps).mul(1 + beta**2)).data.tolist() * float(1.)

if __name__ == '__main__':
    print 'done'