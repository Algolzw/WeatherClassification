import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lovasz_losses import *

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 weight=None,
                 use_focal_loss=False
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.weight = weight
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss2(weight=weight, balance_param=0.5)
            # self.f1_loss = F1Loss()
            self.triplet_loss = TripletLoss(0.3)
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        if self.use_focal_loss:
            floss = self.focal_loss(logits, label)
            # f1loss = self.f1_loss(logits, label)
            # lovasz_loss = lovasz_softmax(torch.softmax(logits, dim=1), label, classes='all')
            triplet_loss = self.triplet_loss(logits, label)

        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.weight is not None:
            sum_loss = -torch.sum(torch.sum((logs*label)*self.weight, dim=1))
        else:
            sum_loss = -torch.sum(torch.sum(logs*label, dim=1))

        if self.reduction == 'mean':
            loss = sum_loss / n_valid
        elif self.reduction == 'sum':
            loss = sum_loss
        if self.use_focal_loss:
            loss = 1.*loss + 0.4*floss
             # + 0.4*triplet_loss
        return loss

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=1.,
                 gamma=2,
                 reduction='sum',
                 ignore_lb=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label[ignore] = 0

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = torch.ones_like(logits)
        mask[[a, torch.arange(mask.size(1)), *b]] = 0

        # compute loss
        probs = torch.sigmoid(logits)
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        pt = torch.where(lb_one_hot == 1, probs, 1-probs)
        alpha = self.alpha*lb_one_hot + (1-self.alpha)*(1-lb_one_hot)
        loss = -alpha*((1-pt)**self.gamma)*torch.log(pt + 1e-12)
        loss[mask == 0] = 0
        if self.reduction == 'mean':
            loss = loss.sum(dim=1).sum()/n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class FocalLoss2(nn.Module):

    def __init__(self, weight=None, focusing_param=2, balance_param=0.25):
        super(FocalLoss2, self).__init__()
        self.weight = weight
        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):

        cross_entropy = F.cross_entropy(output, target, weight=self.weight, reduction='sum')
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target, weight=self.weight, reduction='sum')
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


class F1Loss(nn.Module):
    def __init__(self):
        super(F1Loss, self).__init__()

    def forward(self, predict, targets):
        return self.f1_loss(predict, targets)

    def f1_loss(self, predict, target):
        batch_size = predict.size(0)
        target = target.view(batch_size, 1)
        target = torch.zeros(batch_size, 9).cuda().scatter_(1, target, 1)
        predict = torch.sigmoid(predict)
        # print(predict.size(), target.size())
        predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
        tp = predict * target
        tp = tp.sum(dim=0)
        precision = tp / (predict.sum(dim=0) + 1e-8)
        recall = tp / (target.sum(dim=0) + 1e-8)
        f1 = 2 * (precision * recall / (precision + recall + 1e-8))
        return 1 - f1.mean()

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss



