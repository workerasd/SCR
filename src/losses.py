import torch
from torch import nn
import torch.nn.functional as F

import math

from models import Discriminator

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def feat_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class SRC(torch.nn.Module):
    def __init__(self,args):
        super(SRC, self).__init__()
        self.args = args
    def svd_std(self,feats, labels):
        B = feats.size(0)
        feat_dist_map = feat_dist(feats, feats)
        feat_dist_map = feat_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten()
        if len(labels.size())==1:
            labels = labels.view(-1,1)
        if labels.size(1) == 1:
            label_dist_map = (labels.unsqueeze(1) - labels.unsqueeze(0)).mean(dim=2)
        else:
            label_dist_map = feat_dist(labels, labels)
        label_dist_map = label_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten()
        label_dist_map = torch.abs(label_dist_map).clamp(min=1e-12)
        mask = label_dist_map.gt(1e-12)
        dist_map = feat_dist_map / label_dist_map
        ret_dist_map = dist_map.view(B,B-1)
        dist_map = dist_map[mask]
        loss_std = torch.std(dist_map)
        _,s,_ = torch.linalg.svd(feats)
        # loss_svd = torch.pow(s[0], 2)
        return loss_std, s, ret_dist_map



class RegMetricLoss(nn.Module):
    def __init__(self, sigma=1, w_leak=0.2, p=0.9):
        super(RegMetricLoss, self).__init__()
        self.sigma = sigma
        #         self.c = (1/(sigma * np.sqrt(2*np.pi))).tolist()
        self.w_leak = w_leak
        self.alpha = None
        self.p = p
        print('RMLoss: sigma=%f, p=%f' % (self.sigma, self.p))

    def forward(self, feature, label):
        f2 = (feature ** 2).sum(1)
        f_dist = f2.unsqueeze(0) + f2.unsqueeze(
            1) - 2 * torch.matmul(feature, feature.transpose(0, 1))
        f_dist = torch.sqrt(F.relu(f_dist))
        # label_0 = label.unsqueeze(0)
        # label_1 = label.unsqueeze(1)
        if len(label.size())==1:
            label = label.view(-1,1)
        l_dist = feat_dist(label , label)
        w = (torch.exp(-((l_dist / self.sigma) ** 2) / 2) + self.w_leak).detach()
        diff = torch.abs(f_dist - l_dist)
        w_diff = w * diff

        mean_diff = w_diff.mean().data.cpu().numpy().tolist()
        if self.alpha is None:
            self.alpha = mean_diff
        else:
            self.alpha = self.p * self.alpha + (1 - self.p) * mean_diff
        mask = w_diff.gt(self.alpha)
        nonz_num = mask.sum().float()
        #         nonz_diff = torch.masked_select(diff, mask)
        corr_w = torch.masked_select(w, mask)
        #         loss = corr_w*nonz_diff
        loss = torch.masked_select(w_diff, mask)
        #         print(self.alpha, mean_diff, nonz_num, corr_w.sum())
        return loss.sum() / (corr_w.sum() + 1e-9), nonz_num, self.alpha, mean_diff
