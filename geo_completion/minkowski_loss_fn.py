import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np

def minkowski_masked_l1_loss(x, y, mask):
    diff = x - y
    assert(diff.coordinate_map_key == mask.coordinate_map_key)
    select = mask.F > 0.5
    if len(select.shape) == 2:
        select = select[:, 0]
    else:
        assert(len(select.shape) == 1)
    return torch.abs(diff.F[select]).mean()

def minkowski_masked_l1_loss_rgba(x, y, mask):
    diff = x - y
    num_c4 = diff.F.shape[1] // 4
    assert(diff.coordinate_map_key == mask.coordinate_map_key)
    select = mask.F > 0.5
    if len(select.shape) == 2:
        select = select[:, 0]
    else:
        assert(len(select.shape) == 1)
    loss_alpha = torch.abs(diff.F[select,:num_c4]).mean() * 0.25 + torch.abs(diff.F[:,:num_c4]).mean() * 0.05
    loss_rgb = torch.abs(diff.F[select,num_c4:]).mean() + torch.abs(diff.F[:,num_c4:]).mean() * 0.2
    return loss_alpha, loss_rgb

def minkowski_hinge_loss(x, mode):
    if mode == -1: # larger better
        return -torch.mean(x.F)
    elif mode == 0: # smaller <=-1 better
        return torch.mean(F.relu(1 + x.F, inplace=False))
    elif mode == 1: # larger >=1 better
        return torch.mean(F.relu(1 - x.F, inplace=False))
    else:
        assert(False)
        return 0

def minkowski_binary_loss(x, mode):
    if mode == True:
        return nn.BCELoss()(x.F, x.F.detach() * 0.0 + 1.0)
    elif mode == False:
        return nn.BCELoss()(x.F, x.F.detach() * 0.0)
    else:
        assert(False)
        return 0

def minkowski_binary_loss_with_logits(x, mode):
    assert(x.F.shape[1] == 2)
    mask = x.F[:,1] > 0.5
    if mode == True:
        l = nn.BCEWithLogitsLoss()(x.F[mask,0], x.F[mask,0].detach() * 0.0 + 1.0)
        l += nn.BCEWithLogitsLoss()(x.F[:,0], x.F[:,0].detach() * 0.0 + 1.0)
        return l / 2.0
    elif mode == False:
        l = nn.BCEWithLogitsLoss()(x.F[mask,0], x.F[mask,0].detach() * 0.0)
        l += nn.BCEWithLogitsLoss()(x.F[:,0], x.F[:,0].detach() * 0.0)
        return l / 2.0
    else:
        assert(False)
        return 0