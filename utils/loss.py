# -*- coding: utf-8 -*-


import torch

from torch import nn
from torch.nn.functional import max_pool3d


class crossentropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth))


class B_crossentropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth) + (1 - y_true) * torch.log(1 - y_pred + smooth))

class partial_entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        smooth = 1e-6
        count = torch.sum(torch.where(y_true>0, 1.0, 0.0))
        return -torch.sum(y_true * torch.log(y_pred + smooth))/(1+count)

class dice_coef(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1.
        size = y_true.shape[-1] // y_pred.shape[-1]
        y_true = max_pool3d(y_true, size, size)
        a = torch.sum(y_true * y_pred, (2, 3, 4))
        b = torch.sum(y_true, (2, 3, 4))
        c = torch.sum(y_pred, (2, 3, 4))
        dice = (2 * a) / (b + c + smooth)
        return torch.mean(dice)