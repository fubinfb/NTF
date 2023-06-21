"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from functools import partial
import torch.nn as nn
from base.modules import ResBlock, Flatten


class AuxClassifier(nn.Module):
    def __init__(self, in_shape, num_s, num_c):
        super().__init__()
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero", dropout=0.3)

        # C = in_shape[0]
        C = 128
        self.layers = nn.Sequential(
            ResBlk(C, C*2, 3, 1, downsample=True),
            ResBlk(C*2, C*2, 3, 1),
            nn.AdaptiveAvgPool2d(1),
            Flatten(1),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({"style": nn.Linear(C*2, num_s), "comp": nn.Linear(C*2, num_c)})

    def forward(self, x):
        feat = self.layers(x)

        logit_s = self.heads["style"](feat)
        logit_c = self.heads["comp"](feat)

        return logit_s, logit_c


class AuxClassifierv2(nn.Module):
    def __init__(self, in_shape_s, in_shape_c, num_s, num_c):
        super().__init__()
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero", dropout=0.3)

        # C = in_shape[0]
        # C = 128
        self.layers_s = nn.Sequential(
            ResBlk(in_shape_s, 32, 3, 1, downsample=True),
            ResBlk(32, 128, 3, 1),
            nn.AdaptiveAvgPool2d(1),
            Flatten(1),
            nn.Dropout(0.2),
        )
        C = 128
        self.layers_c = nn.Sequential(
            ResBlk(in_shape_c, C*2, 3, 1, downsample=True),
            ResBlk(C*2, C*2, 3, 1),
            nn.AdaptiveAvgPool2d(1),
            Flatten(1),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({"style": nn.Linear(C, num_s), "comp": nn.Linear(C*2, num_c)})

    def forward(self, feat_s, feat_c):
        n,c,h,w = feat_c.shape
        feat_s = feat_s.view(n,3,1,1).expand(n,3,h,w)
        feat_s = self.layers_s(feat_s)
        feat_c = self.layers_c(feat_c)

        logit_s = self.heads["style"](feat_s)
        logit_c = self.heads["comp"](feat_c)

        return logit_s, logit_c
