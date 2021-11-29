import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple

from .bayes_conv import BayesConv3d, BayesConv2d


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=1, bayes = False):
        super(ConvBlock, self).__init__()
        if bayes:
            self.conv = nn.Sequential(
                nn.InstanceNorm3d(in_channels),
                nn.ReLU(inplace=True),
                BayesConv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False))
        else:
            self.conv = nn.Sequential(
                nn.InstanceNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False))

    def forward(self, x):
        x = self.conv(x)
        return x


class BasicDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample, bayes=False):
        super(BasicDownBlock, self).__init__()
        if downsample:
            str = 2
        else:
            str = 1

        self.conv_1 = ConvBlock(in_ch, out_ch, kernel=3, stride=str, bayes=bayes)
        self.conv_2 = ConvBlock(out_ch, out_ch, kernel=3, stride=1, bayes=bayes)

        self.down = None
        if downsample:
            self.down = ConvBlock(in_ch, out_ch, kernel=1, stride=2, padding=0, bayes=False)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.conv_2(x)
        if self.down is not None:
            return x + self.down(inp)
        else:
            return x + inp


class BasicUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bayes=False):
        super(BasicUpBlock, self).__init__()

        self.upsample = nn.Sequential(
            ConvBlock(in_ch, out_ch, kernel=1, stride=1, padding=0, bayes=False),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        self.conv_1 = ConvBlock(out_ch, out_ch, kernel=3, stride=1, bayes=bayes)
        self.conv_2 = ConvBlock(out_ch, out_ch, kernel=3, stride=1, bayes=bayes)

    def forward(self, inp, skip_connection=None):
        x = self.upsample(inp)
        if skip_connection is not None:
            x = x + skip_connection
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)
        return x1 + x