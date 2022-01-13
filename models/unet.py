import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn.layers import *


class UNet3D(nn.Module):
    def __init__(self, n_classes, n_channels=[4, 16, 32, 64, 128], bayes=False,
                 devices=None, shorten=False):
        super(UNet3D, self).__init__()
        self.bayes = bayes
        self.devices = devices
        self.shorten = shorten
        if bayes:
            self.init_conv = BayesConv3d(n_channels[0], n_channels[1], kernel_size=3,
                                         padding=1, bias=False)
        else:
            self.init_conv = nn.Conv3d(n_channels[0], n_channels[1], kernel_size=3,
                                       padding=1, bias=False)

        self.down1 = BasicDownBlock(n_channels[1], n_channels[2], downsample=True,
                                    bayes=bayes)
        self.down2 = BasicDownBlock(n_channels[2], n_channels[2], downsample=False,
                                    bayes=bayes)
        self.down3 = BasicDownBlock(n_channels[2], n_channels[3], downsample=True,
                                    bayes=bayes)
        self.down4 = BasicDownBlock(n_channels[3], n_channels[3], downsample=False,
                                    bayes=bayes)
        self.down5 = BasicDownBlock(n_channels[3], n_channels[4], downsample=True,
                                    bayes=bayes)
        self.down6 = BasicDownBlock(n_channels[4], n_channels[4], downsample=False,
                                    bayes=bayes)

        if not self.shorten:
            self.down7 = BasicDownBlock(n_channels[4], n_channels[4], downsample=False,
                                        bayes=bayes)
            self.down8 = BasicDownBlock(n_channels[4], n_channels[4], downsample=False,
                                        bayes=bayes)
            self.down9 = BasicDownBlock(n_channels[4], n_channels[4], downsample=False,
                                        bayes=bayes)

        self.up1 = BasicUpBlock(n_channels[4], n_channels[3], bayes=bayes)
        self.up2 = BasicUpBlock(n_channels[3], n_channels[2], bayes=bayes)
        self.up3 = BasicUpBlock(n_channels[2], n_channels[1], bayes=bayes)
        self.out = nn.Conv3d(n_channels[1], n_classes, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input):
        if self.devices is not None:
            input = input.to(self.devices[0])

        x1 = self.init_conv(input)
        x2 = self.down1(x1)
        x2 = self.down2(x2)
        x3 = self.down3(x2)

        x3 = self.down4(x3)
        x4 = self.down5(x3)
        x4 = self.down6(x4)
        if not self.shorten:
            x4 = self.down7(x4)
            x4 = self.down8(x4)
            x4 = self.down9(x4)

        if self.devices is not None:
            x4 = x4.to(self.devices[1])
            x1 = x1.to(self.devices[1])
            x2 = x2.to(self.devices[1])
            x3 = x3.to(self.devices[1])

        x4 = self.up1(x4, x3)
        x4 = self.up2(x4, x2)
        x4 = self.up3(x4, x1)
        return self.softmax(self.out(x4))

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')