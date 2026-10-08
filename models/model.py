import torch
import torch.nn.functional as F
import torch.nn as nn
from models.common import *
import random

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from torchvision import transforms
from torchvision.utils import save_image


class Mynet(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(Mynet, self).__init__()
        self.conv_rgb1 = nn.Conv2d(in_channels=3, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        self.rgb_rb2 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rgb_w2 = SMM(num_feats)
        self.rgb_rb3 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rgb_w3 = SMM(num_feats)
        self.rgb_rb4 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rgb_w4 = SMM(num_feats)

        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg2 = ResidualGroup(default_conv, 64, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg3 = ResidualGroup(default_conv, 96, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg4 = ResidualGroup(default_conv, 128, kernel_size, reduction=16, n_resblocks=4)

        self.bridge1 = FCM(dp_feats=32, rgb_feats=32, scale=scale)
        self.bridge2 = FCM(dp_feats=64, rgb_feats=32, scale=scale)
        self.bridge3 = FCM(dp_feats=96, rgb_feats=32, scale=scale)



        my_tail = [
            ResidualGroup(
                default_conv, 128, kernel_size, reduction=16, n_resblocks=4),
            ResidualGroup(
                default_conv, 128, kernel_size, reduction=16, n_resblocks=4)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(128, 128, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(128, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        image, depth = x

        dp_in = self.act(self.conv_dp1(depth))
        dp1 = self.dp_rg1(dp_in)

        rgb1 = self.act(self.conv_rgb1(image))
        rgb2 = self.rgb_rb2(rgb1)
        rgb3 = self.rgb_w2(rgb2)
        ca1_in = self.bridge1(dp1, rgb3)

        dp2 = self.dp_rg2(ca1_in)

        rgb4 = self.rgb_rb3(rgb3)
        rgb5 = self.rgb_w3(rgb4)
        ca2_in = self.bridge2(dp2, rgb5)

        dp3 = self.dp_rg3(ca2_in)

        rgb6 = self.rgb_rb4(rgb5)
        rgb7 = self.rgb_w4(rgb6)
        ca3_in = self.bridge3(dp3, rgb7)

        dp4 = self.dp_rg4(ca3_in)

        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))

        out = out + self.bicubic(depth)

        return out