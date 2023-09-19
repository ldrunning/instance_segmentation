# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import time


# from .common import ShiftMean


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.8):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats, n_feats * expansion_ratio, kernel_size=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=1)),
            weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        return x + self.module(x) * self.res_scale


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = double_conv(out_ch, out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.pool = nn.AvgPool2d(2, 2)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        head = [weight_norm(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))]
        body = [ResBlock(32, 6, 1.0, 0.8)
                for _ in range(6)]
        tail = [weight_norm(nn.Conv2d(32, 6 * (8 ** 2), kernel_size=3, padding=1)),
                nn.PixelShuffle(8)]
        # skip = [weight_norm(nn.Conv2d(in_channels, 3 * (4 ** 2), kernel_size=3, padding=1)),
        #         weight_norm(nn.Conv2d(48, 32, kernel_size=1)),
        #         weight_norm(nn.Conv2d(32, 3 * (4 ** 2), kernel_size=3, padding=1))]
        skip2 = [weight_norm(nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)),
                 nn.ReLU(inplace=True),
                 weight_norm(nn.Conv2d(8, 8, kernel_size=3, padding=1)),
                 nn.ReLU(inplace=True),
                 weight_norm(nn.Conv2d(8, 6, kernel_size=3, padding=1))]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        # self.skip = nn.Sequential(*skip)
        self.skip2 = nn.Sequential(*skip2)
        self.middle = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        # self.middle2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.outc = outconv(16, out_channels)

    def forward(self, x):
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        s = self.skip2(p1)
        # s = self.pixel_shuffle(s)
        # s2 = self.skip2(p1)
        s2 = self.upsample(s)
        x = self.head(p3)
        x = self.body(x)
        x = self.tail(x)
        x = torch.cat([x, s2], dim=1)
        # x = torch.cat([x, s], dim=1)
        x = self.middle(x)
        # x = self.middle2(x)
        x = self.outc(x)
        return x
