# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, n_channels, out_channels):
        super(UNet, self).__init__()
        ndf = 48
        self.inc = inconv(n_channels, ndf)
        self.down1 = down(ndf, ndf * 2)
        self.down2 = down(ndf * 2, ndf * 4)
        self.down3 = down(ndf * 4, ndf * 8)
        self.down4 = down(ndf * 8, ndf * 16)
        self.down5 = down(ndf * 16, ndf * 16)
        self.up1 = up(ndf * 32, ndf * 8)
        self.up2 = up(ndf * 16, ndf * 4)
        self.up3 = up(ndf * 8, ndf * 2)
        self.up4 = up(ndf * 4, ndf)
        self.up5 = up(ndf * 2, ndf)
        self.outc = outconv(ndf, out_channels)

        # self.out_layer = torch.nn.Softmax2d()

    def forward(self, x):
        x1 = self.inc(x)  # ndf
        x2 = self.down1(x1)  # ndf*2
        x3 = self.down2(x2)  # ndf*4
        x4 = self.down3(x3)  # ndf*8
        x5 = self.down4(x4)  # ndf*8
        x6 = self.down5(x5)  # ndf*8
        x = self.up1(x6, x5)  # ndf*8
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        mask = self.outc(x)
        return mask
