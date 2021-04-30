import math

import torch
from torch import nn
import models.modules.module_util as mutil
import torch.nn.functional as F

class Merge_Run(nn.Module):
    def __init__(self,
                 in_channels, out_channels, init='xavier',
                 ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run, self).__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, ksize, stride, pad),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # init_weights(self.modules)
        # if init == 'xavier':
        #     mutil.initialize_weights_xavier([self.body1, self.body2, self.body3], 0.1)
        # else:
        #     mutil.initialize_weights([self.body1, self.body2, self.body3], 0.1)

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class Merge_Run_dual(nn.Module):
    def __init__(self,
                 in_channels, out_channels, init='xavier',
                 ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run_dual, self).__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, ksize, stride, pad),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # init_weights(self.modules)
        # if init == 'xavier':
        #     mutil.initialize_weights_xavier([self.body1, self.body2, self.body3], 0.1)
        # else:
        #     mutil.initialize_weights([self.body1, self.body2, self.body3], 0.1)

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, init=None,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # init_weights(self.modules)
        # if init == 'xavier':
        #     mutil.initialize_weights_xavier([self.body], 0.1)
        # else:
        #     mutil.initialize_weights([self.body], 0.1)

    def forward(self, x):
        out = self.body(x)
        return out


class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels, init='xavier',
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        # init_weights(self.modules)
        # if init == 'xavier':
        #     mutil.initialize_weights_xavier([self.body], 0.1)
        # else:
        #     mutil.initialize_weights([self.body], 0.1)

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,  init='xavier'):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        # init_weights(self.modules)
        # if init == 'xavier':
        #     mutil.initialize_weights_xavier([self.body], 0.1)
        # else:
        #     mutil.initialize_weights([self.body], 0.1)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, init='xavier',
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        # init_weights(self.modules)
        # if init == 'xavier':
        #     mutil.initialize_weights_xavier([self.body], 0.1)
        # else:
        #     mutil.initialize_weights([self.body], 0.1)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)

class _UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, init='xavier',
                 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        # init_weights(self.modules)
        # if init == 'xavier':
        #     mutil.initialize_weights_xavier(self.body, 0.1)
        # else:
        #     mutil.initialize_weights(self.body, 0.1)

    def forward(self, x):
        out = self.body(x)
        return out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel, channel // reduction, 1, 1, 0)
        self.c2 = BasicBlockSig(channel // reduction, channel, 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2

class RABlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(RABlock, self).__init__()

        self.r1 = Merge_Run_dual(in_channels, out_channels)
        self.r2 = ResidualBlock(in_channels, out_channels)
        self.r3 = EResidualBlock(in_channels, out_channels)
        # self.g = ops.BasicBlock(in_channels, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        r1 = self.r1(x)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        # g = self.g(r3)
        out = self.ca(r3)

        return out
