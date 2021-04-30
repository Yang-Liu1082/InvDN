import torch
from torch import nn
import torch.nn.functional as F
from models.modules.ResAttentionBlock import CALayer
# from utils import weights_init_kaiming

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.bn1(self.conv1(x)))
        out = F.relu(self.conv2(residual) + x)  # addition Structure 6-1 elf.bn2
        return out


class ResidualUnit(nn.Module):  # 4 residual block
    def __init__(self, channels):
        super(ResidualUnit, self).__init__()
        self.block1 = ResidualBlock(channels)
        self.block2 = ResidualBlock(channels)
        self.block3 = ResidualBlock(channels)
        self.block4 = ResidualBlock(channels)
        self.conv = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.ca = CALayer(channels)  # with attention
        # self.cbam = CBAM(channels) # with cbam attention
        # self.aoa = AoA(channels)  # with aoa attention

    def forward(self, x):
        res = self.block1(x)
        res = self.block2(res)
        res = self.block3(res)
        res = self.block4(res)
        mid = torch.cat((x, res), dim=1)
        out = self.conv(mid)
        out = self.ca(out)  # with attention
        # out = self.cbam(out) # with cbam attention
        # out = self.aoa(out)
        return out

class ResidualModule(nn.Module):
    def __init__(self, channels):
        super(ResidualModule, self).__init__()
        self.block1 = ResidualUnit(channels)
        self.block2 = ResidualUnit(channels)
        self.block3 = ResidualUnit(channels)
        self.block4 = ResidualUnit(channels)

    def forward(self, x):
        res = self.block1(x)
        res = self.block2(res)
        res = self.block3(res)
        res = self.block4(res)
        return (x + res)

class MSKResnet(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(MSKResnet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels_in, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding,
                      bias=False),
            nn.ReLU(inplace=True)
        )
        self.block2 = ResidualModule(features)
        self.block3 = nn.Conv2d(in_channels=features, out_channels=channels_out, kernel_size=kernel_size, padding=padding,
                                bias=False)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        out = self.block3(block2)
        return out