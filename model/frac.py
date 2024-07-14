import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model.attention import (CP_Attention_block, Edge_Attention_block,
                             Edge_Attention_Layer)
from model.common import Bottle2neck, Res2NetCustom, default_conv, res2net101


class TiansiOperator(nn.Module):
    def __init__(self, feature, output_feature, order):
        super(TiansiOperator, self).__init__()
        # self.tiansi = F.conv2d()
        self.weight = self.createWeight(order, feature, output_feature)

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def createWeight(self, v, channel, output_channel):
        a0 = 1
        a1 = -v
        a2 = (v**2-v)/2
        m = 8-12*v + 4*(v**2)
        kernel = np.array([
            [a2, 0, a2, 0, a2],
            [0, a1, a1, a1, 0],
            [a2, a1, 8*a0, a1, a2],
            [0, a1, a1, a1, 0],
            [a2, 0, a2, 0, a2]
        ], dtype=np.float32)
        return transforms.ToTensor()(kernel).expand(output_channel, channel, 5, 5)

    def forward(self, input):
        input = F.conv2d(input, self.weight, padding=2)
        return input


class FractionEncoder(nn.Module):
    def __init__(self, channel):
        super(FractionEncoder, self).__init__()
        self.down1 = nn.Sequential(
            TiansiOperator(channel, channel,  0.5),
            nn.Conv2d(channel, channel*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(channel*2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            TiansiOperator(channel*2, channel*2, 0.5),
            nn.Conv2d(channel*2, channel*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(channel*4),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            TiansiOperator(channel*4, channel*4, 0.5),
            nn.Conv2d(channel*4, channel*8, 3, stride=2, padding=1),
            nn.InstanceNorm2d(channel*8),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        x_d1 = self.down1(input)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        return x_d1, x_d2, x_d3


def getTiansiOperator(channels, v):
    a0 = 1
    a1 = -v
    a2 = (v**2-v)/2
    m = 8-12*v + 4*(v**2)
    kernel = torch.tensor([
        [a2, 0, a2, 0, a2],
        [0, a1, a1, a1, 0],
        [a2, a1, 8*a0, a1, a2],
        [0, a1, a1, a1, 0],
        [a2, 0, a2, 0, a2]
    ])
    return kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)


def frac_conv(channels, v):
    kernel_size = 5
    frac_filter = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                            groups=channels, bias=False, padding=kernel_size//2)
    frac_filter.weight.data = getTiansiOperator(channels, v)

    frac_filter.weight.requires_grad = False
    return frac_filter

# 串联效果18.01
# class FracEnhancer(nn.Module):
#     def __init__(self):
#         super(FracEnhancer, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(40, 30, kernel_size=3, padding=1),
#             nn.Conv2d(30, 30, kernel_size=3, padding=1),
#             nn.Conv2d(30, 20, kernel_size=3, padding=1),
#             nn.Conv2d(20, 20, kernel_size=3, padding=1),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(40, 30, kernel_size=3, padding=1),
#             nn.Conv2d(30, 30, kernel_size=3, padding=1),
#             nn.Conv2d(30, 20, kernel_size=3, padding=1),
#             nn.Conv2d(20, 20, kernel_size=3, padding=1),
#         )
#         self.frac_conv_five = frac_conv(40, 0.5)
#         self.frac_conv_six = frac_conv(40, 0.6)

#     def forward(self, x):
#         out_five = self.frac_conv_five(x)
#         out_five += x
#         out_five = self.frac_conv_six(out_five)
#         # out_five = self.conv1(out_five)
#         out_five += x
#         # out_six = self.frac_conv_six(x)
#         # out_six += x
#         # out_six = self.conv2(out_six)

#         # out = torch.cat([out_five, out_six], 1)
#         return out_five
class FracEnhancer(nn.Module):
    def __init__(self):
        super(FracEnhancer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(40, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 20, kernel_size=3, padding=1),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 20, kernel_size=3, padding=1),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
        )
        self.frac_conv_five = frac_conv(40, 0.5)
        self.frac_conv_six = frac_conv(40, 0.6)

    def forward(self, x):
        out_five = self.frac_conv_five(x)
        out_six = self.frac_conv_six(x)
        out = torch.cat([out_five, out_six, x], 1)
        return out
