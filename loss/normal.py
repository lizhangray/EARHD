import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from torch.nn import Conv2d
import numpy as np
import math
from torch.nn.functional import l1_loss
from utils.utils import frac_conv
# 全变分损失


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]

#


def TVBetter(TVLoss1, TVLoss2):
    return math.exp(-(TVLoss2 - TVLoss1))

# 暗通道损失


def DCLoss(img, patch_size):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size),
                           stride=1, padding=(0, patch_size//2, patch_size//2))
    dc = maxpool(1-img[:, None, :, :, :])

    target = torch.FloatTensor(dc.shape).zero_().cuda()

    loss = torch.nn.L1Loss(reduction='sum')(dc, target)
    return -loss


def testDCLoss(path):
    img = Image.open(path)
    img = transforms.ToTensor()(img)
    img = Variable(img[None, :, :, :].cuda(), requires_grad=True)
    loss = DCLoss(img, 7)
    print(loss)
    # loss.backward()


def grad_conv_hor():
    grad = Conv2d(3, 3, (1, 3), stride=1, padding=(0, 1))

    weight = np.zeros((3, 3, 1, 3))
    for i in range(3):
        weight[i, i, :, :] = np.array([[-1, 1, 0]])
    weight = torch.FloatTensor(weight).cuda()
    weight = nn.Parameter(weight, requires_grad=False)
    bias = np.array([0, 0, 0])
    bias = torch.FloatTensor(bias).cuda()
    bias = nn.Parameter(bias, requires_grad=False)
    grad.weight = weight
    grad.bias = bias
    return grad

# vertical gradient, the input_channel is default to 3


def grad_conv_vet():
    grad = Conv2d(3, 3, (3, 1), stride=1, padding=(1, 0))
    weight = np.zeros((3, 3, 3, 1))
    for i in range(3):
        weight[i, i, :, :] = np.array([[-1, 1, 0]]).T
    weight = torch.FloatTensor(weight).cuda()
    weight = nn.Parameter(weight, requires_grad=False)
    bias = np.array([0, 0, 0])
    bias = torch.FloatTensor(bias).cuda()
    bias = nn.Parameter(bias, requires_grad=False)
    grad.weight = weight
    grad.bias = bias
    return grad


def TVLossL1(img):
    hor = grad_conv_hor()(img)
    vet = grad_conv_vet()(img)
    target = torch.autograd.Variable(
        torch.FloatTensor(img.shape).zero_().cuda())
    loss_hor = torch.nn.L1Loss(reduction='sum')(hor, target)
    loss_vet = torch.nn.L1Loss(reduction='sum')(vet, target)
    loss = loss_hor+loss_vet
    return loss
# img = Image.open('Haze2Dehaze/train/B/04_outdoor_hazy.jpg')
# resize = transforms.Resize((256, 256))
# img = resize(transforms.ToTensor()(img))[None, :, :, :]
# img = torch.autograd.Variable(img, requires_grad=True)
# # testDCLoss(r'Haze2Dehaze/train/A/04_outdoor_GT.jpg')
# # testDCLoss(r'Haze2Dehaze/train/B/04_outdoor_hazy.jpg')
# loss = TVLossL1(img.cuda())
# print(loss)


class PSOLoss(nn.Module):
    def __init__(self, region_size=3):
        super(PSOLoss, self).__init__()
        self.avg_region = nn.Sequential(
            nn.AvgPool2d(region_size, stride=region_size),
            nn.ReflectionPad2d(region_size-1)
        )
        self.avg = nn.AvgPool2d(region_size, stride=1, padding=region_size//2)
        # self.pad = nn.ReflectionPad2d(region_size-1)
        self.l1_loss = nn.L1Loss()

    def forward(self, result, gt):
        gt = self.avg(gt)
        # gt = self.avg(gt)
        # print(gt.shape)
        # gt = self.pad(gt)
        # print(gt.shape)
        return self.l1_loss(result, gt)


class FracLoss(nn.Module):
    def __init__(self, channels=3):
        super(FracLoss, self).__init__()

        # self.pad = nn.ReflectionPad2d(region_size-1)
        self.fraction = [0.5, 0.6, 1]
        self.l1_loss = nn.L1Loss()
        self.channels = channels
        # tiansi_kernel = getTiansiOperator(channels)

    def forward(self, result, gt):
        loss = None
        for f in self.fraction:
            conv = frac_conv(self.channels, f).cuda()
            x = conv(result)
            y = conv(gt)
            if not loss:
                loss = self.l1_loss(x, y)
            else:
                loss += self.l1_loss(x, y)
        return loss
