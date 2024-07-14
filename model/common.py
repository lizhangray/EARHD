import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from option import args
from utils.utils import loadWeights

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
    'res2net101': 'res2net101',
}


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3,
                         stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes *
                               self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 原始版本Res2Net 包含全连接层


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                          baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# 去掉全连接层的结果


class Res2NetCustom(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2NetCustom, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                          baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_init = self.relu(x)
        x = self.maxpool(x_init)
        # print('before layer1', x.shape)
        x_layer1 = self.layer1(x)
        # print('layer1', x_layer1.shape)
        x_layer2 = self.layer2(x_layer1)
        x_output = self.layer3(x_layer2)
        return x_init, x_layer1, x_layer2, x_output


class Res2NetCustom3(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2NetCustom3, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                          baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_init = self.relu(x)
        x = self.maxpool(x_init)
        # print('before layer1', x.shape)
        x_layer1 = self.layer1(x)
        # print('layer1', x_layer1.shape)
        x_layer2 = self.layer2(x_layer1)
        return x_init, x_layer1, x_layer2


def res2net101(preTrained=False):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
    if preTrained:
        model.load_state_dict(loadWeights(
            model_urls['res2net101'], './weights'))
    return model


class TiansiOperator(nn.Module):
    def __init__(self):
        super(TiansiOperator, self).__init__()
        self.tiansi = F.conv2d()

    def createTemplate(self, v, channel):
        kernel = torch.ones((1, channel, 8, 8))
        return kernel


class MSRB(nn.Module):
    def __init__(self, feature, conv=default_conv):
        super(MSRB, self).__init__()

        n_feats = feature
        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(
            n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class Encoder3(nn.Module):
    def __init__(self):
        super(Encoder3, self).__init__()

        self.encoder = Res2NetCustom3(
            Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net = res2net101(preTrained=True)
        pretrained_dict = res2net.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)

    def forward(self, x):
        return self.encoder(x)


class Encoder4(nn.Module):
    def __init__(self):
        super(Encoder4, self).__init__()

        self.encoder = Res2NetCustom(
            Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net = res2net101(preTrained=True)
        pretrained_dict = res2net.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)

    def forward(self, x):
        return self.encoder(x)


EncoderInstance3 = Encoder3()
EncoderInstance4 = Encoder4()


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            # Relu可以注释掉
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)
        
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))# 对输入特征图进行卷积操作，产生相同通道数的输出特征图
            if bn:
                m.append(nn.BatchNorm2d(n_feats))# 归一化
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m) # 残差块的主体内容
        self.res_scale = res_scale # 权重系数

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


# class MultiFussionBlock(nn.Module):
#     def __init__(self, in_f, out_f, distance):
#         super(MultiFussionBlock, self).__init__()
#         self.block1 =  nn.Conv2d(80, 60, 3, padding=1),
#         self.block1_2 = nn.Conv2d(60, 40, 3, 1)
#         self.block2 =  nn.Conv2d(80, 40, 3, padding=1),
#         self.block2_3 =  nn.Conv2d(40, 28, 3, padding=1),
#         self.block3 =  nn.Conv2d(80, 28, 3, padding=1),
#     def forward(self, x):
#         f1 = self.block1(x)
#         f1_2 = self.block1_2(f1)
#         f2 = self.block2(x)
#         f2 += f1_2
#         f2_3 = self.block2_3(f2)
#         f3 = self.block3(x)
#         f3 += f2_3
#         return f3

class MultiFussionBlock(nn.Module):
    def __init__(self):
        super(MultiFussionBlock, self).__init__()
        self.block1 = nn.Conv2d(40, 35, kernel_size=3, padding=1)
        self.block2 = nn.Conv2d(40, 30, kernel_size=3, padding=1)
        self.block3 = nn.Conv2d(40, 25, kernel_size=3, padding=1)
        self.block4 = nn.Conv2d(40, 20, kernel_size=3, padding=1)

        self.block1_2 = nn.Conv2d(35, 30, kernel_size=3, padding=1)
        self.block2_3 = nn.Conv2d(30, 25, kernel_size=3, padding=1)
        self.block3_4 = nn.Conv2d(25, 20, kernel_size=3, padding=1)

        self.block2_2 = nn.Conv2d(30, 30, kernel_size=3, padding=1)
        self.block3_3 = nn.Conv2d(25, 25, kernel_size=3, padding=1)
        self.block4_4 = nn.Conv2d(20, 20, kernel_size=3, padding=1)

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(x)
        f3 = self.block3(x)
        f4 = self.block4(x)

        f1_2 = self.block1_2(f1)
        f2 += f1_2
        f2 = self.block2_2(f2)

        f2_3 = self.block2_3(f2)
        f3 += f2_3
        f3 = self.block3_3(f3)

        f3_4 = self.block3_4(f3)
        f4 += f3_4
        f4 = self.block4_4(f4)
        return f4


# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate,
                              kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for _ in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(
            _in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


# class ProgressEnhancer(nn.Module):
#     def __init__(self, feature):
#         super(ProgressEnhancer, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(feature, 60, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(60, feature, 3, padding=1),
#             nn.ReLU(),
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv2d(feature, 50, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, feature, 3, padding=1),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         out1 = self.block1(x)
#         out1 = out1 + x
#         out2 = self.block2(out1)
#         out2 = out2 + out1
#         return out2

class ProgressEnhancer(nn.Module):
    def __init__(self, feature):
        super(ProgressEnhancer, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(feature, 30, 3, padding=1),

        )
        self.block1_1 = nn.Sequential(
            nn.Conv2d(30, feature, 3, padding=1),
        )
        self.block1_2 = nn.Sequential(
            nn.Conv2d(feature, 30, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(30, 20, 3, padding=1),

        )
        self.block2_1 = nn.Sequential(
            nn.Conv2d(20, 30, 3, padding=1),
        )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(30, 20, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.block1(x)  # 30
        x1_1 = self.block1_1(x1)  # 40
        x1_1 = x1_1 + x
        x1_2 = self.block1_2(x1_1)  # 30
        x1_2 = x1_2 + x1  # 30
        x2 = self.block2(x1_2)  # 20
        x2_1 = self.block2_1(x2)  # 30
        x2_1 = x2_1 + x1_2
        x2_2 = self.block2_2(x2_1)  # 20
        x2_2 = x2_2 + x2

        return x2_2

        
class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer=3, growthRate=16):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out