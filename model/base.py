import torch
import torch.nn as nn

from model.attention import (CP_Attention_block, Edge_Attention_block,
                             Edge_Attention_Layer)
from model.common import Bottle2neck, Res2NetCustom, Res2NetCustom3, Encoder3, EncoderInstance3, default_conv, res2net101


def make_model():
    return knowledge_adaptation_UNet()


class knowledge_adaptation_UNet(nn.Module):
    def __init__(self):
        super(knowledge_adaptation_UNet, self).__init__()
        print('Cur Net: Base DW-GAN Net')
        self.encoder = Res2NetCustom(
            Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net = res2net101(preTrained=True)
        pretrained_dict = res2net.state_dict()
        model_dict = self.encoder.state_dict()
        # Res2NetCustom少了全连接层，所以在加载原先的预训练模型的时候，需要将多余的参数去掉
        key_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)
        self.up_block = nn.PixelShuffle(2)
        self.attention0 = CP_Attention_block(default_conv, 1024, 3)
        self.attention1 = CP_Attention_block(default_conv, 256, 3)
        self.attention2 = CP_Attention_block(default_conv, 192, 3)
        self.attention3 = CP_Attention_block(default_conv, 112, 3)
        self.attention4 = CP_Attention_block(default_conv, 44, 3)
        self.conv_process_1 = nn.Conv2d(44, 44, kernel_size=3, padding=1)
        self.conv_process_2 = nn.Conv2d(44, 28, kernel_size=3, padding=1)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
            28, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        # input 3 256 256
        x_inital, x_layer1, x_layer2, x_output = self.encoder(input)
        # 64 256 256
        # 256 128 128
        # 512 64 64
        # 1024 32 32
        x_mid = self.attention0(x_output)
        x = self.up_block(x_mid)
        x = self.attention1(x)
        x = torch.cat((x, x_layer2), 1)
        x = self.up_block(x)
        x = self.attention2(x)
        x = torch.cat((x, x_layer1), 1)
        x = self.up_block(x)
        x = self.attention3(x)
        x = torch.cat((x, x_inital), 1)
        x = self.up_block(x)
        x = self.attention4(x)
        x = self.conv_process_1(x)
        x = self.conv_process_2(x)
        out = self.tail(x)
        return out


class BaseBranch(nn.Module):
    def __init__(self):
        super(BaseBranch, self).__init__()
        self.encoder = Res2NetCustom(
            Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net = res2net101(preTrained=True)
        pretrained_dict = res2net.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)
        self.up_block = nn.PixelShuffle(2)
        self.attention0 = CP_Attention_block(default_conv, 1024, 3)
        self.attention1 = CP_Attention_block(default_conv, 256, 3)
        self.attention2 = CP_Attention_block(default_conv, 192, 3)
        self.attention3 = CP_Attention_block(default_conv, 112, 3)
        self.attention4 = CP_Attention_block(default_conv, 44, 3)

    def forward(self, input):
        x_inital, x_layer1, x_layer2, x_output = self.encoder(input)
        x_mid = self.attention0(x_output)
        x = self.up_block(x_mid)
        x = self.attention1(x)
        x = torch.cat((x, x_layer2), 1)
        x = self.up_block(x)
        x = self.attention2(x)
        x = torch.cat((x, x_layer1), 1)
        x = self.up_block(x)
        x = self.attention3(x)
        x = torch.cat((x, x_inital), 1)
        x = self.up_block(x)
        x = self.attention4(x)
        return x


class BaseBranch3(nn.Module):
    def __init__(self):
        super(BaseBranch3, self).__init__()
        self.encoder = EncoderInstance3
        self.up_block = nn.PixelShuffle(2)
        self.attention0 = CP_Attention_block(default_conv, 512, 3)
        self.attention1 = CP_Attention_block(default_conv, 128, 3)
        self.attention2 = CP_Attention_block(default_conv, 96, 3)
        self.attention3 = CP_Attention_block(default_conv, 40, 3)

    def forward(self, input):
        x_inital, x_layer1, x_layer2 = self.encoder(input)
        # 64 256 512
        # print(x_inital.shape, x_layer1.shape, x_layer2.shape)
        x_mid = self.attention0(x_layer2)
        x = self.up_block(x_mid)
        x = self.attention1(x)
        x = torch.cat((x, x_layer1), 1)
        x = self.up_block(x)
        x = self.attention2(x)
        x = torch.cat((x, x_inital), 1)
        x = self.up_block(x)
        x = self.attention3(x)
        return x
