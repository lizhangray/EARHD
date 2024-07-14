import torch
import torch.nn as nn

from model.attention import (CP_Attention_block, Edge_Attention_block,
                             Edge_Attention_Layer, MSRB_EDGE, RCAB)
from model.common import Bottle2neck, Res2NetCustom, default_conv, res2net101, MSRB, Encoder3, EncoderInstance3, EncoderInstance4, ResBlock


def make_model(args):
    return Edge()


class Encoder2Decoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder2Decoder, self).__init__()
        self.encoder = EncoderInstance3
        self.super_decoder = SuperDecoder()
        self.edge_decoder = EdgeDecoder()
        self.tail = nn.Sequential(
            nn.Conv2d(80, 40, kernel_size=3, padding=1),
            nn.Conv2d(40, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 20, kernel_size=3, padding=1),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
                20, 3, kernel_size=7, padding=0), nn.Tanh())
        )

    def forward(self, input):
        encoder_flist = self.encoder(input)
        super_out = self.super_decoder(encoder_flist)
        edge_out = self.edge_decoder(encoder_flist)
        out = torch.cat((super_out, edge_out), 1)
        out = self.tail(out)
        return out


class Edge(nn.Module):
    def __init__(self):
        super(Edge, self).__init__()
        self.edge_branch = EdgeBranch3()
        # self.super_branch = EdgeSuper()
        self.tail = nn.Sequential(
            nn.Conv2d(40, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 20, kernel_size=3, padding=1),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
                20, 3, kernel_size=7, padding=0), nn.Tanh())
        )

    def forward(self, input):
        x = self.edge_branch(input)
        x = self.tail(x)
        # x = self.super_branch(x)
        return x


class EdgeBranch3(nn.Module):
    def __init__(self, feature=40):
        super(EdgeBranch3, self).__init__()
        self.encoder = EncoderInstance3
        self.up_block = nn.PixelShuffle(2)
        self.attention0 = nn.Sequential(
            Edge_Attention_block(512),

        )
        self.attention1 = nn.Sequential(
            Edge_Attention_block(128),

        )
        self.attention2 = nn.Sequential(
            Edge_Attention_block(96),

        )
        self.attention3 = nn.Sequential(
            Edge_Attention_block(40),

        )

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


class EdgeBranch4(nn.Module):
    def __init__(self, feature=40):
        super(EdgeBranch4, self).__init__()
        self.encoder = EncoderInstance4
        # self.num_blocks = 24
        # self.head = nn.Sequential(
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(3, feature, 7),
        #     nn.InstanceNorm2d(feature),
        #     nn.ReLU(inplace=True),
        # )

        self.up_block = nn.PixelShuffle(2)
        self.attention0 = nn.Sequential(
            Edge_Attention_block(1024),

        )
        self.attention1 = nn.Sequential(
            Edge_Attention_block(256),

        )
        self.attention2 = nn.Sequential(
            Edge_Attention_block(192),

        )
        self.attention3 = nn.Sequential(
            Edge_Attention_block(112),

        )
        self.attention4 = nn.Sequential(
            Edge_Attention_block(44),

        )

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

# 超分分支去掉权重共享部分
class SuperDecoder(nn.Module):
    def __init__(self):
        super(SuperDecoder, self).__init__()
        num_features = 512  # 特征的数量
        n_resblocks = 3 # 残差块的数目
        scale = 2   # 上采样的倍数，设置为2
        body = [
            ResBlock(
                default_conv, num_features, 3, act=nn.ReLU(True), res_scale=1
            ) for _ in range(n_resblocks)
        ]
        self.body = nn.Sequential(*body)

        # 将低分辨率特征图上采样到高分辨率
        # 2倍上采样
        self.upscale1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

        self.upscale2 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

        self.upscale3 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

        self.conv_end = nn.Conv2d(40, 40, kernel_size=3, padding=1)

    def forward(self, encode_flist):
        x_inital, x_layer1, x_layer2 = encode_flist # 第二块、第三块残差、主干

        x = self.body(x_layer2)
        x = self.upscale1(x)

        x = torch.cat((x, x_layer1), 1)

        x = self.upscale2(x)
        x = torch.cat((x, x_inital), 1)

        x = self.upscale3(x)
        x = self.conv_end(x)

        return x

# EA解码器
class EdgeDecoder(nn.Module):
    def __init__(self):
        super(EdgeDecoder, self).__init__()
        self.up_block = nn.PixelShuffle(2) # 上采样
        self.attention0 = nn.Sequential(
            Edge_Attention_block(512),
        )
        self.attention1 = nn.Sequential(
            Edge_Attention_block(128),

        )
        self.attention2 = nn.Sequential(
            Edge_Attention_block(96),

        )
        self.attention3 = nn.Sequential(
            Edge_Attention_block(40),
        )

    def forward(self, encode_flist):
        x_inital, x_layer1, x_layer2 = encode_flist
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
