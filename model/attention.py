import torch
import torch.nn as nn
from model.common import default_conv, ResidualBlock
from utils.checkpoint import checkpoint
ckp = checkpoint()

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y



class RCAB(nn.Module):
    def __init__(self, num_features, reduction=8):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            CALayer(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


# 实验一 上采样之后直接接入该模块
# 模型图
class Edge_Attention_Block_1(nn.Module):
    def __init__(self, channel):
        super(Edge_Attention_Block_1, self).__init__()
        self.ea = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.ea(x)
        return x * y, y


# based on msrb => 效果并不好

# class Edge_Attention_Layer(nn.Module):
#     def __init__(self, channel):
#         super(Edge_Attention_Layer, self).__init__()
#         self.conv1 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
#         self.conv2 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
#         self.conv3 = nn.Conv2d(channel*2, channel*2, 1, padding=0, bias=True)
#         self.conv4 = nn.Conv2d(channel*2, channel*2, 1, padding=0, bias=True)
#         self.conv5 = nn.Conv2d(channel*4, channel*4, 1, padding=0, bias=True)
#         self.conv6 = nn.Conv2d(channel*4, channel*4, 1, padding=0, bias=True)
#         self.confusion = nn.Conv2d(channel*8, 1, 1, padding=0, bias=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmod = nn.Sigmoid()
#     def forward(self, x):
#         input_1 = x
#         output_1_1 = self.relu(self.conv1(input_1))
#         output_1_2 = self.relu(self.conv2(output_1_1))
#         input_2 = torch.cat([output_1_1, output_1_2], 1)
#         output_2_1 = self.relu(self.conv3(input_2))
#         output_2_2 = self.relu(self.conv4(output_2_1))
#         input_3 = torch.cat([output_2_1, output_2_2], 1)
#         output_3_1 = self.relu(self.conv5(input_3))
#         output_3_2 = self.relu(self.conv6(output_3_1))
#         input_4 = torch.cat([output_3_1, output_3_2], 1)
#         output = self.confusion(input_4)
#         edge = self.sigmod(output)
#         return x * edge, edge

# EA块具体长这样
# newEdge no Element wise
class Edge_Attention_Layer(nn.Module):
    def __init__(self, channel):
        super(Edge_Attention_Layer, self).__init__()
        self.XConv = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
        )
        self.YConv = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
        )
        self.InfoConv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
            nn.InstanceNorm2d(channel),
            # Relu可以注释掉
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        # self.InfoConv = nn.Sequential(
        #     ResidualBlock(channel),
        #     ResidualBlock(channel),
        #     ResidualBlock(channel)
        # )

    def forward(self, input):
        x = self.XConv(input)
        y = self.YConv(input)
        ea = x + y
        info = self.InfoConv(input)
        # ckp.saveAttentionMapInBlocks(x, 'x')
        # ckp.saveAttentionMapInBlocks(y, 'y')
        # ckp.saveAttentionMapInBlocks(ea, 'ea')
        return input + info*ea

# EA块X2
class Edge_Attention_block(nn.Module):
    def __init__(self, dim):
        super(Edge_Attention_block, self).__init__()
        # self.ealayer = Edge_Attention_Layer(dim)
        # self.fusion = nn.Conv2d(dim*2, dim, 1, padding=0, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # res_blocks_nums = 1
        # res_blocks = []
        # for _ in range(res_blocks_nums):
        #     res_blocks.append(ResidualBlock(dim))
        # self.res_blocks = nn.Sequential(*res_blocks)
        ea_block_nums = 2
        ea_blocks = []
        for _ in range(ea_block_nums):
            ea_blocks.append(Edge_Attention_Layer(dim))
        self.ealayer = nn.Sequential(*ea_blocks)

    def forward(self, input):
        # res = self.fusion(input)
        # res = self.relu(input)
        x = self.ealayer(input)
        # x = self.res_blocks(input)

        return x


class MSRB_EDGE(nn.Module):
    def __init__(self, feature, conv=default_conv):
        super(MSRB_EDGE, self).__init__()

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
        self.edge_attention1 = Edge_Attention_block(n_feats)
        self.edge_attention2 = Edge_Attention_block(n_feats*2)
        self.edge_attention3 = Edge_Attention_block(n_feats*4)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        input_3 = self.edge_attention3(input_3)
        output = self.confusion(input_3)
        output += x
        return output
