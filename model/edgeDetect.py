import torch
import torch.nn as nn
from model.common import default_conv, RDB


class MSRB(nn.Module):
    def __init__(self, conv=default_conv):
        super(MSRB, self).__init__()

        n_feats = 64
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


class EdgeDetect(nn.Module):
    def __init__(self, conv=default_conv, n_feats=64):
        super(EdgeDetect, self).__init__()

        kernel_size = 3
        n_blocks = 3
        self.n_blocks = n_blocks
        modules_head = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, n_feats, 7),
            nn.InstanceNorm2d(n_feats),
            nn.ReLU(inplace=True),
        ]

        modules_body = []
        for _ in range(n_blocks):
            modules_body.append(
                RDB(n_feats))

        self.Edge_Net_head = nn.Sequential(*modules_head)
        self.Edge_Net_body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.Edge_Net_head(x)
        
        x = self.Edge_Net_body(x)       

        return x



