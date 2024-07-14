
import torch
import torch.nn as nn
from model.common import EncoderInstance3
from model.edge import Edge

def make_model(args):
    return Sym(Edge, Edge)


class Sym(nn.Module):
    def __init__(self, RNet, DNet):
        # RNet: 复原网络
        # DNet: 退化网络
        super(Sym, self).__init__()
        self.RNet = RNet()
        self.DNet = DNet()
    def forward(self, x):
        # 复原-退化
        R1 = self.RNet(x)
        D1 = self.DNet(R1)

        # 退化-复原
        D2 = self.DNet(x)
        R2 = self.RNet(D2)
        return R1, D1, D2, R2
        

class ANet(nn.Module):
    def __init__(self):
        super(ANet, self).__init__()
        self.encoder = EncoderInstance3
        self.up_block = nn.PixelShuffle(2)
        self.tail = nn.Sequential(
            nn.Conv2d(40, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 20, kernel_size=3, padding=1),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
                20, 3, kernel_size=7, padding=0), nn.Tanh())
        )

    def forward(self, input):
        x_inital, x_layer1, x_layer2 = self.encoder(input)
        # 64 256 512
        # print(x_inital.shape, x_layer1.shape, x_layer2.shape)
        x = self.up_block(x_layer2)
        x = torch.cat((x, x_layer1), 1)
        x = self.up_block(x)
        x = torch.cat((x, x_inital), 1)
        x = self.up_block(x)
        x = self.tail(x)
        return x


class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.encoder = EncoderInstance3
        self.up_block = nn.PixelShuffle(2)
        self.tail = nn.Sequential(
            nn.Conv2d(40, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 20, kernel_size=3, padding=1),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
                20, 3, kernel_size=7, padding=0), nn.Tanh())
        )


    def forward(self, input):
        x_inital, x_layer1, x_layer2 = self.encoder(input)
        # 64 256 512
        # print(x_inital.shape, x_layer1.shape, x_layer2.shape)
        x = self.up_block(x_layer2)
        x = torch.cat((x, x_layer1), 1)
        x = self.up_block(x)
        x = torch.cat((x, x_inital), 1)
        x = self.up_block(x)
        x = self.tail(x)
        return x


class PhyNet(nn.Module):
    def __init__(self):
        super(PhyNet, self).__init__()
        self.ANet = ANet()
        self.TNet = TNet()

    def forward(self, I):
        A = self.ANet(I)
        T = self.TNet(I)
        return A, T
