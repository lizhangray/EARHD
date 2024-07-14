import torch
import torch.nn as nn

from model.base import BaseBranch, BaseBranch3
from model.edge import EdgeBranch3, EdgeBranch4
from model.common import MultiFussionBlock, RDB, ProgressEnhancer
from model.frac import FracEnhancer
from model.edgeDetect import EdgeDetect

def make_model(args):
    return Net()


# class MultiFussionBlock(nn.Module):
#     def __init__(self):
#         super(MultiFussionBlock, self).__init__()
#         self.block1 = nn.Conv2d(80, 60, kernel_size=3, padding=1)
#         self.block1_2 = nn.Conv2d(60, 40, kernel_size=3, padding=1)
#         self.block2 = nn.Conv2d(80, 40, kernel_size=3, padding=1)
#         self.block2_3 = nn.Conv2d(40, 28, kernel_size=3, padding=1)
#         self.block3 = nn.Conv2d(80, 28, kernel_size=3, padding=1)

#     def forward(self, x):
#         f1 = self.block1(x)
#         f1_2 = self.block1_2(f1)
#         f2 = self.block2(x)
#         f2 += f1_2
#         f2_3 = self.block2_3(f2)
#         f3 = self.block3(x)
#         f3 += f2_3
#         return f3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        feature = 40
        base_out_feature = 40
       
        # fussion_feature = int(44 + 3)
        # self.baseBranch = BaseBranch3()
        # self.EdgeBranch = EdgeBranch3()
        self.EdgeDetection = EdgeDetect() # 64
        self.EdgeDetectionTail = nn.Sequential(
            nn.Conv2d(64, 40, kernel_size=3, padding=1),
            nn.Conv2d(40, 28, kernel_size=3, padding=1),
            nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
                28, 3, kernel_size=7, padding=0), nn.Tanh())
        )
        self.EdgeBranch = EdgeBranch3()
        # self.baseTail = nn.Sequential(
        #     nn.Conv2d(40, 40, kernel_size=3, padding=1),
        #     nn.Conv2d(40, 28, kernel_size=3, padding=1),
        #     nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
        #         28, 3, kernel_size=7, padding=0), nn.Tanh())
        # )

        # 目前在跑的实验 ea3 18.4 18.3左右
        # 尾处理网络
        self.tail = nn.Sequential(
            nn.Conv2d(104, 40, kernel_size=3, padding=1),
            nn.Conv2d(40, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 30, kernel_size=3, padding=1),
            nn.Conv2d(30, 20, kernel_size=3, padding=1),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
                20, 3, kernel_size=7, padding=0), nn.Tanh())
        )
        # self.EdgeBranchTail = nn.Sequential(
        #     # nn.Conv2d(80, 40, kernel_size=3, padding=1),
        #     nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
        #         20, 3, kernel_size=7, padding=0), nn.Tanh())
        # )
        # self.tail4 = nn.Sequential(
        #     nn.Conv2d(44, 30, kernel_size=3, padding=1),
        #     nn.Conv2d(30, 30, kernel_size=3, padding=1),
        #     nn.Conv2d(30, 20, kernel_size=3, padding=1),
        #     nn.Conv2d(20, 20, kernel_size=3, padding=1),
        #     nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
        #         20, 3, kernel_size=7, padding=0), nn.Tanh())
        # )
        # self.rdb_out = RDB(40, 4, 16)

        self.enhancer = ProgressEnhancer(40)
        # self.fracEnhance = FracEnhancer()
        # self.tail = nn.Sequential(
        #     # nn.Conv2d(40, 35, kernel_size=3, padding=1),
        #     # nn.Conv2d(35, 35, kernel_size=3, padding=1),

        #     # nn.Conv2d(35, 30, kernel_size=3, padding=1),
        #     # nn.Conv2d(30, 30, kernel_size=3, padding=1),

        #     # nn.Conv2d(30, 25, kernel_size=3, padding=1),
        #     # nn.Conv2d(25, 25, kernel_size=3, padding=1),

        #     # nn.Conv2d(25, 20, kernel_size=3, padding=1),
        #     # nn.Conv2d(20, 20, kernel_size=3, padding=1),
        #     nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
        #         40, 3, kernel_size=7, padding=0), nn.Tanh())
        # )
        # self.tail = nn.Sequential(
        #     # MultiFussionBlock(),
        #     nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
        #         20, 3, kernel_size=7, padding=0), nn.Tanh())
        # )
        # self.tail = nn.Sequential(
        #     nn.Conv2d(8, 8, kernel_size=3, padding=1),
        #     nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
        #         8, 3, kernel_size=7, padding=0), nn.Tanh())
        # )
        # self.tail = nn.Conv2d(6, 3, kernel_size=3, padding=1)
        # self.conv_process_1 = nn.Conv2d(fussion_feature, int(
        #     fussion_feature/2), kernel_size=3, padding=1)
        # self.conv_process_2 = nn.Conv2d(
        #     int(fussion_feature/2), int(fussion_feature/2), kernel_size=3, padding=1)
        # self.tail = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(
        #     int(fussion_feature/2), 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        # base_output = self.baseBranch(input)
        edge_feature = self.EdgeDetection(input)
        edge = self.EdgeDetectionTail(edge_feature)
        x = self.EdgeBranch(input)
        x = torch.cat((edge_feature, x), 1)
        x = self.tail(x)
        return x, edge
