# 放平心态，go and fight
# 重新出发20211019
import argparse
import datetime
import itertools
import math
import os
import time

import numpy as np
# import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from numpy.lib.type_check import real
from torch.autograd import Variable
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import make_grid, save_image

from loss import Loss
from model.common import Res2Net, res2net101
from option import args
from utils.dataloader import normalTransform
from utils.utils import make_optimizer, make_scheduler
from model.edge import Edge, Encoder2Decoder
from model.gan import NLayerDiscriminator, DegNet, GeneratorResNet, Discriminator3
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, args, test_loader, ckp):
        self.test_loader = test_loader
        self.ckp = ckp
        self.args = args

        # 改成下面的使用超分模块
        self.RNet = Encoder2Decoder()
        self.DNet = Edge()

        self.RNet.cuda()
        self.DNet.cuda()
        self.optimizer = torch.optim.Adam(
            list(self.RNet.parameters())+list(self.DNet.parameters()), lr=args.lr, betas=(args.beta1, args.beta2)
        )
        self.scheduler = make_scheduler(args, self.optimizer)

        self.D1 = Discriminator3()
        self.D1.cuda()
        self.D1_optimizer = make_optimizer(args, self.D1, 1e-4)
        self.D1_scheduler = make_scheduler(args, self.D1_optimizer)

        self.D2 = Discriminator3()
        self.D2.cuda()
        self.D2_optimizer = make_optimizer(args, self.D2, 1e-4)
        self.D2_scheduler = make_scheduler(args, self.D2_optimizer)


    def testOnly(self):
        self.RNet.load_state_dict(torch.load(self.args.test_model_dir))
        self.RNet = self.prepare(self.RNet)
        print(self.args.test_model_dir)
        self.RNet.eval()
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                haze, gt = data['image']
                haze, gt = self.prepare(haze, gt)
                haze_name, gt_name = data['name']
                dehaze = self.RNet(haze)
                self.ckp.saveTestImage(gt_name[0], dehaze, 'dehaze')
                self.ckp.metric_tensor(dehaze, gt, normalize=True)
            self.ckp.saveTestMetric()


    # 将对应tensor加载到cuda中
    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            return tensor.to(device)

        if len(args) == 1:
            return _prepare(*args)

        return [_prepare(a) for a in args]
    

    def terminate(self):
        if self.args.test_only:
            self.testOnly()
            return True
        return False


