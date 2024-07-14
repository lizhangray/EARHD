from matplotlib import pyplot as plt
import torch.nn as nn
from option import args
from loss.normal import FracLoss


class Loss():
    def __init__(self, loss_type, desc):
        self.loss = []
        self.loss_epoch=[]
        self.batch=87
        self.logDir = args.logDir
        self.ct = 0
        self.desc = desc
        self.lamda=1
        
        if self.desc == 'loss-d1' or self.desc == 'loss-d2':
            self.lamda = 4
        elif self.desc == 'loss-frac-dehaze':
            self.lamda = 2
        if loss_type == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'L1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'FRAC':
            self.loss_fn = FracLoss()
        elif loss_type == 'CrossEntropy':
            self.loss_fn = nn.CrossEntropyLoss()

    def saveLossItem(self, loss, gama):
        self.loss.append(gama*loss.item())

    def saveLossBatch(self, loss):
        self.loss_epoch.append(loss.item())

    def saveLossImage(self, name):
        x = list(range(1, len(self.loss_epoch)+1))
        plt.plot(x, self.loss_epoch)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f'{name}')
        plt.savefig(f'{self.logDir}/{name}.png')
        plt.clf()

    def cal_loss(self, input, target):
        input = input.cuda()
        target = target.cuda()
        loss_item = self.loss_fn(input, target)
        self.ct += 1
        # 对判别器额外增加计算判定
        gama = 3 if self.desc in ['loss-d1', 'loss-d2'] and self.ct % self.lamda == 1 else 1
        self.saveLossItem(loss_item,gama)
        # 计算一个epoch平均损失
        if self.ct % (self.lamda*self.batch) == 0:
            loss_epoch = sum(self.loss[-self.lamda*self.batch:])/(self.lamda*self.batch)
            self.loss_epoch.append(loss_epoch)
            self.saveLossImage(self.desc)
        return loss_item
