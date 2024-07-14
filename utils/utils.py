import datetime
import os
from glob import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from skimage.metrics import structural_similarity as ssim
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import (Compose, Grayscale, Normalize, Resize,
                                    ToTensor)
from torchvision.utils import save_image
import itertools


def loadWeights(url, dir):
    # print(url, dir)
    if not url.startswith('https'):
        return torch.load(f'{dir}/{url}.pth')
    # return torch.utils.model_zoo.load_url(url, dir)


def make_optimizer(args, my_model, lr=None):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}

    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = lr if lr else args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler


def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_edge_image(image, name, savePath):
    # edge = cv2.Canny(image, 32, 128)
    createDir(savePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=3, scale=1)
    y = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=3, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    cv2.imwrite(f'{savePath}/{name}', edge)


def generate_edge_image(sourceDir, targetDir):
    file_list = glob(f'{sourceDir}/*.*')
    ct = 0
    for path in file_list:
        _, name = os.path.split(path)
        img = cv2.imread(path)
        get_edge_image(img, name, targetDir)
        ct += 1
        print(f"{ct}/{len(file_list)}")


def cal_ssim_edge(sourceDir, targetDir):
    source_list = glob(f'{sourceDir}/*.*')
    target_list = glob(f'{targetDir}/*.*')
    ssim_list = []
    for i in range(0, len(source_list)):
        source = cv2.imread(source_list[i], -1)
        target = cv2.imread(target_list[i], -1)
        cur_ssim = ssim(source, target)
        ssim_list.append(cur_ssim)
        # print(i, cur_ssim)
    avg_ssim = np.mean(ssim_list)
    # print(avg_ssim)

def crop_by_multiple(img, mul):
    """
        img: PIL Image
        mul: 倍数处理 将img的尺寸处理成mul的倍数
    """
    width, height = img.size
    crop_width_right = width - width % mul
    crop_height_bottom = height - height % mul
    crop_img = img.crop((0, 0, crop_width_right, crop_height_bottom))
    return crop_img


def getTiansiOperator(channels, v):
    a0 = 1
    a1 = -v
    a2 = (v**2-v)/2
    m = 8-12*v + 4*(v**2)
    kernel = torch.tensor([
        [a2, 0, a2, 0, a2],
        [0, a1, a1, a1, 0],
        [a2, a1, 8*a0, a1, a2],
        [0, a1, a1, a1, 0],
        [a2, 0, a2, 0, a2]
    ])
    return kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)


def frac_conv(channels, v):
    kernel_size = 5
    frac_filter = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                            groups=channels, bias=False, padding=kernel_size//2)
    frac_filter.weight.data = getTiansiOperator(channels, v)

    frac_filter.weight.requires_grad = False
    return frac_filter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reNormalize(tensor):
    tensor = tensor.clone()
    value_range = None

    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))
        return img

    def norm_range(t, value_range):

        if value_range is not None:

            return norm_ip(t, value_range[0], value_range[1])
        else:
            return norm_ip(t, float(t.min()), float(t.max()))

    return norm_range(tensor, value_range)
# generate_edge_image('/home/kzq/pro/sym/datasets/2020NHHAZE/train/haze', '/home/kzq/pro/sym/datasets/2020NHHAZE/train/gt-sobel/')