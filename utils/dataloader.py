import glob
import os
import random
from random import randrange

import torchvision.transforms as transforms
import torchvision.transforms as tfs
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (Compose, Grayscale, Normalize, Resize,
                                    ToTensor)
from torchvision.transforms import functional as FF

from constant import (NoResizeNoNormTransform, NoResizeNormTransform,
                      ResizeNoNormTransform, ResizeNormTransform)
from option import args
from utils.utils import crop_by_multiple, setup_seed

setup_seed(20)

normalTransform = Compose(
    [transforms.Resize([args.h_size, args.w_size], Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

noResizeTransfrom = Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

singleChannelTransform = Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
singleChannelTransform_x4 = Compose(
    [transforms.Resize([128, 128], Image.Resampling.BICUBIC),
     transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
resize800Transforms = Compose(
    [transforms.Resize([800, 600], Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

downScale_x2 = Compose(
    [transforms.Resize([int(args.h_size/2), int(args.w_size/2)], Image.Resampling.BICUBIC),
     transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

downScale_x4 = Compose(
    [transforms.Resize([int(args.h_size/4), int(args.w_size/4)], Image.Resampling.BICUBIC),
     transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


print(args)


class TestData(Dataset):
    def __init__(self, args, crop_mult=-1, resize=False):
        self.crop_mult = crop_mult
        self.resize = resize
        self.test_haze_path = f'input/haze/'
        self.test_gt_path = f'input/gt/'
        fpaths = glob.glob(os.path.join(self.test_haze_path, f'*.png'))
        self.haze_names = []
        self.gt_names = []
        for path in fpaths:
            self.haze_names.append(path.split('/')[-1])
            gt = path.split('/')[-1].split('_')[0]
            gt = gt + '_GT'
            self.gt_names.append(str(gt)+f'.png')

    def __getitem__(self, index):
        hazeImage = Image.open(self.test_haze_path +
                               self.haze_names[index % len(self.haze_names)])
        gtImage = Image.open(self.test_gt_path +
                             self.gt_names[index % len(self.gt_names)])

        if args.test_only:
            gtItem = crop_by_multiple(gtImage, 16)
            hazeItem = crop_by_multiple(hazeImage, 16)
            gtItem = NoResizeNormTransform(gtImage)
            hazeItem = NoResizeNormTransform(hazeImage)
        else:
            gtItem = NoResizeNormTransform(gtImage)
            hazeItem = NoResizeNormTransform(hazeImage)
        return {
            'image': (hazeItem, gtItem),
            'name': (self.haze_names[index].split('.')[0], self.gt_names[index].split('.')[0])
        }

    def __len__(self):
        return max(len(self.haze_names), len(self.gt_names))


