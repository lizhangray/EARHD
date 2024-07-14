import torchvision.transforms as transforms
import glob
import os
from PIL import Image
from shutil import copy


def createDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def aug_img(hazy, gt, method):
    list = {
        'Horizon': transforms.RandomHorizontalFlip(p=1),
        'Vertical': transforms.RandomVerticalFlip(p=1)
    }

    return list[method](hazy), list[method](gt)


def main():
    thick_hazy = glob.glob('/home/kzq/pro/sym/datasets/2020NHHAZE_Split/hazy/*.*')
    thick_hazy_gt = glob.glob('/home/kzq/pro/sym/datasets/2020NHHAZE_Split/hazy_gt/*.*')

    target_thick_hazy_dir = '/home/kzq/pro/sym/datasets/2020NHHAZE_Split_Aug/auged_hazy/'
    target_gt_dir = '/home/kzq/pro/sym/datasets/2020NHHAZE_Split_Aug/auged_gt/'

    createDir(target_thick_hazy_dir)
    createDir(target_gt_dir)
    startIndex = 56
    # endIndex = 98 + 24
    for index in range(0, len(thick_hazy)):
        thick_hazy_im = Image.open(thick_hazy[index])
        gt_im = Image.open(thick_hazy_gt[index])

        copy(thick_hazy[index], target_thick_hazy_dir)
        copy(thick_hazy_gt[index], target_gt_dir)

        thick_hazy_im_horizon, gt_im_horizon = aug_img(
            thick_hazy_im, gt_im, 'Horizon')
        thick_hazy_im_vertical, gt_im_vertical = aug_img(
            thick_hazy_im, gt_im, 'Vertical')

        thick_hazy_im_horizon.save(
            f'{target_thick_hazy_dir}/{startIndex}_hazy.png')
        gt_im_horizon.save(f'{target_gt_dir}/{startIndex}_GT.png')
        startIndex += 1
        thick_hazy_im_vertical.save(
            f'{target_thick_hazy_dir}/{startIndex}_hazy.png')
        gt_im_vertical.save(f'{target_gt_dir}/{startIndex}_GT.png')
        startIndex += 1
        
        # if(startIndex == endIndex):
        #     break
main()
