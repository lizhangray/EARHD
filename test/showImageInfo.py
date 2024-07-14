import glob
from PIL import Image
import torch
from torchvision.transforms import ToTensor


def showImageInfo(dir):
    file_list = glob.glob(dir)
    for path in file_list:
        img = Image.open(path)
        tensor_img = ToTensor(img)
        print(tensor_img.shape)


showImageInfo("datasets/2018OHAZE/train/gt_edge/*.*")
