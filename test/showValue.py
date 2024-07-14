import cv2
import numpy as np
from PIL import Image
import torch
# 使用numpy方式创建一个二维数组
im = torch.rand((3, 1512, 1512))*255
im = im.numpy().transpose((1, 2, 0))

image = Image.fromarray(im.astype(np.uint8))
image.show()
