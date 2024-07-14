import cv2
import numpy as np

from utils.utils import generate_edge_image
 
# generate_edge_image("datasets/2018OHAZE/train/haze", "datasets/2018OHAZE/train/haze_edge")
def get_edge_image(image):
    edge = cv2.Canny(image, 32, 128)
    cv2.imshow(edge)
    cv2.waitKey(0)
    # createDir(savePath)
    # cv2.imwrite(f'{savePath}/{name}', edge)
im = cv2.imread('dataset/2018OHAZE/train/gt/01_outdoor_GT.jpg')
get_edge_image(im)