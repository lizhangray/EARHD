import cv2 as cv
import numpy as np
#USM锐化增强方法(Unsharpen Mask)
#先对原图高斯模糊，用原图减去系数x高斯模糊的图像
#再把值Scale到0~255的RGB像素范围
#优点：可以去除一些细小细节的干扰和噪声，比卷积更真实
#（原图像-w*高斯模糊）/（1-w）；w表示权重（0.1~0.9），默认0.6
src = cv.imread(r"datasets\01_outdoor_GT.jpg")
cv.imshow("input", src)

# sigma = 5、15、25
blur_img = cv.GaussianBlur(src, (0, 0), 5)
usm = cv.addWeighted(src, 1.5, blur_img, -0.5, 0)
#cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
cv.imshow("mask image", usm)

h, w = src.shape[:2]
result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h,0:w,:] = src
result[0:h,w:2*w,:] = usm
cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.putText(result, "sharpen image", (w+10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
#cv.putText(图像名，标题，（x坐标，y坐标），字体，字的大小，颜色，字的粗细）
cv.imshow("sharpen_image", result)

cv.waitKey(0)
cv.destroyAllWindows()