import os

import numpy
import torch
from matplotlib import pyplot as plt
from PIL import Image
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
from piqa import PSNR, SSIM
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from constant import inv_normalize
from option import args
from utils.utils import reNormalize


def printNetWorkSummary():
    pass


class checkpoint():
    def __init__(self, args=args):
        self.args = args

        self.psnr = []
        self.bestpsnr = -1
        self.bestpsnrepoch = -1
        self.bestpsnr_ssim = -1

        self.ssim = []
        self.bestssim = -1
        self.bestssimepoch = -1
        self.bestssim_psnr = -1

        self.loss = []
        self.logDir = args.logDir
        self.createDir(self.logDir)
        self.index = 0
        self.saveArgs()
        # 在attention块中生成对应的注意力图的计数
        self.att_index = 0

    def saveAttentionMapInBlocks(self, attentionMap, saveInfo=''):
        outputDir = f'{self.logDir}/attentionMap'
        self.createDir(outputDir)
        save_image(
            attentionMap, f'{outputDir}/{self.att_index}-{saveInfo}.png', normalize=True)
        self.att_index = self.att_index + 1

    def saveAttentionMap(self, attentionMap, saveInfo):
        outputDir = f'{self.logDir}/attentionMap'
        self.createDir(outputDir)
        save_image(attentionMap, f'{outputDir}/{saveInfo}.png', normalize=True)

    def saveImage(self, epoch, comment, image1, image2, image3,image4):
        outputDir = f'{self.logDir}/trainImage'
        self.createDir(outputDir)
        output_image1 = make_grid(image1, nrow=5, normalize=True)
        output_image2 = make_grid(image2, nrow=5, normalize=True)
        output_image3 = make_grid(image3, nrow=5, normalize=True)
        output_image4 = make_grid(image4, nrow=5, normalize=True)
        im_grid = torch.cat((output_image1, output_image2,output_image3,output_image4), 1)
        # if image3 is not None:
        #     output_image3 = make_grid(image3, nrow=5, normalize=True)
        #     im_grid = torch.cat((output_image1, output_image2,output_image3), 1)
        # else:
        #     im_grid = torch.cat((output_image1, output_image2), 1)
        save_image(im_grid, f'{outputDir}/{epoch}-{comment}.png', normalize=True)

    # test in the train process
    def saveTempImage(self, batch, epoch, output, origin_size=None):
        outputDir = f'{self.logDir}/temp'
        self.createDir(outputDir)

        # if epoch % 100 == 0:
        save_image(output, f'{outputDir}/{batch}.png', normalize=True)
        return Image.open(f'{outputDir}/{batch}.png')
        # else:
        #     save_image(output, f'{outputDir}/temp.png', normalize=True)
        #     return Image.open(f'{outputDir}/{batch}.png')

    # test-only
    def saveTestImage(self, batch, output, name='test'):
        outputDir = f'{self.logDir}/{name}'
        self.createDir(outputDir)
        save_image(output, f'{outputDir}/{batch}.png', normalize=True)
        # return Image.open(f'{outputDir}/{batch}.png')

    def saveTestMetric(self):
        psnr_avg = numpy.mean(self.psnr)
        ssim_avg = numpy.mean(self.ssim)
        with open(f'{self.logDir}/test.txt', 'a') as f:
            f.write(
                f'size:{self.args.h_size} x {self.args.w_size} psnr:{psnr_avg} ssim: {ssim_avg}\n')

    # 训练过程中保存test的输出图
    def save_valid_img(self, epoch, batch, dehaze, gt):
        outputDir = f'{self.logDir}/test'
        self.createDir(outputDir)
        if dehaze.shape != gt.shape:
            dehaze = transforms.Resize(
                gt.shape[-2:], Image.Resampling.BICUBIC)(dehaze)
        save_image(dehaze, f'{outputDir}/{batch}.png', normalize=True)

    # 计算dehaze tensor和gt tensor的指标
    def metric_tensor(self, dehaze, gt, normalize=False):
        # if normalize:
        #     # tensor1 =
        #     dehaze = inv_normalize(dehaze)
        #     gt = inv_normalize(gt)

        # if dehaze.shape != gt.shape:
        #     dehaze = transforms.Resize(
        #         gt.shape[-2:], Image.Resampling.BICUBIC)(dehaze)

        # dehaze = torch.clamp(dehaze, 0.00001, 0.99999)
        # gt = torch.clamp(gt, 0.00001, 0.99999)
        if normalize:
            dehaze = reNormalize(dehaze)
            gt = reNormalize(gt)
        self.index += 1
        curPSNR = PSNR()(dehaze, gt).item()
        curSSIM = SSIM().cuda()(dehaze, gt).item()
        self.psnr.append(curPSNR)
        self.ssim.append(curSSIM)
        print(f'testing: {self.index}')

    # def metric(self, img1, gt_img):
    #     # img1 = img1.resize(gt_img.size,Image.Resampling.BICUBIC)
    #     if self.args.test_only:
    #         img1 = img1.resize(gt_img.size, Image.Resampling.BICUBIC)

    #         # (h1, w1) = img1.size
    #         # (h2, w2) = gt_img.size
    #         # if h1 < h2 and w1 < w2:
    #         #     img1 = img1.resize((h2, w2), Image.Resampling.BICUBIC)
    #         # elif h1 > h2 and w1 > w2:
    #         #     gt_img = gt_img.resize((h1, w1), Image.Resampling.BICUBIC)
    #         # print(img1.size, gt_img.size)
    #     self.index += 1
    #     img1 = numpy.array(img1)
    #     gt_img = numpy.array(gt_img)
    #     curPSNR = psnr(img1, gt_img)
    #     curSSIM = ssim(img1, gt_img, channel_axis=-1)
    #     self.psnr.append(curPSNR)
    #     self.ssim.append(curSSIM)
    #     print(f'testing: {self.index}')

    def testSave(self, epoch, model):
        modelPath = f'{self.logDir}/model'
        self.createDir(modelPath)
        psnr_avg = numpy.mean(self.psnr)
        ssim_avg = numpy.mean(self.ssim)

        if self.bestpsnr < psnr_avg:
            self.bestpsnr = psnr_avg
            self.bestpsnrepoch = epoch
            self.bestpsnr_ssim = ssim_avg

            torch.save(model.state_dict(),  f'{modelPath}/bestpsnr.pth')

        # if self.bestssim < ssim_avg:
        #     self.bestssim = ssim_avg
        #     self.bestssimepoch = epoch
        #     self.bestssim_psnr = psnr_avg
        #     torch.save(model.state_dict(),  f'{modelPath}/bestssim.pth')

        # torch.save(model.state_dict(),  f'{modelPath}/last.pth')

        with open(f'{self.logDir}/log.txt', 'a') as f:
            f.write(
                f'[epoch: {epoch}] [psnr: {psnr_avg}]  [ssim: {ssim_avg}] \n')

        with open(f'{self.logDir}/last-result.txt', 'w') as f:
            f.writelines(
                f'[bestpsnr-epoch: {self.bestpsnrepoch}] [psnr: {self.bestpsnr}] [ssim: {self.bestpsnr_ssim}]\n')
            # f.writelines(
            #     f'[bestssim-epoch: {self.bestssimepoch}] [psnr: {self.bestssim_psnr}] [ssim: {self.bestssim}]')

        self.psnr = []
        self.ssim = []

    def createDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def saveArgs(self):
        argsDict = self.args.__dict__
        with open(f'{self.logDir}/setting.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
