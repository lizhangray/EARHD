from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torchvision

from option import args


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Cur Model: ', args.model)
        module = import_module('model.' + args.model.lower())
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.model = module.make_model(args).to(self.device)

    def forward(self, x):
        if args.crop:
            forward_fn = self.forward_chop
        else:
            forward_fn = self.model.forward
        return forward_fn(x)

    def crop2OriginSize(self, image, originSize, pad_list):
        top, bottom, left, right = pad_list
        h, w = originSize
        return image[:, :, top:top+h, left:left+w]


    def pad2affine(self, image, mod):
        pad_list = []
        if image.shape[-2] % mod != 0:
            padnum = int((image.shape[-2]//mod+1)*mod-image.shape[-2])
            if padnum % 2 == 0:
                top_pad_num = int(padnum / 2)
                bottom_pad_num = top_pad_num
            else:
                top_pad_num = int((padnum + 1) / 2)
                bottom_pad_num = padnum - top_pad_num
            pad_list.append(top_pad_num)
            pad_list.append(bottom_pad_num)
            pad = torch.nn.ReflectionPad2d((0, 0, top_pad_num, bottom_pad_num))
            image = pad(image)

        if image.shape[-1] % mod != 0:
            padnum = (image.shape[-1]//mod+1)*mod-image.shape[-1]
            if padnum % 2 == 0:
                left_pad_num = int(padnum / 2)
                right_pad_num = left_pad_num
            else:
                left_pad_num = int((padnum + 1) / 2)
                right_pad_num = padnum - left_pad_num
            pad_list.append(left_pad_num)
            pad_list.append(right_pad_num)
            pad = torch.nn.ReflectionPad2d((left_pad_num, right_pad_num, 0, 0))
            image = pad(image)
        return image, pad_list


    def forward_chop1(self, img, win_size=args.h_size):
        img, pad_list = self.pad2affine(img, args.size)
        nn_unflod = nn.Unfold(kernel_size=(win_size),
                              dilation=1, padding=0, stride=win_size)
        patch_img = nn_unflod(img)
        reshape_patch_img = patch_img.view(
            img.shape[0], img.shape[1], win_size, win_size, -1)
        reshape_patch_img = reshape_patch_img.permute(
            4, 0, 1, 2, 3)  # 变成 [L, N, C, k, k]
        # print(reshape_patch_img.shape)
        # img_G = make_grid(torch.clamp(reshape_patch_img.squeeze(), min=-1, max=1), nrow=8, normalize=True)
        # imgpath = "temp/img.jpg"
        # save_image(img_G, imgpath)
        result_img = []
        i = 0
        for patch in zip(reshape_patch_img):
            haze = patch[0].to(self.device)
            dehaze_img = self.model(haze)
            # torch.cuda.empty_cache()
            print(torch.cuda.memory_allocated()/1024/1024/1024)
            print(f'finish: {i}' )
            i += 1
            result_img.append(dehaze_img)
        reshape_patch_img = torch.stack(result_img, dim=0)
        reshape_patch_img = reshape_patch_img.permute(1, 2, 3, 4, 0)
        reshape_patch_img = reshape_patch_img.view(
            img.shape[0], img.shape[1]*win_size*win_size, -1)
        fold = torch.nn.Fold(output_size=(img.shape[2], img.shape[3]), kernel_size=(
            win_size, win_size), stride=win_size)
        re_img = fold(reshape_patch_img)
        crop_img = self.crop2OriginSize(re_img, img.shape[-2:], pad_list)        
        return crop_img

    def forward_chop(self, *args, h_shave=10, w_shave=10, min_size=400000):
        scale = 1
        n_GPUs = 1
        # height, width
        h, w = args[0].size()[-2:]
        while (h // 2 + h_shave)% 16  != 0:
            h_shave += 1
            print(h_shave)

        while (w // 2 + w_shave) % 16 != 0:
            w_shave += 1
            print(w_shave)

        top = slice(0, h//2 + h_shave)
        bottom = slice(h - h//2 - h_shave, h)
        left = slice(0, w//2 + w_shave)
        right = slice(w - w//2 - w_shave, w)
        x_chops = [torch.cat([
            args[0][..., top, left],
            args[0][..., top, right],
            args[0][..., bottom, left],
            args[0][..., bottom, right]
        ])]
        y_chops = []
        if h * w < 4 * min_size:
            print("h*w < 4*min_size")
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                print(x[0].shape)
                y = P.data_parallel(self.model, *x, range(n_GPUs))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                cur = p[0].unsqueeze(0)
                y = self.forward_chop(cur, 10, 10, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]
        return y

    def load_state_dict(self, dic):
        self.model.load_state_dict(dic)

    def state_dict(self):
        return self.model.state_dict()