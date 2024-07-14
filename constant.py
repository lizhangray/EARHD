from option import args
from torchvision import transforms
import torchvision
import torch
from PIL import Image

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)


# inv_normalize = NormalizeInverse(mean, std)

ResizeNormTransform = transforms.Compose([transforms.Resize(
    [args.h_size, args.w_size], Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean, std)])

ResizeNoNormTransform = transforms.Compose([transforms.Resize(
    [args.h_size, args.w_size], Image.Resampling.BICUBIC), transforms.ToTensor()])

NoResizeNormTransform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

NoResizeNoNormTransform = transforms.Compose([transforms.ToTensor()])
