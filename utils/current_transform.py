from generic_utils.segmentation.abstract import DualCompose
from generic_utils.segmentation.abstract import ImageOnly
from generic_utils.segmentation.util_transform import *
from torchvision.transforms import *

from utils.common import *

CROP_SIZE = 96
UNET_CROP_SIZE = 96

RESIZE_DIM = 128
PADDING = 1
MAX_ROTATE_ANGLE = 10

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

to_tensor = ToTensor()

gray_norm = Normalize((0.5,), (0.5,))


class MyTransform(object):
    def __init__(self):

        self.train_transform = DualCompose([
            Randomize(HorizontalFlip()),
            TestSingleChannelToTensor(),
            ImageOnly(gray_norm)
        ])

        self.val_transform = DualCompose([
            TestSingleChannelToTensor(),
            ImageOnly(gray_norm)
        ])

    def __call__(self, x, y, mode):
        if mode == 'train':
            return self.train_transform(x, y)
        else:
            return self.val_transform(x, y)


class AlbunetTransform(object):
    def __init__(self):

        self.train_transform = DualCompose([
            Randomize(RandomGamma(0.2)),
            DualResize((128, 128)),
            HorizontalFlip(),
            TestToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])

        self.val_transform = DualCompose([
            DualResize((128, 128)),
            TestToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])

    def __call__(self, x, y, mode):
        if mode == 'train':
            return self.train_transform(x, y)
        else:
            return self.val_transform(x, y)


class UnetRGBTransform(object):
    def __init__(self):

        self.train_transform = DualCompose([
            Randomize(RandomGamma(0.2)),
            DualResize((128, 128)),
            Randomize(HorizontalFlip()),
            TestSingleChannelToTensor(),
            ImageOnly(gray_norm)
        ])
        self.val_transform = DualCompose([
            DualResize((128, 128)),
            TestSingleChannelToTensor(),
            ImageOnly(gray_norm)
        ])

    def __call__(self, x, y, mode):
        if mode == 'train':
            return self.train_transform(x, y)
        else:
            return self.val_transform(x, y)


class Linknet152Transform(object):
    def __init__(self):

        self.train_transform = DualCompose([
            Randomize(RandomGamma(0.2)),
            DualResize((128,128)),
            Randomize(HorizontalFlip()),
            TestSingleChannelToTensor(),
            ImageOnly(gray_norm)
        ])
        self.val_transform = DualCompose([
            DualResize((128, 128)),
            TestSingleChannelToTensor(),
            ImageOnly(gray_norm)
        ])

    def __call__(self, x, y, mode):
        if mode == 'train':
            return self.train_transform(x, y)
        else:
            return self.val_transform(x, y)
