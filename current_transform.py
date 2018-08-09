from generic_utils.segmentation.abstract import DualCompose
from generic_utils.segmentation.abstract import ImageOnly
from generic_utils.segmentation.dualcolor import RandomGamma, OCVGrayscale
from generic_utils.segmentation.dualcrop import DualRotatePadded, DualCrop
from generic_utils.segmentation.util_transform import *
from torchvision.transforms import *

from common import RandomFlip, Gray, UnsqueezeLeft, To01, UnsqueezeRight, Squeeze

CROP_SIZE = 96
UNET_CROP_SIZE = 96

RESIZE_DIM = 128
PADDING = 1
MAX_ROTATE_ANGLE = 10

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)


class MyTransform(object):
    def __init__(self):

        self.train_transform = DualCompose([
            # RandomGamma(0.9, 1.1),
            # Randomize(ImageOnly(EmbossAndOverlay())),
            # Randomize(ImageOnly(Sharpen())),
            DualCrop(CROP_SIZE),
            RandomFlip(),
            # ImageOnly(RandomBlur()),
            DualToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])

        self.val_transform = DualCompose([
            DualCrop(CROP_SIZE),
            DualToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])

    def __call__(self, x, y, mode):
        if mode == 'train':
            return self.train_transform(x, y)
        else:
            return self.val_transform(x, y)


class AlbunetTransform(object):
    def __init__(self):

        self.train_transform = DualCompose([
            DualResize((128, 128)),
            RandomFlip(),
            DualToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])

        self.val_transform = DualCompose([
            DualCrop(CROP_SIZE),
            DualToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])

    def __call__(self, x, y, mode):
        if mode == 'train':
            return self.train_transform(x, y)
        else:
            return self.val_transform(x, y)


class UnetTransform(object):
    def __init__(self):

        self.train_transform = DualCompose([
            ImageOnly(Gray()),
            ImageOnly(UnsqueezeRight()),
            DualCrop(96),
            RandomFlip(),
            ImageOnly(UnsqueezeLeft()),
            ImageOnly(To01()),
            DualSingleChannelToTensor(),
            ImageOnly(Normalize((0.5,), (0.5,)))
        ])

        self.val_transform = DualCompose([
            ImageOnly(Gray()),
            ImageOnly(UnsqueezeRight()),
            DualCrop(96),
            ImageOnly(Squeeze()),
            ImageOnly(UnsqueezeLeft()),
            ImageOnly(To01()),
            DualSingleChannelToTensor(),
            ImageOnly(Normalize((0.5,), (0.5,)))
        ])

    def __call__(self, x, y, mode):
        if mode == 'train':
            return self.train_transform(x, y)
        else:
            return self.val_transform(x, y)
