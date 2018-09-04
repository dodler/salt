from generic_utils.segmentation.abstract import DualCompose, MaskOnly
from generic_utils.segmentation.abstract import ImageOnly
from generic_utils.segmentation.dualcrop import DualCrop, DualPad
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


class MyTransform(object):
    def __init__(self):

        self.train_transform = DualCompose([
            # Randomize(ImageOnly(EmbossAndOverlay())),
            Rotate(15),
            RandomFlip(),
            TestToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])

        self.val_transform = DualCompose([
            TestToTensor(),
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
            # ImageOnly(RandomGamma(0.3)),
            CustomPad(),
            Rotate(20),
            TestToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])
        self.val_transform = DualCompose([
            CustomPad(),
            TestToTensor(),
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
            Randomize(ImageOnly(RandomBlur())),
            RandomFlip(),
            ImageOnly(Gray()),
            ImageOnly(UnsqueezeRight()),
            DualCrop(96),
            RandomFlip(),
            ImageOnly(UnsqueezeLeft()),
            ImageOnly(To01()),
            MaskOnly(Binarize()),
            DualSingleChannelToTensor(),
            ImageOnly(Normalize((0.5,), (0.5,))),
        ])

        self.val_transform = DualCompose([
            ImageOnly(Gray()),
            ImageOnly(UnsqueezeRight()),
            DualCrop(96),
            ImageOnly(Squeeze()),
            ImageOnly(UnsqueezeLeft()),
            ImageOnly(To01()),
            MaskOnly(Binarize()),
            DualSingleChannelToTensor(),
            ImageOnly(Normalize((0.5,), (0.5,)))
        ])

    def __call__(self, x, y, mode):
        if mode == 'train':
            return self.train_transform(x, y)
        else:
            return self.val_transform(x, y)
