from albumentations import (
    HorizontalFlip, ShiftScaleRotate,
    RandomBrightness, RandomContrast, Compose,
    VerticalFlip)
from generic_utils.segmentation.abstract import DualCompose

from utils.common import Randomize, CustomPad


def strong_aug():
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomBrightness(limit=0.1, p=0.5),
        RandomContrast(limit=0.1, p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0., rotate_limit=0., ),
    ])


aug = strong_aug()


class Albumentation:
    def __call__(self, img, mask):
        res = aug(image=img, mask=mask)
        return res['image'], res['mask']


class MyTransform(object):
    def __init__(self):
        self.train_transform = DualCompose([
            Albumentation(),
            CustomPad(),
        ])

        self.val_transform = DualCompose([
            CustomPad(),
        ])

    def __call__(self, x, y, mode):
        if mode == 'train':
            return self.train_transform(x, y)
        else:
            return self.val_transform(x, y)
