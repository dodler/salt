from albumentations import (
    HorizontalFlip, ShiftScaleRotate,
    RandomBrightness, RandomContrast, Compose,
    VerticalFlip, Blur, RandomCrop)
from generic_utils.segmentation.abstract import DualCompose

from utils.common import Randomize, CustomPad


def strong_aug():
    return Compose([
        ShiftScaleRotate(shift_limit=0.5,
                         scale_limit=0.4,
                         rotate_limit=15, p=0.9),
        RandomBrightness(limit=0.4, p=0.8),
        RandomContrast(limit=0.4, p=0.8),
        Blur(blur_limit=5, p=0.5),
        HorizontalFlip(p=0.5),
    ])


def light_aug():
    return Compose([
        RandomBrightness(limit=0.1, p=0.2),
        RandomContrast(limit=0.1, p=0.2),
        Blur(blur_limit=3, p=0.2),
        HorizontalFlip(p=0.3),
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
