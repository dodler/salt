import random

import torch
import torch.nn as nn
import pandas as pd
import os.path as osp
from dataset.dataset import GenericXYDataset
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from utils.ext import reflect_center_pad
from utils.lovasz import lovasz_hinge

import logging as l

logFormatter = l.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = l.getLogger()
logger.setLevel(l.DEBUG)

fileHandler = l.FileHandler('log.txt')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = l.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

TRAIN_IMAGES_PATH = '/root/data/salt/train/images/'
TRAIN_MASKS_PATH = '/root/data/salt/train/masks/'
TEST_IMGS_PATH = '/root/data/salt/test/images'

x_offset = 14
x_end_offset = 13
y_offset = 13
y_end_offset = 14

NODE='ggl'
# NODE='gg'


class SegmentationDataset(GenericXYDataset):
    def read_x(self, index):
        if self.mode == 'train':
            reader = self.x_reader(osp.join(TRAIN_IMAGES_PATH, self.train_x[index]))
            return reader
        else:
            return self.x_reader(osp.join(TRAIN_IMAGES_PATH, self.val_x[index]))

    def read_y(self, index):
        if self.mode == 'train':
            return self.y_reader(osp.join(TRAIN_MASKS_PATH, self.train_y[index]))
        else:
            return self.y_reader(osp.join(TRAIN_MASKS_PATH, self.val_y[index]))


class PandasProvider:
    def __init__(self, df):
        self.files = df
        self.index = 0

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index > len(self.files) - 1:
            raise StopIteration
        obj = self.files.iloc[self.index][0] + '.png'
        self.index += 1
        return obj, obj


class SegmentationPathProvider(object):  # rename to names provider
    def __init__(self, paths_csv):
        self.files = pd.read_csv(paths_csv)
        self.index = 0

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index > len(self.files) - 1:
            raise StopIteration
        obj = self.files.iloc[self.index][0] + '.png'
        print(obj)
        self.index += 1
        return obj, obj


class TestSingleChannelToTensor():
    def __call__(self, img, mask):
        return torch.from_numpy(img[np.newaxis, :, :]), \
               torch.from_numpy(mask[np.newaxis, :, :])


class TestToTensor(object):
    def __call__(self, img, mask):
        print(mask.shape)
        m = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(img[:, :, (2, 1, 0)].astype(np.float32)).permute(2, 0, 1), \
               torch.from_numpy(m[np.newaxis, :, :]).float()


class TestReader(object):
    def __call__(self, path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

bce = nn.BCELoss()
mse = nn.MSELoss()
bce_with_logits = nn.BCEWithLogitsLoss()


def myloss(x, y):
    return bce_with_logits(x.squeeze(), y.squeeze())# + 0.7 * lovasz_hinge(x.squeeze(), y.squeeze())


def lovasz(logits, gt):
    return lovasz_hinge(logits.squeeze(), gt.squeeze())


THRESH = 0.5


def iou(img_true, img_pred):
    i = np.sum((img_true * img_pred) > 0)
    u = np.sum((img_true + img_pred) > 0)
    if u == 0:
        return u
    return i / u


SMOOTH = 1e-6


def iou_numpy(outputs, labels):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs.cpu().detach().numpy() > THRESH).astype(np.uint8)
    labels = (labels.cpu().detach().numpy() > THRESH).astype(np.uint8)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded.mean()


def mymetric(x, y, threshold=THRESH):
    m = (x > threshold).float()
    pred_t = m.view(-1).float()
    target = y.view(-1).float()
    inter = 2 * (pred_t * target).sum()
    union = (pred_t + target).sum()

    return (inter / union).cpu().item();


rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
norm = Normalize(rgb_mean, rgb_std)

test_to_tensor = TestToTensor()


class Sharpen():
    def __call__(self, img):
        return cv2.filter2D(img, -1, kernel)


class Randomize():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask):
        if random.random() > 0.5:
            return self.transform(img, mask)
        else:
            return img, mask


kernel_emboss_1 = np.array([[0, -1, -1],
                            [1, 0, -1],
                            [1, 1, 0]])
kernel_emboss_2 = np.array([[-1, -1, 0],
                            [-1, 0, 1],
                            [0, 1, 1]])
kernel_emboss_3 = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, -1]])


def random_filter():
    return random.choice([kernel_emboss_1, kernel_emboss_2, kernel_emboss_3])


def overlay(img, mask):
    result = mask.copy()
    cv2.addWeighted(img.copy(), 0.5, mask, 0.5, 0, result)
    return result


class EmbossAndOverlay():
    def __call__(self, img):
        filtr = random_filter()
        emboss = cv2.filter2D(img, -1, filtr)
        return overlay(img, emboss)


class Gray():
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class AddDim():
    def __call__(self, img):
        return img[..., np.newaxis]


class To01():
    def __call__(self, img):
        return img / 255.0


class RandomFlip():
    def __call__(self, img, mask):
        how = random.choice([0, 1, -1])
        return cv2.flip(img, how), cv2.flip(mask, how)


class RandomBlur():
    def __call__(self, img):
        if random.random() > 0.5:
            return cv2.medianBlur(img, 1 + pow(2, random.randint(1, 5)))
        else:
            return img


class CustomPad:
    def __call__(self, img, mask):
        return reflect_center_pad(img), reflect_center_pad(mask, mask=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_loader(dataset, mode, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def get_directory():
    if NODE == 'ggl':
        return '/home/avn8068/salt/'
    else:
        return '/root/data/salt/'
