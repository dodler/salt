import random

import torch
import torch.nn as nn
import pandas as pd
import os.path as osp
from dataset.dataset import GenericXYDataset
import os
import cv2
import numpy as np
from torchvision.transforms import Normalize

TRAIN_IMAGES_PATH = '/root/data/train/images/'
TRAIN_MASKS_PATH = '/root/data/train/masks/'
TEST_IMGS_PATH = '/root/data/test/images'


def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
    else:
        img = np.power(img, gamma)

    return img


class RandomGamma:
    def __init__(self, max_amp):
        self.max_amp = max_amp

    def __call__(self, img):
        gamma = random.uniform(1 - self.max_amp, 1 + self.max_amp)
        return gamma_transform(img, gamma)


class SegmentationDataset(GenericXYDataset):
    def read_x(self, index):
        if self.mode == 'train':
            return self.x_reader(osp.join(TRAIN_IMAGES_PATH, self.train_x[index]))
        else:
            return self.x_reader(osp.join(TRAIN_IMAGES_PATH, self.val_x[index]))

    def read_y(self, index):
        if self.mode == 'train':
            return self.y_reader(osp.join(TRAIN_MASKS_PATH, self.train_y[index]))
        else:
            return self.y_reader(osp.join(TRAIN_MASKS_PATH, self.val_y[index]))


class SegmentationPathProvider(object): # rename to names provider
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
        self.index += 1
        return obj, obj


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return result


class Rotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, mask):
        target_angle = random.randint(-self.angle, self.angle)
        return rotateImage(img, target_angle), rotateImage(mask, target_angle)


class TestToTensor(object):
    def __call__(self, img, mask):
        mask = mask[:, :, 0:1] // 255
        img = img / 255.0
        return torch.from_numpy(img).float().permute([2, 0, 1]), \
               torch.from_numpy(mask).float().permute([2, 0, 1])


kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

bce = nn.BCELoss()
mse = nn.MSELoss()
bce_with_logits = nn.BCEWithLogitsLoss()


def myloss(x, y):
    return bce_with_logits(x.squeeze(), y.squeeze())


THRESH = 0.5


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


class PrintShape():
    def __call__(self, img, mask):
        print(img.shape, mask.shape)
        return img, mask


class Gray:
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class UnsqueezeLeft():
    def __call__(self, img):
        return img[np.newaxis, ...]


class UnsqueezeRight():
    def __call__(self, img):
        return img[..., np.newaxis]


class Squeeze():
    def __call__(self, img):
        return img.squeeze()


class To01():
    def __call__(self, img):
        return img / 255.0


class RandomFlip():
    def __call__(self, img, mask):
        how = random.choice([0, 1, -1])
        return cv2.flip(img, how), cv2.flip(mask, how)


class RandomBlur():
    def __call__(self, img):
        return cv2.medianBlur(img, 1 + pow(2, random.randint(1, 5)))


class Binarize:
    def __call__(self, mask):
        t = mask.copy()
        t[t > 0] = 1
        return t


class CustomPad:
    def __call__(self, img, mask):
        p = 13
        pad_img = cv2.copyMakeBorder(img, p + 1, p, p + 1, p, cv2.BORDER_REFLECT_101)
        pad_mask = cv2.copyMakeBorder(mask, p + 1, p, p, p + 1, cv2.BORDER_CONSTANT, value=0)
        return pad_img, pad_mask


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

# pred_dict = {id_[:-4]:RLenc(np.round(preds_test_upsampled[i] > best_thres)) for i,id_ in tqdm_notebook(enumerate(test_ids))}
