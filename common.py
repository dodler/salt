import random

import os.path as osp
from dataset.dataset import GenericXYDataset
import os
import cv2
import numpy as np


class OCVMaskReader():
    def __call__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img > 1] = 1
        return img


class SegmentationDataset(GenericXYDataset):
    def read_x(self, index):
        if self.mode == 'train':
            return self.x_reader(osp.join('/home/dhc/salt/images/', self.train_x[index]))
        else:
            return self.x_reader(osp.join('/home/dhc/salt/images/', self.val_x[index]))

    def read_y(self, index):
        if self.mode == 'train':
            return self.y_reader(osp.join('/home/dhc/salt/masks/', self.train_y[index]))
        else:
            return self.y_reader(osp.join('/home/dhc/salt/masks/', self.val_y[index]))


class SegmentationPathProvider(object):
    def __init__(self):
        self.files = os.listdir('/home/dhc/salt/masks/')
        self.index = 0

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index > len(self.files) - 1:
            raise StopIteration
        obj = self.files[self.index]
        self.index += 1
        return obj, obj


kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])


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
