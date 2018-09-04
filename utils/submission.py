import pickle
import random

import cv2
import numpy as np
import os.path as osp
import pandas as pd
import torch
from PIL import Image
from generic_utils.visualization.visualization import VisdomValueWatcher
from tqdm import *

from albunet import predict_albunet
from common import RLenc, TEST_IMGS_PATH
from salt_models import AlbuNet, LinkNet34
from train import predict_linknet
from train_unet import predict_unet
from albunet import AlbuNet
from models.segmentation.unet.unet11 import UNet11


# checkpoint_name = '/tmp/pycharm_project_959/linknet_loss_0.026078532.pth.tar'
# checkpoint_name = '/tmp/pycharm_project_959/linknet_loss_0.054118212.pth.tar'
# checkpoint_name = '/tmp/pycharm_project_959/linknet_loss_0.07725099.pth.tar'
checkpoint_name = '/tmp/pycharm_project_959/linknet_loss_0.032785438.pth.tar'
checkpoint_name = '/tmp/pycharm_project_959/linknet_no_resize_loss_0.15353289.pth.tar'
checkpoint_name = '/tmp/pycharm_project_959/unet_loss_0.10436103.pth.tar'

model = pickle.load(open(checkpoint_name, 'rb')).to(0)

THRESH = 0.7
subm = pd.read_csv('/root/data/sample_submission.csv')


def read_img(img_name):
    return cv2.imread(osp.join(TEST_IMGS_PATH, img_name + '.png'))


def threshold_mask(mask, thresh=THRESH):
    cpy = mask.copy()
    cpy[cpy > thresh] = 1
    cpy[cpy <= thresh] = 0
    return cpy.astype(np.int)


def save_to_csv(encs):
    subm.rle_mask = encs
    subm.to_csv('my_subm.csv', index=False)


images = [read_img(path) for path in tqdm(subm['id'])]
masks = [predict_unet(model, img) for img in tqdm(images)]
threshold_masks = [threshold_mask(mask) for mask in tqdm(masks)]
encodings = [RLenc(img) for img in tqdm(threshold_masks)]
save_to_csv(encodings)
