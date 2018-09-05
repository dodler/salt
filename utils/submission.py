import pickle

import cv2
import numpy as np
import os.path as osp
import pandas as pd
from scipy.stats import gmean
from tqdm import *

from train import predict_linknet
from train_unet import predict_unet

from training import predict_multiple
from utils.common import TEST_IMGS_PATH, RLenc

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
# masks = [predict_unet(model, img) for img in tqdm(images)]
masks = predict_multiple([
    'linknet_fold_0_loss_0.15052992.pth.tar', 'linknet_fold_1_loss_0.15408346.pth.tar',
    'linknet_fold_2_loss_0.30064672.pth.tar', 'linknet_fold_3_loss_0.14445938.pth.tar',
    'linknet_fold_5_loss_0.25142106.pth.tar'
], images, predict_linknet, gmean)
threshold_masks = [threshold_mask(mask) for mask in tqdm(masks)]
encodings = [RLenc(img) for img in tqdm(threshold_masks)]
save_to_csv(encodings)
