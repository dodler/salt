import pickle

import cv2
import numpy as np
import os.path as osp
import pandas as pd
from generic_utils.segmentation import dense_crf
from scipy.stats import gmean
from tqdm import *

from test_metric import get_iou_vector
from train import predict_linknet
from train_unet import predict_unet

from training import predict_multiple
from utils.common import TEST_IMGS_PATH, RLenc, TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, cut_target

THRESH = 0.4


def read_img(img_name, path=TEST_IMGS_PATH):
    return cv2.imread(osp.join(path, img_name + '.png'))


def threshold_mask(mask, thresh=THRESH):
    cpy = mask.copy()
    cpy[cpy > thresh] = 1
    cpy[cpy <= thresh] = 0
    return cpy.astype(np.int)


def save_to_csv(encs, subm):
    subm.rle_mask = encs
    subm.to_csv('my_subm.csv', index=False)


def make_raw_predict(subm, ckpt_name):
    model = pickle.load(open(ckpt_name, 'rb')).float().to(1)
    images = [read_img(path) for path in tqdm(subm['id'])]
    return [predict_unet(model, img) for img in tqdm(images)]


def inference(ckpt_name):
    subm = pd.read_csv('/root/data/sample_submission.csv')
    masks = make_raw_predict(subm, ckpt_name)
    make_submit(masks, subm)
    return masks


def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_submit(raw_masks, subm):
    threshold_masks = [threshold_mask(mask) for mask in tqdm(raw_masks)]
    encodings = [rle_encode(img) for img in tqdm(threshold_masks)]
    save_to_csv(encodings, subm)


def dense(img, mask_prob):
    return dense_crf(img, threshold_mask(cut_target(mask_prob)[:, :, np.newaxis]).astype(np.float32))


def optimize_threshold(ckpt_name, ):
    model = pickle.load(open(ckpt_name, 'rb')).float().to(1)
    train_ids = pd.read_csv('/root/data/train.csv').sample(n=3000)
    images = [read_img(path, TRAIN_IMAGES_PATH) for path in tqdm(train_ids['id'])]
    gt_masks = [read_img(path, TRAIN_MASKS_PATH) for path in tqdm(train_ids['id'])]
    gt_masks = [m[:, :, 0:1] // 255 for m in gt_masks]
    masks = [predict_unet(model, img) for img in tqdm(images)]

    max_metric = 0
    max_thresh = 0
    for i in tqdm(range(20, 100)):
        thrsh = float(i) / 100.0
        iou = get_iou_vector(np.array(gt_masks), np.array([threshold_mask(mask, thrsh) for mask in masks]))
        if iou > max_metric:
            max_metric = iou
            max_thresh = thrsh

    return masks, max_metric, max_thresh


ckpt_path = 'unet_loss_0.13917223.pth.tar'
masks, metric, thresh = optimize_threshold(ckpt_path)
print(thresh, metric)
THRESH = thresh
# make_submit(pickle.load(open('test_prediction_' + ckpt_path + '_.pkl', 'rb')),
#             pd.read_csv('/root/data/sample_submission.csv'))
test_masks = inference(ckpt_path)
pickle.dump(masks, open('prediction_' + ckpt_path + '_.pkl', 'wb'))
pickle.dump(test_masks, open('test_prediction_' + ckpt_path + '_.pkl', 'wb'))
