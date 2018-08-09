from models.segmentation.models import LinkNet34
from torchvision.transforms import Normalize
from tqdm import *
import os.path as osp
import cv2
import pandas as pd
import torch
import numpy as np

from common import RLenc
from train_unet import predict_unet, UNet

checkpoint_name = '/tmp/salt/unet_loss_0.2864511013031006.pth.tar'
model = UNet(1,1).float().to(0)
model.load_state_dict(torch.load(checkpoint_name))

subm = pd.read_csv('sample_submission.csv')
rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
norm = Normalize(rgb_mean, rgb_std)

CROP_SIZE = 64
IMG_SIZE = 101
THRESH = 0.62


def read_img(img_name):
    return cv2.imread(osp.join('/home/dhc/salt/test/images', img_name + '.png'))


def transform(img):
    img_tensor = torch.from_numpy(img[:, :, (2, 1, 0)].astype(np.float32) / 255.0).permute(2, 0, 1)
    return norm(img_tensor)


def get_top_left_crop(image):
    return image.copy()[0:CROP_SIZE, 0:CROP_SIZE, :]


def get_top_right_crop(image):
    start = IMG_SIZE - CROP_SIZE
    return image.copy()[start:IMG_SIZE, 0:CROP_SIZE, :]


def get_bot_left_crop(image):
    start_y = IMG_SIZE - CROP_SIZE
    return image.copy()[0:CROP_SIZE, start_y:IMG_SIZE]


def get_bot_right_crop(image):
    start_x = start_y = IMG_SIZE - CROP_SIZE
    return image.copy()[start_x:IMG_SIZE, start_y:IMG_SIZE, :]


def merge_crops(top_left, top_right, bot_left, bot_right):
    start_x = start_y = IMG_SIZE - CROP_SIZE
    result = np.zeros((101, 101))
    result[0:CROP_SIZE, 0:CROP_SIZE] = top_left
    result[0:CROP_SIZE, start_y:IMG_SIZE] = bot_left
    result[start_x:IMG_SIZE, 0:CROP_SIZE] = top_right
    result[start_x:IMG_SIZE, start_y:IMG_SIZE] = bot_right
    return result


def predict_crop(crop, model, transform):
    mask = model(transform(crop).unsqueeze(0).to(0))
    mask = mask.squeeze().cpu().detach()
    return mask.numpy()


def predict_image(image, model, transform):
    top_left = predict_crop(get_top_left_crop(image), model, transform)
    top_right = predict_crop(get_top_right_crop(image), model, transform)
    bot_left = predict_crop(get_bot_left_crop(image), model, transform)
    bot_right = predict_crop(get_bot_right_crop(image), model, transform)

    return merge_crops(top_left, top_right, bot_left, bot_right)


def threshold_mask(mask):
    t = mask.copy()
    t[t > THRESH] = 1
    t[t <= THRESH] = 0
    return t.astype(np.int8)


def save_to_csv(encs):
    subm = pd.read_csv('sample_submission.csv')
    subm.rle_mask = encs
    subm.to_csv('my_subm.csv', index=False)


images = list(map(read_img, tqdm(subm['id'])))
masks = list(map(lambda img: predict_unet(model, img), tqdm(images)))
threshold_masks = list(map(threshold_mask, tqdm(masks)))
encodings = [RLenc(img) for img in tqdm(threshold_masks)]
save_to_csv(encodings)
