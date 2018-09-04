import pickle
import torch

import numpy as np
import os
import os.path as osp
import cv2
from reader.image_reader import OpencvReader
from torch.utils.data import DataLoader
from tqdm import *
from common import TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, SegmentationDataset, SegmentationPathProvider, mymetric
from current_transform import AlbunetTransform
from train_unet import predict_unet
from train_unet import UNet
from albunet import AlbuNet, predict_albunet
from training import Trainer
import torch.nn as nn


checkpoint_name = '/tmp/pycharm_project_959/unet_loss_0.039061274.pth.tar'
checkpoint_name = '/tmp/pycharm_project_959/albunet_loss_0.11421003.pth.tar'

model = pickle.load(open(checkpoint_name, 'rb')).to(0)
dataset = SegmentationDataset(AlbunetTransform(), SegmentationPathProvider(), x_reader=OpencvReader(),
                                  y_reader=OpencvReader())

trainer = Trainer(None, mymetric, torch.optim.Adam(model.parameters(), lr=1e-4), 'albunet', 0)

train_loader = DataLoader(dataset, batch_size=1)
dataset.setmode('val')
val_loader = DataLoader(dataset, batch_size=1)
dataset.setmode('train')
print(trainer.optimize_threshold(val_loader, model, np.arange(0,1,0.01),mymetric))