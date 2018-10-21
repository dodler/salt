import os
import cv2
import os.path as osp
import pandas as pd
import torch
import numpy as np
from models.salt_models import LinkNet34, UNet16, AlbuNet, Linknet152
from models.vanilla_unet import UNet
from training import Trainer
from utils.common import myloss, iou_numpy, \
    count_parameters, get_loader, lovasz, get_directory
from utils.ush_dataset import TGSSaltDataset

from albumentations import (
    VerticalFlip,
    HorizontalFlip,
    Compose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    Resize)
torch.manual_seed(42)
np.random.seed(42)

MODEL_NAME = 'unet_128'

original_height = 101
original_width = 101

aug = Compose([
    HorizontalFlip(p=0.7),
    RandomGamma(p=0.7),
    # RandomBrightness(p=0.7),
    Resize(width=128,height=128, interpolation=cv2.INTER_LANCZOS4),
    GridDistortion(p=0.6),
    OpticalDistortion(p=0.6),
    ElasticTransform(p=0.6),
])
# aug = strong_aug()


# aug = Compose([
#     VerticalFlip(p=0.1),
#     HorizontalFlip(p=0.5),
#     RandomGamma(p=0.3)])

if __name__ == '__main__':

    directory = get_directory()
    n_fold = 8
    depths = pd.read_csv(os.path.join(directory, 'depths.csv'))
    depths.sort_values('z', inplace=True)
    depths['fold'] = (list(range(n_fold)) * depths.shape[0])[:depths.shape[0]]
    depths.head()

    train_path = os.path.join(directory, 'train', 'images')
    test_path = os.path.join(directory, 'test', 'images')
    y_train = os.path.join(directory, 'train', 'masks')

    file_list = list(depths['id'].values)

    train_images = os.listdir(train_path)
    train = pd.DataFrame(train_images, columns=['id'])
    train.id = train.id.apply(lambda x: x[:-4]).astype(str)
    train = pd.merge(train, depths, on='id', how='left')
    train[:3]

    DEVICE = 0
    EPOCHS = 1000
    BATCH_SIZE = 24

    model = UNet().type(torch.float).to(DEVICE)

    print(count_parameters(model))

    current_val_fold = 0
    x_train = train[train.fold != current_val_fold].id.values
    x_val = train[train.fold == current_val_fold].id.values

    train_dataset = TGSSaltDataset(osp.join(directory, 'train'), x_train, is_test=False,
                                   is_val=False, augment_func=aug)

    val_dataset = TGSSaltDataset(osp.join(directory, 'train'), x_val,
                                 is_test=False, is_val=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(myloss, iou_numpy, optimizer, MODEL_NAME, None, DEVICE)

    train_loader = get_loader(train_dataset, 'train', BATCH_SIZE)
    val_loader = get_loader(val_dataset, 'val', BATCH_SIZE)

    for i in range(EPOCHS):
        trainer.train(train_loader, model, i)
        trainer.validate(val_loader, model)