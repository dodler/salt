import os

import pandas as pd
import torch

from models.gcn import GCN
from models.salt_models import Linknet152, AlbuNet, UNet16, LinkNet34
from models.vanilla_unet import UNet
from training import Trainer
from utils.common import myloss, iou_numpy, \
    count_parameters, get_loader, lovasz, get_directory
from utils.current_transform import strong_aug, light_aug
from utils.ush_dataset import TGSSaltDataset
import os.path as osp

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness
)

MODEL_NAME = 'gcn'

original_height = 101
original_width = 101

aug = Compose([
    VerticalFlip(p=0.1),
    HorizontalFlip(p=0.5),
    RandomGamma(p=0.3)])

# aug=light_aug()


aug = Compose([
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    OneOf([
        ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(p=0.5),
        OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
    ], p=0.8),
    CLAHE(p=0.8),
    RandomContrast(p=0.8),
    RandomBrightness(p=0.8),
    RandomGamma(p=0.8)])


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

    DEVICE = 1
    EPOCHS = 300
    BATCH_SIZE = 64

    model = GCN(input_size=(128,128),num_classes=1).type(torch.float).to(DEVICE)

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

    EPOCHS=500
    trainer.optimizer=torch.optim.SGD(model.parameters(), lr=1e-4 / 2.0, momentum=0.9)

    for i in range(EPOCHS):
        trainer.train(train_loader, model, i)
        trainer.validate(val_loader, model)
