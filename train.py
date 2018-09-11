import pickle

import cv2
import torch
import os
import os.path as osp
from reader.image_reader import OpencvReader
from torch.utils.data import DataLoader

from models.salt_models import LinkNet34, Linknet152
from training import train_fold, Trainer
from utils.common import norm, myloss, mymetric, SegmentationDataset, SegmentationPathProvider, cut_target, TestReader
from utils.current_transform import Linknet152Transform, gray_norm, MyTransform


def predict_linknet(model, image):
    import numpy as np
    with torch.no_grad():
        # p = 13
        # pad_img = cv2.copyMakeBorder(image, p + 1, p, p + 1, p, cv2.BORDER_REFLECT_101)
        # img = pad_img[:, :, 0:1] / 255.0
        t_img = cv2.resize(image, (128, 128))
        t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY) / 255.0
        t_img = t_img[:, :, np.newaxis]
        image_tensor = gray_norm(torch.from_numpy(t_img).float().permute([2, 0, 1]))
        image_tensor = gray_norm(image_tensor)
        mask1 = torch.sigmoid(model(image_tensor.unsqueeze(0).to(1)))
        mask1 = mask1.squeeze().cpu().numpy()


        return cv2.resize(mask1, (101, 101))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    DEVICE = 0
    EPOCHS = 200
    BATCH_SIZE = 256
    model = LinkNet34(num_channels=1).float().to(DEVICE)
    print(count_parameters(model))
    dataset = SegmentationDataset(MyTransform(), SegmentationPathProvider('/root/data/train.csv'),
                                  x_reader=TestReader(),
                                  y_reader=TestReader())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = Trainer(myloss, mymetric, optimizer, 'linknet34', None, DEVICE)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    dataset.setmode('val')
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    dataset.setmode('train')

    for i in range(EPOCHS):
        trainer.train(train_loader, model, i)
        trainer.validate(val_loader, model)
