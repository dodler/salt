import pickle

import cv2
from reader.image_reader import OpencvReader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from models.salt_models import UNet16
from training import Trainer
from utils.common import norm, bce_with_logits, SegmentationDataset, SegmentationPathProvider, cut_target
from utils.current_transform import UnetRGBTransform, gray_norm

DEVICE = 1
EPOCHS = 200

bce = nn.BCELoss()
mse = nn.MSELoss()
THRESH = 0.62


# norm = Normalize((0.5,), (0.5,))

def predict_unet(model, image):
    import numpy as np
    fliped = np.fliplr(image.copy()).copy()

    fliped = fliped[:, :, np.newaxis]

    image_tensor = gray_norm(torch.from_numpy(image[:, :, np.newaxis]).float().permute([2, 0, 1]))
    mask = model(image_tensor.unsqueeze(0).to(1))
    mask = mask.squeeze().detach().cpu().numpy()

    fliped_tensor = gray_norm(torch.from_numpy(fliped).float().permute([2, 0, 1]))
    fliped_mask = model(fliped_tensor.unsqueeze(0).to(1))
    fliped_mask = fliped_mask.squeeze().detach().cpu().numpy()
    fliped_mask = np.fliplr(fliped_mask).copy()

    result = (mask + fliped_mask) / 2.0
    # result = mask

    return cv2.resize(result, (101,101))


def myloss(x, y):
    return bce_with_logits(x.squeeze(), y.squeeze())


def mymetric(x, y):
    m = (x > THRESH).float()
    pred_t = m.view(-1).float()
    target = y.view(-1).float()
    inter = 2 * (pred_t * target).sum()
    union = (pred_t + target).sum()

    return (inter / union).cpu().item();


if __name__ == "__main__":
    model = UNet16(num_channels=1).float().to(DEVICE)
    dataset = SegmentationDataset(UnetRGBTransform(), SegmentationPathProvider('/root/data/train.csv'), x_reader=OpencvReader(),
                                  y_reader=OpencvReader())

    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes = [20, 64, 32]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(myloss, mymetric, optimizer, 'unet', None, DEVICE)

    train_loader = DataLoader(dataset, batch_size=batch_sizes[0])
    dataset.setmode('val')
    val_loader = DataLoader(dataset, batch_size=batch_sizes[0])
    dataset.setmode('train')

    for i in range(EPOCHS):
        trainer.train(train_loader, model, i)
        trainer.validate(val_loader, model)
