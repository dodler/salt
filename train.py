import torch.nn as nn
from generic_utils.metrics import dice_loss, iou
from generic_utils.segmentation.util_transform import *
from models.segmentation.models import LinkNet34
from reader.image_reader import OpencvReader
from torch.utils.data import DataLoader
from training.training import Trainer

from common import SegmentationPathProvider, SegmentationDataset, OCVMaskReader
from current_transform import MyTransform

THRESH = 0.9

bce = nn.BCELoss()
mse = nn.MSELoss()


def myloss(x, y):
    return bce(x, y.unsqueeze(1))


def mymetric(x, y):
    m = (x > THRESH).float()
    return iou(m, y)


DEVICE = 1
EPOCHS = 400
model = LinkNet34().float().to(DEVICE)
# model.load_state_dict(torch.load('linknet_loss_0.5063234567642212.pth.tar'))
dataset = SegmentationDataset(MyTransform(), SegmentationPathProvider(), x_reader=OpencvReader(),
                              y_reader=OCVMaskReader())

lrs = [1e-2, 1e-3, 1e-4]
batch_sizes = [128,256, 512]

for i in range(1):
    if i < 2:
        optimizer = torch.optim.Adam(model.parameters(), lr=lrs[i])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lrs[i], momentum=0.9)
    trainer = Trainer(myloss, mymetric, optimizer, 'linknet', DEVICE)

    train_loader = DataLoader(dataset, batch_size=batch_sizes[i])
    dataset.setmode('val')
    val_loader = DataLoader(dataset, batch_size=batch_sizes[i])
    dataset.setmode('train')

    for i in range(EPOCHS):
        trainer.train(train_loader, model, i)
        trainer.validate(val_loader, model)
