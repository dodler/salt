import cv2
import numpy as np
from generic_utils.metrics import iou
from models.segmentation.unet_parts import *
from reader.image_reader import OpencvReader
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from training.training import Trainer

from common import SegmentationDataset, SegmentationPathProvider, OCVMaskReader
from current_transform import MyTransform, AlbunetTransform
from salt_models import AlbuNet

DEVICE = 3
EPOCHS = 400

bce = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()
THRESH = 0.62

norm = Normalize((0.5,), (0.5,))


def predict_unet(model, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[0:96, 0:96]
    image = image[np.newaxis, ...]
    image_tensor = norm(torch.FloatTensor(image / 255.0))
    mask = model(image_tensor.unsqueeze(0).to(0))
    mask = mask.squeeze().detach().cpu().numpy()
    return cv2.resize(mask, (101,101))


def myloss(x, y):
    return bce(x, y.unsqueeze(1))


def mymetric(x, y):
    m = (x > THRESH).float()
    return iou(m, y)


if __name__ == "__main__":
    model = AlbuNet().float().to(DEVICE)
    dataset = SegmentationDataset(AlbunetTransform(), SegmentationPathProvider(), x_reader=OpencvReader(),
                                  y_reader=OCVMaskReader())

    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes = [64, 64, 32]

    for i in range(1):
        if i < 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=lrs[i])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lrs[i], momentum=0.9)
        trainer = Trainer(myloss, mymetric, optimizer, 'albunet', DEVICE)

        train_loader = DataLoader(dataset, batch_size=batch_sizes[i])
        dataset.setmode('val')
        val_loader = DataLoader(dataset, batch_size=batch_sizes[i])
        dataset.setmode('train')

        for i in range(EPOCHS):
            trainer.train(train_loader, model, i)
            trainer.validate(val_loader, model)
