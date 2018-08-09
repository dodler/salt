import cv2
import numpy as np
from generic_utils.metrics import dice_loss, iou
from generic_utils.utils import ocv2torch
from models.segmentation.unet_parts import *
from reader.image_reader import OpencvReader
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from training.training import Trainer

from common import SegmentationDataset, SegmentationPathProvider, OCVMaskReader
from current_transform import MyTransform, UnetTransform

DEVICE = 2
EPOCHS = 400

bce = nn.BCELoss()
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


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if self.n_classes == 1:
            return F.sigmoid(x)
        else:
            return x


if __name__ == "__main__":
    model = UNet(1, 1).float().to(DEVICE)
    dataset = SegmentationDataset(UnetTransform(), SegmentationPathProvider(), x_reader=OpencvReader(),
                                  y_reader=OCVMaskReader())

    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes = [96, 64, 32]

    for i in range(1):
        if i < 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=lrs[i])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lrs[i], momentum=0.9)
        trainer = Trainer(myloss, mymetric, optimizer, 'unet', DEVICE)

        train_loader = DataLoader(dataset, batch_size=batch_sizes[i])
        dataset.setmode('val')
        val_loader = DataLoader(dataset, batch_size=batch_sizes[i])
        dataset.setmode('train')

        for i in range(EPOCHS):
            trainer.train(train_loader, model, i)
            trainer.validate(val_loader, model)
