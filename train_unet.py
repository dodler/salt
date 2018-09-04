import cv2
from reader.image_reader import OpencvReader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from salt_models import UNet16
from common import SegmentationDataset, SegmentationPathProvider, bce_with_logits, norm
from current_transform import UnetRGBTransform
from training import Trainer

DEVICE = 1
EPOCHS = 400

bce = nn.BCELoss()
mse = nn.MSELoss()
THRESH = 0.62


# norm = Normalize((0.5,), (0.5,))

def predict_unet(model, image):
    p = 13
    pad_img = cv2.copyMakeBorder(image, p + 1, p, p + 1, p, cv2.BORDER_REFLECT_101) / 255.0
    image_tensor = norm(torch.from_numpy(pad_img).float().permute([2, 0, 1]))
    mask = model(image_tensor.unsqueeze(0).to(0))
    return mask.squeeze().detach().cpu().numpy()[14:115, 13:114]


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
    model = UNet16().float().to(DEVICE)
    dataset = SegmentationDataset(UnetRGBTransform(), SegmentationPathProvider(), x_reader=OpencvReader(),
                                  y_reader=OpencvReader())

    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes = [16, 64, 32]

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    trainer = Trainer(myloss, mymetric, optimizer, 'unet', DEVICE)

    train_loader = DataLoader(dataset, batch_size=batch_sizes[0])
    dataset.setmode('val')
    val_loader = DataLoader(dataset, batch_size=batch_sizes[0])
    dataset.setmode('train')

    for i in range(EPOCHS):
        trainer.train(train_loader, model, i)
        trainer.validate(val_loader, model)
