import cv2
import torch
from reader.image_reader import OpencvReader
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from models.salt_models import AlbuNet
from training import Trainer
from utils.common import SegmentationDataset, SegmentationPathProvider, myloss, mymetric
from utils.current_transform import AlbunetTransform

DEVICE = 0
EPOCHS = 400

THRESH = 0.5

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

norm = Normalize(rgb_mean, rgb_std)


def predict_albunet(model, image):
    with torch.no_grad():
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image_tensor = torch.from_numpy(image).float().permute([2, 0, 1])
        image_tensor = norm(image_tensor)
        mask = torch.sigmoid(model(image_tensor.unsqueeze(0).to(0)))
        mask = mask.squeeze().detach().cpu().numpy()
        return cv2.resize(mask, (101, 101))


if __name__ == "__main__":
    model = AlbuNet().float().to(DEVICE)
    dataset = SegmentationDataset(AlbunetTransform(), SegmentationPathProvider('/root/data/train.csv'),
                                  x_reader=OpencvReader(),
                                  y_reader=OpencvReader())

    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes = [32, 64, 32]

    for i in range(1):
        optimizer = torch.optim.SGD(model.parameters(), lr=lrs[i], momentum=0.9)
        trainer = Trainer(myloss, mymetric, optimizer, 'albunet', DEVICE)

        train_loader = DataLoader(dataset, batch_size=batch_sizes[i])
        dataset.setmode('val')
        val_loader = DataLoader(dataset, batch_size=batch_sizes[i])
        dataset.setmode('train')

        for i in range(EPOCHS):
            trainer.train(train_loader, model, i)
            trainer.validate(val_loader, model)
