import torch
from reader.image_reader import OpencvRGBReader

from models.salt_models import AlbuNet, LinkNet34
from training import Trainer
from utils.common import myloss, SegmentationDataset, SegmentationPathProvider, iou_numpy, \
    count_parameters, get_loader
from utils.current_transform import MyTransform

if __name__ == '__main__':
    DEVICE = 0
    EPOCHS = 200
    BATCH_SIZE = 24
    model = LinkNet34().type(torch.float).to(DEVICE)

    print(count_parameters(model))

    dataset = SegmentationDataset(MyTransform(), SegmentationPathProvider('/root/data/salt/train.csv'),
                                  x_reader=OpencvRGBReader(),
                                  y_reader=OpencvRGBReader())

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    trainer = Trainer(myloss, iou_numpy, optimizer, 'albunet_3cn', None, DEVICE)

    train_loader = get_loader(dataset, 'train', BATCH_SIZE)
    val_loader = get_loader(dataset, 'val', BATCH_SIZE)

    for i in range(EPOCHS):
        trainer.train(train_loader, model, i)
        trainer.validate(val_loader, model)
