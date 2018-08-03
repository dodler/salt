from training.training import Trainer
from reader.image_reader import OpencvReader, OpencvGrayReader
from reader.image_reader import PillowReader
import torch.nn as nn
from dataset.dataset import GenericXYDataset
import os
import os.path as osp
import torch.nn.functional as F
from torchvision.transforms import * 
from generic_utils.segmentation.abstract import DualCompose
from generic_utils.segmentation.abstract import ImageOnly, Dualized
from generic_utils.segmentation.dualcolor import *
from generic_utils.segmentation.dualcrop import DualRotatePadded, DualCrop, DualPad
from generic_utils.segmentation.util_transform import DualResize, DualToTensor, DualSingleChannelToTensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.segmentation.models import UNet16
from generic_utils.metrics import dice_loss, iou

DEVICE=2
PADDING=13
MAX_ROTATE_ANGLE=45
BATCH_SIZE=96
EPOCHS=200

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

class SegmentationDataset(GenericXYDataset):
    def read_x(self, index):
        if self.mode == 'train':
            return self.x_reader(osp.join('/home/dhc/salt/images/',self.train_x[index]))
        else:
            return self.x_reader(osp.join('/home/dhc/salt/images/',self.val_x[index]))
        
    def read_y(self, index):
        if self.mode == 'train':
            return self.y_reader(osp.join('/home/dhc/salt/masks/',self.train_y[index]))
        else:
            return self.y_reader(osp.join('/home/dhc/salt/masks/',self.val_y[index]))


class SegmentationPathProvider(object):
    def __init__(self):
        self.files = os.listdir('/home/dhc/salt/masks/')
        self.index = 0
    
    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index > len(self.files)-1:
            raise StopIteration
        obj = self.files[self.index]
        self.index += 1
        return obj, obj
        

class MyTransform(object):
    def __init__(self):

        self.train_transform = DualCompose([
            ImageOnly(OCVGrayscale()),
            ImageOnly(RandomSingleChannelGamma(0.5, 1.5)),
            DualResize((102,102)),
            DualRotatePadded(MAX_ROTATE_ANGLE, PADDING), # to make final shape as 128 128
            DualSingleChannelToTensor(),
            ImageOnly(Normalize((0.5,), (0.5,)))])

        self.val_transform = DualCompose([
            ImageOnly(OCVGrayscale()),
            DualResize((102,102)),
            DualPad(PADDING),
            DualSingleChannelToTensor(),
            ImageOnly(Normalize((0.5,), (0.5,))),
        ])
    
    def __call__(self, x,y,mode):
        if mode == 'train':
            return self.train_transform(x,y)
        else:
            return self.val_transform(x,y)



class OCVMaskReader():
    def __call__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img > 1] = 1
        return img

bce = nn.BCELoss()
mse = nn.MSELoss()
THRESH = 0.5

def myloss(x,y):
    return torch.pow(bce(x,y.unsqueeze(1)) + dice_loss(x.view(-1),y.view(-1)),2)

def mymetric(x,y):
    m = (x > THRESH).float()
    return iou(m,y)


model = UNet16().float().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.975)
trainer = Trainer(myloss , mymetric, optimizer, 'unet_salt', DEVICE)
dataset = SegmentationDataset(MyTransform(), SegmentationPathProvider(), x_reader=OpencvReader(), y_reader=OCVMaskReader())

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
dataset.setmode('val')
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
dataset.setmode('train')

for i in range(EPOCHS):
    trainer.train(train_loader, model, i)
    trainer.validate(val_loader, model)
