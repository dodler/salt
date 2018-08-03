import torch.nn.functional as F
from training.training import Trainer
from reader.image_reader import OpencvReader
from reader.image_reader import PillowReader
import torch.nn as nn
from dataset.dataset import GenericXYDataset
import os
import os.path as osp
import torch.nn.functional as F
from torchvision.transforms import * 
from generic_utils.segmentation.abstract import DualCompose
from generic_utils.segmentation.abstract import ImageOnly, Dualized, MaskOnly
from generic_utils.segmentation.dualcolor import *
from generic_utils.segmentation.dualcrop import DualRotatePadded, DualCrop
from generic_utils.segmentation.util_transform import DualResize, DualToTensor, DualImgsToTensor

from torch.utils.data import DataLoader

from generic_utils.metrics import dice_loss, iou

from deeplab import Res_Deeplab

DEVICE=3
BATCH_SIZE=4
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
    


class OpencvGrayscaleReader():
    def __call__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img > 1] = 1
        return img


bce = nn.BCELoss()
mse = nn.MSELoss()
THRESH = 0.5


DIM = 256

class OCVResize(object):
    def __init__(self, shape):
        self.shape = shape
        
    def __call__(self, img):
        return cv2.resize(img, self.shape)


class MyTransform(object):
    def __init__(self):
        self.train_transform = DualCompose([
            RandomGamma(0.5, 1.5),
            ImageOnly(OCVResize((DIM,DIM))),
            MaskOnly(OCVResize((outS(DIM), outS(DIM)))),
            DualToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))])

        self.val_transform = DualCompose([
            DualResize((DIM, DIM)),
            DualToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])
    
    def __call__(self, x,y,mode):
        if mode == 'train':
            return self.train_transform(x,y)
        else:
            return self.val_transform(x,y)

def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j+1)/2
    j = int(np.ceil((j+1)/2.0))
    j = (j+1)/2
    return int(j)

def deeplab_4loss(pred, ground_truth):
    return torch.log(torch.sqrt(bce(F.sigmoid(pred),ground_truth.unsqueeze(1)) + dice_loss(pred.view(-1),ground_truth.view(-1))))

def myloss(pred,ground_truth):
    return torch.log(torch.sqrt(bce(pred,ground_truth.unsqueeze(1)) + dice_loss(pred.view(-1),ground_truth.view(-1))))

def mymetric(x,y):
    m = (x > THRESH).float()
    return iou(m,y)

model = Res_Deeplab(NoLabels=1).float().to(DEVICE)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()) , lr=1e-3)#, momentum=0.975)

trainer = Trainer('salt', deeplab_4loss , mymetric, optimizer, 'deeplab', DEVICE)
dataset = SegmentationDataset(MyTransform(), SegmentationPathProvider(), x_reader=OpencvReader(), y_reader=OpencvGrayscaleReader())

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
dataset.setmode('val')
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
dataset.setmode('train')

#model = nn.DataParallel(Res_Deeplab(NoLabels=1)).float().cuda()

for i in range(EPOCHS):
    trainer.train(train_loader, model, i)
    trainer.validate(val_loader, model)

