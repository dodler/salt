import torch.nn as nn
import torch.nn.functional as F
from generic_utils.metrics import dice_loss, iou
from generic_utils.segmentation.abstract import DualCompose
from generic_utils.segmentation.abstract import ImageOnly, MaskOnly
from generic_utils.segmentation.dualcolor import *
from generic_utils.segmentation.util_transform import DualResize, DualToTensor
from reader.image_reader import OpencvReader
from torch.utils.data import DataLoader
from torchvision.transforms import *
from training.training import Trainer

from deeplab import Res_Deeplab

DEVICE=3
BATCH_SIZE=4
EPOCHS=200

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)


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

