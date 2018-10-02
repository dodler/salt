import pandas as pd

from models.salt_models import AlbuNet
from models.segnet import SegNet
from training import train_fold
from utils.common import myloss, iou_numpy

if __name__ == '__main__':
    folds = pd.read_csv('/root/data/salt/folds.csv')
    train_fold(folds, myloss, iou_numpy, Linknet152,'linknet152_3cn',
               batch_size=40, device=1, model_name='linknet152_3cn',epochs=200)