import pandas as pd

from models.segnet import SegNet
from training import train_fold
from utils.common import myloss, iou_numpy

if __name__ == '__main__':
    folds = pd.read_csv('/root/data/salt/folds.csv')
    train_fold(folds, myloss, iou_numpy, SegNet,'segnet_3cn', batch_size=8, device=0, model_name='segnet_3cn')