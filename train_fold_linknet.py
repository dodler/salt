import pandas as pd

from models.salt_models import AlbuNet, LinkNet34
from training import train_fold
from utils.common import myloss, iou_numpy

if __name__ == '__main__':
    folds = pd.read_csv('/root/data/salt/folds.csv')
    train_fold(folds, myloss, iou_numpy, LinkNet34, 'linknet34_3cn',
               batch_size=128, device=0, model_name='linknet34', epochs=100)
