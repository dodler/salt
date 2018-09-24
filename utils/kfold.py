import os
import os.path as osp
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

DATA_ROOT = '/root/data/salt/'


def save_folds(folds):
    for i in set(folds['fold']):
        fold_at_i = folds[folds['fold'] == i]['id']
        fold_at_i.to_csv(osp.join('/root/data/', 'fold_' + str(i) + '.csv'), index=False, header='id')


def main():
    n_fold = 5
    depths = pd.read_csv('/root/data/salt/depths.csv')
    train = pd.read_csv('/root/data/salt/train.csv')

    # train['id'] = train['id'].apply(int)
    # depths['id'] = depths['id'].apply(int)
    depths = train.merge(depths,on='id',how='left')
    depths.sort_values('z', inplace=True)
    depths.drop('z', axis=1, inplace=True)
    depths['fold'] = (list(range(n_fold)) * depths.shape[0])[:depths.shape[0]]
    depths.to_csv(os.path.join('/root/data/salt/', 'folds.csv'), index=False)
    save_folds(depths)
    print(pd.read_csv('/root/data/fold_0.csv').head())


main()
