import glob
import os
import torch.utils.data as data
import numpy as np
import torch
from tqdm import *
import pandas as pd
from models.salt_models import Linknet152, LinkNet34, UNet16, WiderResnetNet, AlbuNet
from models.vanilla_unet import UNet
from utils.common import get_directory
from utils.optimize_threshold import optimize_thresh, filter_image
from utils.ush_dataset import TGSSaltDataset

THRESH = 0.7

directory = get_directory()
n_fold = 8
depths = pd.read_csv(os.path.join(directory, 'depths.csv'))
depths.sort_values('z', inplace=True)
depths['fold'] = (list(range(n_fold)) * depths.shape[0])[:depths.shape[0]]
depths.head()

train_path = os.path.join(directory, 'train', 'images')
test_path = os.path.join(directory, 'test', 'images')
y_train = os.path.join(directory, 'train', 'masks')

file_list = list(depths['id'].values)

train_images = os.listdir(train_path)
train = pd.DataFrame(train_images, columns=['id'])
train.id = train.id.apply(lambda x: x[:-4]).astype(str)
train = pd.merge(train, depths, on='id', how='left')

test_path = os.path.join(directory, 'test')
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
test_file_list[:3], test_path

print("len(test_file_len): {}".format(len(test_file_list)))
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test=True)
torch.manual_seed(0)

DEVICE = 0
model = UNet().to(DEVICE)
model.eval()
model.load_state_dict(torch.load('/tmp/pycharm_project_959/unet_128_best.pth.tar'))

height, width = 101, 101

if height % 32 == 0:
    y_min_pad = 0
    y_max_pad = 0
else:
    y_pad = 32 - height % 32
    y_min_pad = int(y_pad / 2)
    y_max_pad = y_pad - y_min_pad

if width % 32 == 0:
    x_min_pad = 0
    x_max_pad = 0
else:
    x_pad = 32 - width % 32
    x_min_pad = int(x_pad / 2)
    x_max_pad = x_pad - x_min_pad

all_predictions = []
for image in tqdm(data.DataLoader(test_dataset, batch_size=1, shuffle=False)):
    image = image[0].type(torch.float).to(DEVICE)
    y_pred = model(image)
    y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()
    y_pred = y_pred[:, 0, y_min_pad:128 - y_max_pad,
             x_min_pad:128 - x_max_pad]

    all_predictions.append(y_pred)

all_predictions = np.vstack(all_predictions)

best_t, best_score = optimize_thresh(train.id.values, model, DEVICE, 4)

threshold = best_t
binary_prediction = (all_predictions > threshold).astype(bool)


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


all_masks = []
for p_mask in tqdm(list(binary_prediction)):
    p_mask = filter_image(p_mask)
    p_mask = rle_encoding(p_mask)
    all_masks.append(' '.join(map(str, p_mask)))

submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('my_subm.csv', index=False)