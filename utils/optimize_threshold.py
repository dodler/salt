import torch
import numpy as np
import torch.utils.data as data

from utils.ush_dataset import TGSSaltDataset

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

batch_size = 64
device = 1
SMOOTH = 1e-6


def iou_numpy(outputs: np.array, labels: np.array, thresh=0.5):
    outputs = (outputs > thresh).astype(np.uint8)
    labels = (labels > thresh).astype(np.uint8)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded.mean()


def filter_image(img):
    if img.sum() < 100:
        return np.zeros(img.shape).astype(bool)
    else:
        return img.astype(bool)


def optimize_thresh(x_val, model, device, batch_size):
    val_predictions = []
    val_masks = []
    val_iou = []

    val_dataset = TGSSaltDataset("/root/data/salt/train/", x_val,
                                 is_test=False, is_val=True)
    print(x_val)

    with torch.no_grad():
        for image, mask in data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False):
            image = image.to(device)
            y_pred = model(image)
            y_pred = torch.sigmoid(y_pred).cpu().numpy()
            mask = mask[:, 0, y_min_pad:128 - y_max_pad,
                   x_min_pad:128 - x_max_pad].numpy()

            y_pred = y_pred[:, 0, y_min_pad:128 - y_max_pad,
                     x_min_pad:128 - x_max_pad]

            labels = y_pred > 0.5
            iou_score = iou_numpy(labels, mask.astype(bool))
            val_iou.append(iou_score)

            val_predictions.append(y_pred)
            val_masks.append(mask)

    val_predictions_stacked = np.vstack(val_predictions)
    val_masks_stacked = np.vstack(val_masks)

    thresholds = np.linspace(0.2, 0.95, 100)

    best_t = None
    best_score = 0

    for t in thresholds:
        val_predictions_ = val_predictions_stacked > t
        val_predictions_ = np.array([filter_image(img) for img in val_predictions_])
        score = iou_numpy(val_predictions_, val_masks_stacked.astype(bool))
        if score > best_score:
            best_score = score
            best_t = t
    print(best_t, best_score)

    return best_t, best_score
