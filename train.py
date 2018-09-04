import torch
import os
import os.path as osp

from models.salt_models import LinkNet34
from training import train_fold
from utils.common import norm, myloss, mymetric


def predict_linknet(model, image):
    with torch.no_grad():
        p = 13
        # pad_img = cv2.copyMakeBorder(image, p + 1, p, p + 1, p, cv2.BORDER_REFLECT_101)/255.0
        image_tensor = torch.from_numpy(image).float().permute([2, 0, 1])
        image_tensor = norm(image_tensor)
        mask = torch.sigmoid(model(image_tensor.unsqueeze(0).to(0)))
        mask = mask.squeeze().cpu().numpy()  # [14:115, 13:114]
        return mask


if __name__ == '__main__':
    folds = [k for k in os.listdir('/root/data/') if k.endswith('.csv') and 'fold' in k]
    folds = [osp.join('/root/data/', k) for k in folds]
    print(folds)
    train_fold(folds, myloss, mymetric, LinkNet34, 'linknet')
