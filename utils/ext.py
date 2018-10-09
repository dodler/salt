import os

import cv2
import torch
import torch.utils.data as data


def reflect_center_pad(img, mask=False):
    height, width, _ = img.shape

    # Padding in needed for UNet models because they need image size to be divisible by 32
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

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
    if mask:
        # Convert mask to 0 and 1 format
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(img).float().permute([2, 0, 1])
    else:
        img = img / 255.0
        return torch.from_numpy(img).float().permute([2, 0, 1])


def load_image(path):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test=False, is_val=False, augment_func=None):
        self.is_test = is_test
        self.is_val = is_val
        self.root_path = root_path
        self.file_list = file_list
        self.aug = augment_func

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            raise IndexError

        file_id = self.file_list[index]

        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")

        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")

        image = load_image(image_path)

        if self.is_test:
            image = reflect_center_pad(image, mask=False)
            return (image,)
        else:
            mask = load_image(mask_path)

            if self.is_val:
                image = reflect_center_pad(image, mask=False)
                mask = reflect_center_pad(mask, mask=True)
                return image, mask
            else:
                if self.aug is not None:
                    augmented_res = self.aug(image=image, mask=mask)
                    image = augmented_res['image']
                    mask = augmented_res['mask']

                image = reflect_center_pad(image, mask=False)
                mask = reflect_center_pad(mask, mask=True)
                return image, mask
