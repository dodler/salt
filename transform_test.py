from generic_utils.segmentation.abstract import DualCompose
from generic_utils.segmentation.abstract import ImageOnly
from generic_utils.segmentation.dualcolor import *
from generic_utils.segmentation.dualcrop import DualRotatePadded, DualPad, DualCrop
from generic_utils.segmentation.util_transform import DualResize, DualToTensor, DualSingleChannelToTensor
from torchvision.transforms import *
import numpy as np
from common import *

DEVICE = 2
PADDING = 13
MAX_ROTATE_ANGLE = 45
BATCH_SIZE = 2
EPOCHS = 200

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

test_img = '/home/dhc/salt/images/287aea3a3b.png'
test_mask = '/home/dhc/salt/masks/287aea3a3b.png'

train_transform = DualCompose([
    ImageOnly(Gray()),
    ImageOnly(UnsqueezeRight()),
    DualCrop(96),
    RandomFlip(),
    ImageOnly(UnsqueezeLeft()),
    ImageOnly(To01()),
    DualSingleChannelToTensor(),
    ImageOnly(Normalize((0.5,), (0.5,)))
])

val_transform = DualCompose([
    ImageOnly(Gray()),
    ImageOnly(UnsqueezeRight()),
    DualCrop(96),
    ImageOnly(Squeeze()),
    ImageOnly(UnsqueezeLeft()),
    ImageOnly(To01()),
    DualSingleChannelToTensor(),
    ImageOnly(Normalize((0.5,), (0.5,)))
])

mask_reader = OCVMaskReader()

size = 96
img = cv2.imread(test_img)
print(img.shape[0])
print(img.shape[0] - size)
t = img.shape[0] - size
print('t', t)
print('rand',random.randint(0, t))
x = random.randint(0, img.shape[0] - size)
y = random.randint(0, img.shape[0] - size)
print(x,y)

tensor = train_transform(cv2.imread(test_img), mask_reader(test_mask))
print(tensor[0].size(), tensor[0].max(), tensor[0].min(), tensor[1].size())


tensor = val_transform(cv2.imread(test_img), mask_reader(test_mask))
print(tensor[0].size(), tensor[0].max(), tensor[0].min(), tensor[1].size())

train_transform = DualCompose([
            # RandomGamma(0.9, 1.1),
            # Randomize(ImageOnly(EmbossAndOverlay())),
            # Randomize(ImageOnly(Sharpen())),
            DualCrop(96),
            RandomFlip(),
            # ImageOnly(RandomBlur()),
            DualToTensor(),
            ImageOnly(Normalize(rgb_mean, rgb_std))
        ])


tensor = train_transform(cv2.imread(test_img), mask_reader(test_mask))
print(tensor[0].size(), tensor[0].max(), tensor[0].min(), tensor[1].size())

image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = image[0:96, 0:96]
image = image[np.newaxis, ...]
print(image.shape)
image_tensor = torch.FloatTensor(image / 255.0)
print(image_tensor.size())