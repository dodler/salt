from generic_utils.metrics import iou
from generic_utils.segmentation.util_transform import *
from generic_utils.utils import AverageMeter
from models.segmentation.models import LinkNet34
from reader.image_reader import OpencvReader
from torch.autograd import Variable
from torch.utils.data import DataLoader

from common import SegmentationPathProvider, SegmentationDataset, OCVMaskReader
from current_transform import MyTransform, UnetTransform
from train_unet import UNet

DEVICE = 1
BATCH_SIZE = 128


def mymetric(x, y, threshold):
    m = (x > threshold).float()
    return iou(m.view(-1), y.view(-1))


DEVICE = 0

checkpoint_name = '/tmp/salt/unet_loss_0.2864511013031006.pth.tar'
model = UNet(1, 1).float().to(DEVICE)
model.load_state_dict(torch.load(checkpoint_name))
dataset = SegmentationDataset(UnetTransform(), SegmentationPathProvider(), x_reader=OpencvReader(),
                              y_reader=OCVMaskReader())

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
dataset.setmode('val')
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
dataset.setmode('train')

dataset.setmode('val')


def get_raw_predicts(model):
    # switch to evaluate mode
    model.eval()

    targets = []
    outputs = []

    for batch_idx, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input_var = Variable(input.to(DEVICE))
            target_var = Variable(target.to(DEVICE))
            output = model(input_var)
            targets.append(target_var)
            outputs.append(output)

    return targets, outputs


def get_metric(targets, outputs, threshold):
    acc = AverageMeter()

    for i, batch_pred in enumerate(outputs):
        metric_val = mymetric(batch_pred, targets[i], float(threshold))
        acc.update(metric_val)

    return acc.avg


from tqdm import *

thresholds = np.arange(0.95, 1, 0.0001, dtype=np.float32)
ious = []

print('doing predict')
targets, outputs = get_raw_predicts(model)
print('predict done')

for threshold in tqdm(thresholds):
    ious.append(get_metric(targets, outputs, threshold))

print(ious)
print(thresholds[np.argmax(ious)])
