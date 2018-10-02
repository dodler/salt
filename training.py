import pickle
import time

import numpy as np
import torch
from generic_utils.output_watchers import ClassificationWatcher
from generic_utils.utils import AverageMeter
from reader.image_reader import OpencvReader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import *

from utils.common import SegmentationDataset, PandasProvider
from utils.current_transform import MyTransform
from utils.visualization import VisdomValueWatcher
from utils.common import logger

VAL_LOSS = 'val loss'
VAL_ACC = 'val metric'
TRAIN_ACC_OUT = 'train metric'
TRAIN_LOSS_OUT = 'train loss'

LR = 1e-3
BATCH_SIZE = 1
EPOCHS = 200
DEVICE = 0


def predict_multiple(model_dump_paths, images, predict_func, average_func):
    predictions = []
    for mdl_dump_path in model_dump_paths:
        mdl_dump_path = pickle.load(open(mdl_dump_path, 'rb')).to(DEVICE)
        masks = [predict_func(mdl_dump_path, img) for img in tqdm(images)]
        predictions.append(masks)

    predictions = np.array(predictions)
    return average_func(predictions, axis=0)


def train_fold(folds, loss, metric, ModelClass, base_checkpoint_name, model_name,
               epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE):
    for fold in folds['fold'].unique():
        print('doing fold', fold)
        model = ModelClass().float().to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        trainer = Trainer(loss, metric, optimizer, model_name + '_fold_' + str(fold),
                          base_checkpoint_name + '_fold_' + str(fold), device)

        train_loader = DataLoader(SegmentationDataset(MyTransform(),
                                                      PandasProvider(folds[folds['fold'] != fold]),
                                                      x_reader=OpencvReader(),
                                                      y_reader=OpencvReader(), split=False), batch_size=batch_size,
                                  shuffle=True, num_workers=1)

        val_loader = DataLoader(SegmentationDataset(MyTransform(),
                                                    PandasProvider(folds[folds['fold'] == fold]),
                                                    x_reader=OpencvReader(),
                                                    y_reader=OpencvReader(), split=False), batch_size=batch_size,
                                num_workers=1)
        for epch in range(epochs):
            trainer.train(train_loader, model, epch)
            trainer.validate(val_loader, model)


class Trainer(object):

    def __init__(self, criterion, metric, optimizer, model_name, base_checkpoint_name=None, device=0):
        '''

        :param watcher_env: environment for visdom
        :param criterion - loss function
        '''
        if base_checkpoint_name is None:
            self.base_checkpoint_name = model_name
        else:
            self.base_checkpoint_name = base_checkpoint_name

        self.metric = metric
        self.criterion = criterion
        self.watcher = VisdomValueWatcher(model_name)
        self.output_watcher = ClassificationWatcher(self.watcher)
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(optimizer, 'min')
        self.best_loss = np.inf
        self.model_name = model_name
        self.device = device
        self.epoch_num = 0

    def set_output_watcher(self, output_watcher):
        self.output_watcher = output_watcher

    def get_watcher(self):
        return self.watcher

    def save_checkpoint(self, state, name):
        print('saving state at', name)
        torch.save(state, name)

    def get_checkpoint_name(self, loss):
        return self.base_checkpoint_name + '_loss_' + str(loss.detach().cpu().numpy()) + '.pth.tar'

    def is_best(self, avg_loss):
        best = avg_loss < self.best_loss
        if best:
            self.best_loss = avg_loss

        return best

    def validate(self, val_loader, model):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        model.eval()

        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            with torch.no_grad():
                input_var = input.to(self.device)
                target_var = target.to(self.device)

                output = model(input_var)

                loss = self.criterion(output, target_var)
                losses.update(loss.detach(), input.size(0))
                metric_val = self.metric(output, target_var)
                acc.update(metric_val)

                self.watcher.log_value(VAL_ACC, metric_val)
                self.watcher.log_value(VAL_LOSS, loss.detach())
                self.watcher.display_every_iter(batch_idx, input_var, target, output)

            batch_time.update(time.time() - end)
            end = time.time()

            print('\rValidation: [{0}/{1}]\t'
                  'ETA: {time:.0f}/{eta:.0f} s\t'
                  'loss {loss.avg:.4f}\t'
                  'metric {acc.avg:.4f}\t'.format(
                batch_idx, len(val_loader), eta=batch_time.avg * len(val_loader),
                time=batch_time.sum, loss=losses, acc=acc), end='')
        print()
        self.scheduler.step(losses.avg)

        if self.is_best(losses.avg) and self.epoch_num % 3 == 0:
            self.save_checkpoint(model.state_dict(), self.get_checkpoint_name(losses.avg))
            # pickle.dump(model, open(self.get_checkpoint_name(losses.avg), 'wb'))

        self.epoch_num += 1
        return losses.avg, acc.avg

    def train(self, train_loader, model, epoch):
        print(self.device)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for batch_idx, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            input_var = input.to(self.device)
            target_var = target.to(self.device)

            output = model(input_var)

            loss = self.criterion(output, target_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                losses.update(loss.item(), input.size(0))
                metric_val = self.metric(output, target_var)  # todo - add output dimention assertion
                acc.update(metric_val, batch_idx)

                self.watcher.log_value(TRAIN_ACC_OUT, metric_val)
                self.watcher.log_value(TRAIN_LOSS_OUT, loss.item())
                self.watcher.display_every_iter(batch_idx, input_var, target, output)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            logger.debug('\rEpoch: {0}  [{1}/{2}]\t'
                  'ETA: {time:.0f}/{eta:.0f} s\t'
                  'data loading: {data_time.val:.3f} s\t'
                  'loss {loss.avg:.4f}\t'
                  'metric {acc.avg:.4f}\t'.format(
                epoch, batch_idx, len(train_loader), eta=batch_time.avg * len(train_loader),
                time=batch_time.sum, data_time=data_time, loss=losses, acc=acc))
        return losses.avg, acc.avg
