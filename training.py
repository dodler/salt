import pickle
import time

import numpy as np
import torch
from reader.image_reader import OpencvReader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from generic_utils.output_watchers import ClassificationWatcher
from generic_utils.utils import AverageMeter
from generic_utils.visualization.visualization import VisdomValueWatcher
from torch.utils.data import DataLoader

from utils.common import SegmentationDataset, SegmentationPathProvider
from utils.current_transform import MyTransform

VAL_LOSS = 'val loss'
VAL_ACC = 'val metric'
TRAIN_ACC_OUT = 'train metric'
TRAIN_LOSS_OUT = 'train loss'

LR = 1e-3
BATCH_SIZE=128
EPOCHS=150
DEVICE = 0


def train_fold(folds, loss,metric, ModelClass, base_checkpoint_name):
    for i, fold in enumerate(folds):
        print('doing fold',fold)
        model = ModelClass().float().to(DEVICE)
        dataset = SegmentationDataset(MyTransform(), SegmentationPathProvider(fold),
                                      x_reader=OpencvReader(),
                                      y_reader=OpencvReader())

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        trainer = Trainer(loss, metric, optimizer, base_checkpoint_name, DEVICE)

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        dataset.setmode('val')
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        dataset.setmode('train')

        for i in range(EPOCHS):
            trainer.train(train_loader, model, i)
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

    def optimize_threshold(self, val_loader, model, thresholds, metric):
        model.eval()

        predicted_masks = []
        expected_masks = []
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(val_loader):
                input_var = Variable(input.to(self.device))
                output = torch.sigmoid(model(input_var))
                predicted_masks.append(output.cpu())
                expected_masks.append(target.cpu())

        print(len(predicted_masks))
        best_thresh = thresholds[0]
        best_metric = 0
        for threshold in thresholds:
            avg_metric = 0.0
            for i in range(len(predicted_masks)):
                expected = expected_masks[i]
                predicted = predicted_masks[i]
                avg_metric += metric(expected, predicted, threshold)

            avg_metric /= float(len(predicted_masks))
            print(avg_metric)
            if avg_metric > best_metric:
                best_metric = avg_metric
                best_thresh = threshold

        return best_metric, best_thresh

    def validate(self, val_loader, model):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        model.eval()

        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            with torch.no_grad():
                input_var = Variable(input.to(self.device))
                target_var = Variable(target.to(self.device))

            output = model(input_var)

            loss = self.criterion(output, target_var)
            losses.update(loss.detach(), input.size(0))

            metric_val = self.metric(output, target_var)
            acc.update(metric_val)

            self.watcher.log_value(VAL_ACC, metric_val)
            self.watcher.log_value(VAL_LOSS, loss.detach())
            self.watcher.display_every_iter(batch_idx, input_var, target, torch.sigmoid(output))

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
            pickle.dump(model, open(self.get_checkpoint_name(losses.avg), 'wb'))

        self.epoch_num += 1
        return losses.avg, acc.avg

    def train(self, train_loader, model, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for batch_idx, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            input_var = torch.autograd.Variable(input.to(self.device))
            target_var = torch.autograd.Variable(target.to(self.device))

            self.optimizer.zero_grad()
            output = model(input_var)

            loss = self.criterion(output, target_var)

            loss.backward()
            self.optimizer.step()

            output = torch.sigmoid(output)
            losses.update(loss.item(), input.size(0))

            metric_val = self.metric(output, target_var)  # todo - add output dimention assertion
            acc.update(metric_val, batch_idx)

            self.watcher.log_value(TRAIN_ACC_OUT, metric_val)
            self.watcher.log_value(TRAIN_LOSS_OUT, loss.item())
            self.watcher.display_every_iter(batch_idx, input_var, target, output)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('\rEpoch: {0}  [{1}/{2}]\t'
                  'ETA: {time:.0f}/{eta:.0f} s\t'
                  'data loading: {data_time.val:.3f} s\t'
                  'loss {loss.avg:.4f}\t'
                  'metric {acc.avg:.4f}\t'.format(
                epoch, batch_idx, len(train_loader), eta=batch_time.avg * len(train_loader),
                time=batch_time.sum, data_time=data_time, loss=losses, acc=acc), end='')
        return losses.avg, acc.avg
