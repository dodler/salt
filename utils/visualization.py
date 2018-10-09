import random

import numpy as np
import visdom
from torch.autograd import Variable

from generic_utils.visualization.visdom import FirstImage, ImageByIndex


class VisdomValueWatcher(object):
    def __init__(self, env_name='main',
                 in_x_watcher = ImageByIndex(), in_y_watcher=ImageByIndex(),
                 prediction_watcher=ImageByIndex()):
        self._watchers = {}
        self._wins = {}
        self._vis = visdom.Visdom(use_incoming_socket=False,env=env_name)
        #https://github.com/facebookresearch/visdom/issues/450
        self.vis_img = None
        self.vis_hist = None
        self.vis_conf_er = None

        self.in_x_watcher = in_x_watcher
        self.in_y_watcher = in_y_watcher
        self.prediction_watcher = prediction_watcher

        self.wins = {}
        self.every_iter_n = 10

    def display_and_add(self, img, title, key=None):
        if key is None:
            key = title
        if key in self.wins.keys():
            self._vis.image(img, opts=dict(title=title), win=self.wins[key])
        else:
            self.wins[key] = self._vis.image(img, opts=dict(title=title))

    def display_hist_and_add(self, vals, key, title=None):
        ''''
        vals - numpy array
        '''

        if key in self.wins.keys():
            self._vis.histogram(vals, win=self.wins[key], opts=dict(title=title))
        else:
            self.wins[key] = self._vis.histogram(vals, opts=dict(title=title))

    def display_labels_hist(self, labels):
        key = 'source_labels_histogram'
        numpy = labels.view(-1).cpu().data.numpy()
        if len(numpy) == 1:
            return
        self.display_hist_and_add(numpy, key, title='labels distribution')

    def display_most_confident_error(self, iter_num, img_batch, gt, batch_probs):
        if iter_num % 10 == 0:
            probs, cls = batch_probs.max(dim=1)
            gt = gt.data.cpu().numpy()
            probs = probs.data.cpu().numpy()
            cls = cls.data.cpu().numpy()

            bad_prob = -1
            bad_index = 0
            for i, prob in enumerate(probs):
                if cls[i] != gt[i] and prob > bad_prob:
                    bad_index = i
                    bad_prob = prob

            bad_cls_gt = gt[bad_index]
            bad_prob = probs[bad_index]
            bad_class = cls[bad_index]
            bad_img = img_batch.data.squeeze(0).cpu().numpy()[bad_index]

            if bad_prob > 0:
                self.display_and_add(bad_img,
                                     'error at gt:' + str(bad_cls_gt) + ', pred:' + str(bad_class) + ', prob:' + str(
                                         bad_prob),
                                     'confident_errors')

    def display_img_every(self, n, iter, image, key, title):
        '''
        display first image from variable or tensor in batch
        :param n:
        :param iter:
        :param image:
        :param key:
        :param title:
        :return:
        '''
        if iter % n == 0:
            if isinstance(image, Variable):
                img = image.data.squeeze(0).cpu().numpy()[0]
            else:
                img = image.squeeze(0).cpu().numpy()[0]
            self.display_and_add(img, title, key)

    def display_every_iter(self, iter_num, X, y, prediction):
        if iter_num%  self.every_iter_n == 0:
            index = random.randint(0, X.size()[0]-1)
            self.display_and_add(self.in_x_watcher(X, index), 'source x', 'x')
            self.display_and_add(self.in_y_watcher(y, index), 'source y', 'y')
            self.display_and_add(self.prediction_watcher(prediction, index),'prediction', 'pred')

    def get_vis(self):
        return self._vis

    def log_value(self, name, value, output=True):
        if name in self._watchers.keys():
            self._watchers[name].append(value)
        else:
            self._watchers[name] = [value]

        if output:
            self.output(name)

    def output_all(self):
        for name in self._wins.keys():
            self.output(name)

    def movingaverage(self, values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma

    def output(self, name):
        if name in self._wins.keys():

            y = self.movingaverage(self._watchers[name], 1)
            x = np.array(range(len(y)))

            self._vis.line(Y=y, X=x,
                           win=self._wins[name], update='new',
                           opts=dict(title=name))
        else:
            self._wins[name] = self._vis.line(Y=np.array(self._watchers[name]),
                                              X=np.array(range(len(self._watchers[name]))),
                                              opts=dict(title=name))

    def clean(self, name):
        self._watchers[name] = []