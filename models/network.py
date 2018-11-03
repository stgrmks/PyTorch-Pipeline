_author__ = 'MSteger'

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torchvision import models
from components.summary import summary
from itertools import chain

class pretrainedNetwork(nn.Module):
    # TODO: handle replace_clf differently. if yes 0> replace last layer. if nn.Sequential 0> replace

    def __init__(self, pretrained_models = models.alexnet(pretrained=True) , input_shape = (3, 224, 224), num_class = 200, freeze_layers = range(5), replace_clf=True):
        super(pretrainedNetwork, self).__init__()
        self.features, self.classifier = pretrained_models.features, pretrained_models.classifier
        self.flat_fts = self.get_flat_fts(input_shape, self.features)
        if replace_clf: self.classifier = nn.Sequential(nn.Linear(self.flat_fts, 100),nn.Dropout(p=0.2),nn.ReLU(),nn.Linear(100, num_class))
        if freeze_layers is not None:
            for idx, layer in enumerate(chain(self.features.children(), self.classifier.children())):
                if idx in freeze_layers:
                    for p in layer.parameters(): p.requires_grad = False
        self.input_shape = input_shape

    def get_flat_fts(self, in_size, fts):
        f = fts(autograd.Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts)
        return F.softmax(self.classifier(flat_fts), dim = 1)

class networkTraining(object):

    def __init__(self, model, optimizer, loss, batch_size, device, LE = None, checkpoint_path = None, verbose = True):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.current_epoch = 1
        self.batch_size = batch_size
        self.device = device
        self.LE = LE
        self.verbose = verbose
        self.epoch = None
        self.summary = self.set_summary()
        if checkpoint_path is not None: self._load_model_from_chkp(checkpoint_path)

    def set_summary(self):
        return summary(model = self.model, device = self.device, input_size =(1,) + self.model.input_shape, verbose = self.verbose).summary

    def _load_model_from_chkp(self, checkpoint_path):
        chkp_dict = torch.load(checkpoint_path)
        self.epoch = chkp_dict['epoch']
        self.model.load_state_dict(chkp_dict['state_dict'])
        self.optimizer.load_state_dict(chkp_dict['optimizer'])
        if self.verbose: print 'Loading Model States & Optimizer from: {}'.format(checkpoint_path)
        return self

    def train_epoch(self, epoch, train_data, callbacks):
        self.model.train()
        self.logger['epoch'] = epoch
        callbacks = self._callbacks(callbacks=callbacks, state='on_epoch_begin', set_model=True, set_logger=True, retrieve_logger = True)
        y_train, yHat_train, batches = [], [], len(train_data)
        for batch_idx, (X, y) in enumerate(train_data):
            self.logger['batch'] = batch_idx + 1
            callbacks = self._callbacks(callbacks=callbacks, state='on_batch_begin', set_model=True, set_logger=True, retrieve_logger = True)
            X = autograd.Variable(X).to(self.device)
            if self.LE is not None: y = self.LE.transform(y)
            y = autograd.Variable(torch.from_numpy(np.array(y)), requires_grad=False).to(self.device)
            self.optimizer.zero_grad()
            yHat = self.model(X)
            yHat_train.append(yHat)
            y_train.append(y)
            batch_loss = self.loss(yHat, y)
            batch_loss.backward()
            self.optimizer.step()
            callbacks = self._callbacks(callbacks=callbacks, state='on_batch_end', set_model=True, y_train = y, yHat_train = yHat, retrieve_logger = True)
        return epoch, callbacks, y_train, yHat_train

    def validate(self, val_data):
        self.model.eval()
        y_val, yHat_val = [], []
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_data):
                X = X.to(self.device)
                if self.LE is not None: y = self.LE.transform(y)
                y = torch.from_numpy(np.array(y)).to(self.device)
                yHat_val.append(self.model(X))
                y_val.append(y)
        return y_val, yHat_val

    def fit(self, epochs, train_data, val_data = None, callbacks = None):
        self.logger = {}
        self.logger['epochs'], self.logger['batches'], self.logger['device'], self.logger['continue_training'] = epochs, len(train_data), self.device, True
        callbacks = self._callbacks(callbacks = callbacks, state = 'on_train_begin', set_model = True, set_logger = True, retrieve_logger = True)
        epoch = 0

        if self.epoch is not None:
            epoch = self.epoch
            epochs += self.epoch

        while epoch < epochs:
            epoch, callbacks, y_train, yHat_train = self.train_epoch(epoch = epoch+1, train_data = train_data, callbacks = callbacks)
            y_val, yHat_val = self.validate(val_data = val_data)
            callbacks = self._callbacks(callbacks=callbacks, state='on_epoch_end', set_model=True, y_val = y_val, yHat_val = yHat_val, y_train = y_train, yHat_train = yHat_train, set_logger=True, retrieve_logger = True)
            if not self.logger['continue_training']: break

        self.callbacks = self._callbacks(callbacks = callbacks, state = 'on_train_end', set_model = True, set_logger = True, retrieve_logger = True)
        return self

    def evaluate(self, test_data):
        # TODO: well...
        return self

    def _callbacks(self, callbacks, state, set_model = False, set_logger = False, retrieve_logger = False, **params):
        if callbacks is not None:
            for cb in callbacks:
                if set_model:
                    cb.set_model(model = self.model, optimizer = self.optimizer)
                if set_logger: cb.set_logger(logger = self.logger)
                getattr(cb, state)(**params)
                if retrieve_logger: self.logger = cb.get_logger()
        return callbacks

if __name__ == '__main__':
    print 'done'