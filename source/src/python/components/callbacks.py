_author__ = 'MSteger'

import os
import numpy as np
import torch
import datetime
from tqdm import tqdm
from helpers import geo_mean
from tensorboardX import SummaryWriter

class Callback(object):
    """
    based on https://github.com/keras-team/keras/blob/master/keras/callbacks.py
    """
    def __init__(self, **_):
        self.model = None
        self.logger = {}

    def set_model(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def set_logger(self, logger):
        self.logger = logger

    def get_logger(self):
        return self.logger

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, y_val, yHat_val, y_train, yHat_train):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self, y_train, yHat_train):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

class MetricTracker(Callback):

    def __init__(self, metrics, save_folder_path = None, verbose = True):
        super(Callback, self).__init__()
        self.metrics = metrics
        self.save_folder_path = save_folder_path
        self.verbose = verbose

    def _compute_metrics(self, y_train = None, yHat_train = None, y_val = None, yHat_val = None):
        performance = {}
        for m_name, m_fn in self.metrics:
            performance[m_name] = {}
            if y_train is not None:
                performance[m_name]['train'] = m_fn(torch.cat(y_train), torch.cat(yHat_train)) if type(y_train) == type([]) else m_fn(torch.cat([y_train]), torch.cat([yHat_train]))

            if y_val is not None:
                performance[m_name]['val'] = m_fn(torch.cat(y_val), torch.cat(yHat_val)) if type(y_val) == type([]) else m_fn(torch.cat([y_val]), torch.cat([yHat_val]))

        return performance

    def _write_file(self, file_path, line, mode = 'w'):
        with open(file_path, mode) as f:
            f.write(line)
            f.close()
        return self

    def on_train_begin(self):
        if 'batch_metrics' not in self.logger.keys(): self.logger['batch_metrics'] = {}
        if 'epoch_metrics' not in self.logger.keys(): self.logger['epoch_metrics'] = {}
        if (not os.path.exists(self.save_folder_path)) & (self.save_folder_path is not None): os.makedirs(self.save_folder_path)
        self.file_path = os.path.join(self.save_folder_path, 'MetricTracker.csv')
        if (not os.path.exists(self.file_path)) & (self.save_folder_path is not None):
            headers = sorted(['{}__train'.format(x[0]) for x in self.metrics] + ['{}__val'.format(x[0]) for x in self.metrics])
            self._write_file(file_path=self.file_path, line=','.join(['','epoch'] + headers + ['\n']), mode='w')
        return self

    def on_batch_end(self, y_train, yHat_train):
        performance = {self.logger['batch']: self._compute_metrics(y_train = y_train, yHat_train = yHat_train)}
        self.logger['batch_metrics'].update(performance)
        return self

    def on_epoch_end(self, y_val, yHat_val, y_train, yHat_train):
        performance = {self.logger['epoch']: self._compute_metrics(y_train = y_train, yHat_train = yHat_train, y_val = y_val, yHat_val = yHat_val)}
        self.logger['epoch_metrics'].update(performance)
        if self.verbose: print '\nPerformance Epoch {}: {}'.format(self.logger['epoch'], performance)
        if self.save_folder_path is not None:
            headers, line = sorted(['{}__train'.format(x[0]) for x in self.metrics] + ['{}__val'.format(x[0]) for x in self.metrics]), '{}, {}'.format(datetime.datetime.now().__str__(), self.logger['epoch'])
            for metric in headers:
                m_name, m_partition = metric.split('__')
                line = '{},{}'.format(line, performance[self.logger['epoch']][m_name][m_partition])
            line = '{},\n'.format(line)
            self._write_file(file_path=self.file_path, line=line, mode='a')
        return self

class ProgressBar(Callback):

    def __init__(self, show_batch_metrics = ['accuracy_score', 'log_loss']):
        super(Callback, self).__init__()
        self.show_batch_metrics = show_batch_metrics

    def on_epoch_begin(self):
        self.progbar = tqdm(total=self.logger['batches'], unit=' batches')
        self.epochs = self.logger['batches']

    def on_batch_end(self, y_train, yHat_train):
        self.progbar.update(1)
        desc_string = 'MODE[TRAIN] EPOCH[{}|{}]'.format(self.logger['epoch'], self.logger['epochs'])
        if self.show_batch_metrics is not None:
            for b_metric in self.show_batch_metrics:
                b_metric_val = self.logger['batch_metrics'][self.logger['batch']][b_metric].values()[0]
                b_metric_avg = geo_mean([d[b_metric].values()[0] for d in self.logger['batch_metrics'].values()])
                desc_string = '{} {}[{:.4f}|{:.4f}(avg)]'.format(desc_string, b_metric, b_metric_val, b_metric_avg)
        self.progbar.set_description(desc_string)
        return self

    def on_epoch_end(self, y_val, yHat_val, y_train, yHat_train):
        self.progbar.close()
        return self

class ModelCheckpoint(Callback):

    def __init__(self, save_folder_path, metric = 'log_loss', best_metric_highest = False, best_only = False, write_frequency = 1, verbose = True):
        super(Callback, self).__init__()
        self.save_folder_path = save_folder_path
        self.metric = metric
        self.best_metric_highest = best_metric_highest
        self.verbose = verbose
        self.best_only = best_only
        self.write_frequency = write_frequency
        if not os.path.exists(save_folder_path): os.makedirs(save_folder_path)

    def _save_checkpoint(self, best_performance, current_performance, dstn, improvement):
        checkpoint = {
            'epoch': self.logger['epoch'],
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, dstn)
        if self.verbose: print 'Performance  Epoch {} {} from {} to {}! Saving States & Optimizer to: {}'.format(self.logger['epoch'], 'improved' if improvement else 'changed', best_performance, current_performance, dstn)
        self.logger['best_performance'][self.metric] = {'value': current_performance , 'epoch': self.logger['epoch']}
        return self

    def on_train_begin(self):
        if not 'best_performance' in self.logger.keys(): self.logger['best_performance'] = {}
        self.logger['best_performance'][self.metric] = {'value': -np.inf if self.best_metric_highest else np.inf , 'epoch': 0}
        return self

    def on_epoch_end(self, **_):
        current_performance, best_performance, best_epoch = self.logger['epoch_metrics'][self.logger['epoch']][self.metric]['val'], self.logger['best_performance'][self.metric]['value'], self.logger['best_performance'][self.metric]['epoch']
        current_time = datetime.datetime.now().__str__()
        dstn_string = '{}__{}__{}'.format(current_time, current_performance, self.logger['epoch'])
        dstn = os.path.join(self.save_folder_path, '{}.pkl'.format(dstn_string))
        improvement = current_performance > best_performance if self.best_metric_highest else current_performance < best_performance
        do_write = (self.logger['epoch'] % self.write_frequency == 0) * improvement if self.best_only else (self.logger['epoch'] % self.write_frequency == 0)
        if do_write: self._save_checkpoint(best_performance=best_performance, current_performance=current_performance,dstn=dstn, improvement=improvement)
        return self

class EarlyStopping(ModelCheckpoint):

    def __init__(self, save_folder_path, metric = 'log_loss', best_metric_highest = False, best_only = False, write_frequency = 1, patience = 0, min_delta = 0, verbose = True):
        super(Callback, self).__init__()
        self.save_folder_path = save_folder_path
        self.metric = metric
        self.best_metric_highest = best_metric_highest
        self.verbose = verbose
        self.best_only = best_only
        self.write_frequency = write_frequency
        self.patience = patience
        self.min_delta = min_delta
        if not os.path.exists(save_folder_path): os.makedirs(save_folder_path)

    def _save_checkpoint(self, best_performance, current_performance, dstn, improvement):
        checkpoint = {
            'epoch': self.logger['epoch'],
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, dstn)
        if self.verbose: print 'Performance  Epoch {} {} from {} to {}! Saving States & Optimizer to: {}'.format(self.logger['epoch'], 'improved' if improvement else 'changed', best_performance, current_performance, dstn)
        self.logger['best_performance'][self.metric] = {'value': current_performance , 'epoch': self.logger['epoch']}
        return self

    def on_train_begin(self):
        if not 'best_performance' in self.logger.keys(): self.logger['best_performance'] = {}
        self.logger['best_performance'][self.metric] = {'value': -np.inf if self.best_metric_highest else np.inf , 'epoch': 0}
        return self

    def on_epoch_end(self, **_):
        current_performance, best_performance, best_epoch = self.logger['epoch_metrics'][self.logger['epoch']][self.metric]['val'], self.logger['best_performance'][self.metric]['value'], self.logger['best_performance'][self.metric]['epoch']
        current_time = datetime.datetime.now().__str__()
        dstn_string = '{}__{}__{}'.format(current_time, current_performance, self.logger['epoch'])
        dstn = os.path.join(self.save_folder_path, '{}.pkl'.format(dstn_string))
        improvement = current_performance - self.min_delta > best_performance if self.best_metric_highest else current_performance + self.min_delta < best_performance
        do_write = (self.logger['epoch'] % self.write_frequency == 0)  * improvement if self.best_only else (self.logger['epoch'] % self.write_frequency == 0)
        if do_write: self._save_checkpoint(best_performance = best_performance, current_performance = current_performance, dstn = dstn, improvement=improvement)
        if ((self.logger['epoch'] - best_epoch) > self.patience) & (not improvement):
            self.logger['continue_training'] = False
            if self.verbose: print '[STOP TRAIN] Patience of {} iterations reached!'.format(self.patience)
        return self

class TensorBoard(Callback):

    # TODO: add option to write images; find fix for graph

    def __init__(self, log_dir, update_frequency = 10):
        super(Callback, self).__init__()
        self.log_dir = log_dir
        self.writer = None
        self.update_frequency = update_frequency

    def on_train_begin(self, **_):
        self.writer = SummaryWriter(os.path.join(self.log_dir, datetime.datetime.now().__str__()))
        rndm_input = torch.autograd.Variable(torch.rand(1, *self.model.input_shape), requires_grad = True).to(self.logger['device'])
        fwd_pass = self.model(rndm_input)
        self.writer.add_graph(self.model, fwd_pass)
        return self

    def on_epoch_end(self, **_):
        if (self.logger['epoch'] % self.update_frequency) == 0:
            epoch_metrics = self.logger['epoch_metrics'][self.logger['epoch']]
            for e_metric, e_metric_dct in epoch_metrics.iteritems():
                for e_metric_split, e_metric_val in e_metric_dct.iteritems():
                    self.writer.add_scalar('{}/{}'.format(e_metric_split, e_metric), e_metric_val, self.logger['epoch'])
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name.replace('.', '/'), param.clone().cpu().data.numpy(), self.logger['epoch'])
        return self

    def on_train_end(self, **_):
        return self.writer.close()




