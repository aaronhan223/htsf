import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, dataset
import logging
import json
import shutil
import models.deepar as net
from preprocess.hierarchical import TreeNodes
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import seaborn as sns
import os
import pdb

logger = logging.getLogger('MECATS.Utils')

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, train, valid, cuda, horizon, window, dataset, method, normalize):
        self.tr = train
        self.va = valid
        self.cuda = cuda
        self.method = method
        self.dataset = dataset
        self.P = window
        self.h = horizon
        if isinstance(dataset, (np.ndarray, np.generic)):
            self.rawdat = dataset
        else:
            self.rawdat = dataset.values
        if len(self.rawdat.shape) == 1:
            self.rawdat = np.expand_dims(self.rawdat, 1)

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        if self.method == 'sharq' and self.m != 1:
            self.feats = dataset.columns.values
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(self.tr * self.n), int((self.tr + self.va) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test_prep[1] * self.scale.expand(self.test_prep[1].size(0), self.m)

        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix before train/test split.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        if len(valid_set) == 0:
            self.valid = self.train
        else:
            self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)
        if self.method == 'sharq':
            self.method = 'base'
        self.test_prep = self._batchify(test_set, self.h)
        if isinstance(self.dataset, (np.ndarray, np.generic)):
            self.test_data = self.dataset[test_set]
        else:
            self.test_data = self.dataset.iloc[test_set]

    def _batchify(self, idx_set, horizon):

        # idx_set is the range for end point, window size is the length for looking back
        n = len(idx_set)
        if self.method == 'sharq' and self.m != 1:
            X = np.zeros((n, self.P, self.m))
            Y = np.zeros((n, self.m))
        else:
            X = torch.zeros((n, self.P, self.m))
            Y = torch.zeros((n, self.m))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            if self.method == 'sharq' and self.m != 1:
                X[i, :, :] = self.dat[start:end, :]
                Y[i, :] = self.dat[idx_set[i], :]
            else:
                X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
                Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])

        if self.method == 'sharq' and self.m != 1:
            return [dict(zip(self.feats, np.split(X, X.shape[2], 2))),
                    dict(zip(self.feats, np.split(Y, Y.shape[1], 1)))]
        else:
            return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size


def set_logger(log_path):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `save/history.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    _logger = logging.getLogger('MECATS')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)


class Params:
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        '''Loads parameters from json file'''
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)


class TrainDataset(Dataset):
    def __init__(self, data, window, set_range):
        self.data, self.label, _ = prep_data(window, data, True, set_range)
        self.train_len = self.data.shape[0]

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index, :, :], 0, self.label[index])


class ValidDataset(Dataset):
    def __init__(self, data, window, set_range):
        self.data, self.label, self.v = prep_data(window, data, False, set_range)
        self.valid_len = self.data.shape[0]

    def __len__(self):
        return self.valid_len

    def __getitem__(self, index):
        return (self.data[index, :, :], 0, self.v[index], self.label[index])


class GatingDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :, :]


class QuantileDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :]


def prep_data(window, data, train, set_range):
    data = np.expand_dims(data, 1)
    params = Params(os.path.join('parameters', 'deepar_params.json'))
    num_series, stride = 1, params.stride
    total_windows, n = len(range(0, len(set_range), stride)), len(set_range)
    # stride for forecasting window, larger then smoother, but normally we don't have very long seq
    print('Stride size set to {} for DeepAR'.format(stride))

    x_input = np.zeros((total_windows, window, num_series), dtype='float32')
    label = np.zeros((total_windows, window), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    input_size = window - stride

    for i in range(0, n, stride):
        window_end = set_range[i]
        window_start = window_end - window
        x_input[i, 1:, :] = data[window_start: window_end - 1]
        label[i, :] = data[window_start: window_end, 0]
        nonzero_sum = (x_input[i, 1:input_size, 0] != 0).sum()
        if nonzero_sum == 0:
            v_input[i, 0] = 0
        else:
            v_input[i, 0] = np.true_divide(x_input[i, 1:input_size, 0].sum(), nonzero_sum) + 1
            x_input[i, :, 0] = x_input[i, :, 0] / v_input[i, 0]
            if train:
                label[i, :] = label[i, :] / v_input[i, 0]

    return x_input, label, v_input


def init_metrics(sample=True):
    metrics = {
        'ND': np.zeros(2),  # numerator, denominator
        'RMSE': np.zeros(3),  # numerator, denominator, time step count
        'test_loss': np.zeros(2),
    }
    if sample:
        metrics['rou90'] = np.zeros(2)
        metrics['rou50'] = np.zeros(2)
    return metrics


def update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, predict_start, samples=None, relative=False):
    raw_metrics['ND'] = raw_metrics['ND'] + net.accuracy_ND(sample_mu, labels[:, predict_start:], relative=relative)
    raw_metrics['RMSE'] = raw_metrics['RMSE'] + net.accuracy_RMSE(sample_mu, labels[:, predict_start:], relative=relative)
    input_time_steps = input_mu.numel()
    raw_metrics['test_loss'] = raw_metrics['test_loss'] + [
        net.loss_fn(input_mu, input_sigma, labels[:, :predict_start]) * input_time_steps, input_time_steps]
    if samples is not None:
        raw_metrics['rou90'] = raw_metrics['rou90'] + net.accuracy_ROU(0.9, samples, labels[:, predict_start:], relative=relative)
        raw_metrics['rou50'] = raw_metrics['rou50'] + net.accuracy_ROU(0.5, samples, labels[:, predict_start:], relative=relative)
    return raw_metrics


def get_metrics(sample_mu, labels, predict_start, samples=None, relative=False):
    metric = dict()
    metric['ND'] = net.accuracy_ND_(sample_mu, labels[:, predict_start:], relative=relative)
    metric['RMSE'] = net.accuracy_RMSE_(sample_mu, labels[:, predict_start:], relative=relative)
    if samples is not None:
        metric['rou90'] = net.accuracy_ROU_(0.9, samples, labels[:, predict_start:], relative=relative)
        metric['rou50'] = net.accuracy_ROU_(0.5, samples, labels[:, predict_start:], relative=relative)
    return metric


def final_metrics(raw_metrics, sampling=False):
    summary_metric = {}
    summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
    summary_metric['RMSE'] = np.sqrt(raw_metrics['RMSE'][0] / raw_metrics['RMSE'][2]) / (
                raw_metrics['RMSE'][1] / raw_metrics['RMSE'][2])
    summary_metric['test_loss'] = (raw_metrics['test_loss'][0] / raw_metrics['test_loss'][1]).item()
    if sampling:
        summary_metric['rou90'] = raw_metrics['rou90'][0] / raw_metrics['rou90'][1]
        summary_metric['rou50'] = raw_metrics['rou50'][0] / raw_metrics['rou50'][1]
    return summary_metric


def save_checkpoint(state, is_best, epoch, checkpoint, dataset, method, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_{dataset}_{method}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}_{dataset}_{method}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    print(f'Checkpoint saved to {filepath}')
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, '{}_{}_best.pth.tar'.format(dataset, method)))
        print('Best checkpoint copied to best.pth.tar')


def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def plot_all_epoch(variable, save_name, location='./figures/'):
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()


def load_checkpoint(checkpoint, model, optimizer=None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def plot_weights(weights, num_epochs, level, node, dataset, method, legend):
    logger.info('Plot weights for node {} | level {} | dataset {}'.format(node, level, dataset))
    keys = ['d', 'a', 'f', 'l', 'da']
    colors = {'d': '#D05A6E', 'a': '#3A8FB7', 'f': '#24936E', 'l': '#939650', 'da': '#43341B'}
    legends = {'d': 'DLM', 'a': 'Auto-ARIMA', 'f': 'FB-Prophet', 'l': 'LSTNet', 'da': 'Deep-AR'}
    markers = {'d': 'o', 'a': 'o', 'f': 'o', 'l': 'o', 'da': 'o'}
    linestyles = {'d': 'solid', 'a': 'solid', 'f': 'solid', 'l': 'solid', 'da': 'solid'}
    plt.rcParams["figure.figsize"] = (8.5, 7.5)

    x = np.arange(num_epochs)
    dlm, arima, prophet, lstnet, deepar = weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3], weights[:, 4]
    data = {'d': dlm, 'a': arima, 'f': prophet, 'l': lstnet, 'da': deepar}

    for key in keys:
        plt.plot(x, data[key], label=legends[key], color=colors[key], lw=8, ls=linestyles[key], zorder=1)
    if legend:
        plt.legend(loc='upper left', fontsize=27, fancybox=True)
    plt.xlim([-1, num_epochs + 1])
    plt.xticks(np.arange(0, num_epochs + 1, 400), fontsize=23)
    plt.ylim([-0.01, 1.01])
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=23)
    plt.grid()
    plt.xlabel('Epochs', fontsize=26)
    plt.ylabel('Expert Weights', fontsize=26)
    plt.title('Level {} Vertex {}'.format(level, node), fontsize=30)
    plt.savefig('./results/weights/{}_{}_{}_{}.pdf'.format(method, dataset, level, node))
    plt.close()


def forecast_plot(train, pred, target, quantiles, level, node, dataset, legend):
    t = np.arange(len(train) + len(pred))
    train_display, test_display = len(train), len(pred)
    t_train, t_test = t[-(train_display + test_display): -test_display], t[-(test_display + 1):]
    train_plot = np.array(train[-train_display:])

    target_plot = np.concatenate((np.array([train_plot[-1]]), target))
    pred_plot = np.concatenate((np.array([train_plot[-1]]), pred))
    # n_quantiles = quantiles.shape[1]
    quantile_0 = np.concatenate((np.array([train_plot[-1]]), quantiles[:, 0]))
    quantile_1 = np.concatenate((np.array([train_plot[-1]]), quantiles[:, 1]))
    quantile_2 = np.concatenate((np.array([train_plot[-1]]), quantiles[:, 2]))
    quantile_3 = np.concatenate((np.array([train_plot[-1]]), quantiles[:, 3]))

    plt.rcParams["figure.figsize"] = (8.5, 7.5)
    plt.plot(t_train, train_plot, color='k', linewidth=1)
    plt.plot(t_test, target_plot, color='#C1328E', linewidth=1)
    plt.plot(t_test, pred_plot, color='#21a5e3', linewidth=1)
    plt.fill_between(t_test, quantile_3, quantile_0, color='#21a5e3', alpha=.1)
    plt.fill_between(t_test, quantile_2, quantile_1, color='#21a5e3', alpha=.17)
    if dataset != 'sim_small' and legend:
        plt.legend(('Training Samples', 'True Value', 'Point Prediction', '95-5 Quantile', '70-30 Quantile'), loc='upper left', fontsize=18, fancybox=True)
    # for i in range(n_quantiles):
    #     plt.plot(t_test, np.concatenate((np.array([train_plot[-1]]), quantiles[:, i])), color='#C1328E', linewidth=1)
    plt.axvline(x=t_train[-1], color='k', linestyle='-', linewidth=3)
    plt.tick_params(labelsize=20)
    if dataset != 'sim_small':
        plt.grid()
    plt.xlabel('Time Stamp', fontsize=24)
    plt.title('Level {} Vertex {}'.format(level, node), fontsize=30)
    plt.savefig('./results/forecast/{}_{}_{}.pdf'.format(dataset, level, node))
    plt.close()


def quantile_forecast_plot(train, pred, target, quantiles, level, node, dataset):
    t = np.arange(len(train) + len(pred))
    train_display, test_display = len(train), len(pred)
    t_train, t_test = t[-(train_display + test_display): -test_display], t[-(test_display + 1):]
    train_plot = np.array(train[-train_display:])

    target_plot = np.concatenate((np.array([train_plot[-1]]), target))
    pred_plot = np.concatenate((np.array([train_plot[-1]]), pred))
    n_quantiles = quantiles.shape[1]

    plt.rcParams["figure.figsize"] = (8.5, 7.5)
    plt.plot(t_train, train_plot, color='k', linewidth=1)
    plt.plot(t_test, target_plot, color='#a5e321', linewidth=1)
    plt.plot(t_test, pred_plot, color='#21a5e3', linewidth=1)

    for i in range(n_quantiles):
        plt.plot(t_test, np.concatenate((np.array([train_plot[-1]]), quantiles[:, i])), color='#C1328E', linewidth=1)
        if i == 0:
            plt.legend(('Training Samples', 'True Value', 'Point Prediction', 'Quantiles'), loc='upper left', fontsize=18, fancybox=True)
    plt.axvline(x=t_train[-1], color='k', linestyle='-', linewidth=3)
    plt.tick_params(labelsize=20)
    plt.grid()
    plt.xlabel('Time Stamp', fontsize=24)
    plt.title('Level {} Vertex {}'.format(level, node), fontsize=30)
    plt.savefig('./results/forecast/qplot_{}_{}_{}.pdf'.format(dataset, level, node))
    plt.close()

    
def plot_weights_cp(training_weights, online_weights, epochs_precp, epochs_aftcp, dataset):
    logger.info('Plot weights under change points.')
    keys = ['d', 'a', 'f', 'l', 'da']
    colors = {'d': '#D05A6E', 'a': '#3A8FB7', 'f': '#24936E', 'l': '#939650', 'da': '#43341B'}
    legends = {'d': 'DLM', 'a': 'Auto-ARIMA', 'f': 'FB-Prophet', 'l': 'LSTNet', 'da': 'Deep-AR'}
    linestyles = {'d': 'solid', 'a': 'solid', 'f': 'solid', 'l': 'solid', 'da': 'solid'}
    plt.rcParams["figure.figsize"] = (18, 8)

    weights = np.concatenate((training_weights, online_weights))
    t = np.arange(epochs_precp + epochs_aftcp)
    dlm, arima, prophet, lstnet, deepar = weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3], weights[:, 4]
    data = {'d': dlm, 'a': arima, 'f': prophet, 'l': lstnet, 'da': deepar}

    for key in keys:
        plt.plot(t, data[key], label=legends[key], color=colors[key], lw=8, ls=linestyles[key], zorder=1)
    
    plt.axvline(x=t[epochs_precp], color='k', linestyle='-', linewidth=3, zorder=2)
    plt.axvline(x=2050, color='#F75C2F', linestyle='-', linewidth=3, zorder=2)
    plt.axvline(x=2700, color='#90B44B', linestyle='-', linewidth=3, zorder=2)
    plt.legend(loc='upper left', fontsize=27, fancybox=True)
    plt.xlim([-1, epochs_precp + epochs_aftcp + 1])
    plt.xticks(np.arange(0, epochs_precp + epochs_aftcp + 1, 400), fontsize=24)
    plt.ylim([-0.01, 1.01])
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=24)
    plt.grid()
    plt.xlabel('Steps', fontsize=30)
    plt.ylabel('Model Weights', fontsize=30)
    # plt.title('Model Bank Weights with Change Points', fontsize=30)
    plt.savefig('./results/weights/{}_weights.pdf'.format(dataset))
    plt.close()


def plot_loss_cp(epoch_loss_no_cp, epoch_loss_cp, epoch, params):
    logger.info('Plot loss under change points.')
    keys = ['c', 'n']
    colors = {'c': '#FFC408', 'n': '#43341B'}
    legends = {'c': 'With CP Detection', 'n': 'Without CP Detection'}
    linestyles = {'c': 'solid', 'n': 'solid'}
    plt.rcParams["figure.figsize"] = (10, 8.5)

    start, end = params['gating_network'].num_epochs, params['gating_network'].num_epochs + epoch * params['online'].epoch
    t = np.arange(start, end, params['online'].epoch)
    loss_cp = np.log10(epoch_loss_cp)
    loss_no_cp = np.log10(epoch_loss_no_cp)
    data = {'c': loss_cp, 'n': loss_no_cp}

    for key in keys:
        plt.plot(t, data[key], label=legends[key], color=colors[key], lw=3, ls=linestyles[key], zorder=1)
    
    plt.axvline(x=2700, color='#90B44B', linestyle='-', linewidth=2, zorder=2)
    plt.legend(loc='lower right', fontsize=27, fancybox=True)
    plt.xlim([start - 1, end + 1])
    plt.xticks(np.arange(start, end + 1, 500), fontsize=24)
    plt.ylim([-7.5, 2.01])
    plt.yticks([-7, -4, -1, 2], fontsize=24)
    plt.grid()
    plt.xlabel('Steps', fontsize=30)
    plt.ylabel(r'$\log_{10}$ loss', fontsize=30)
    # plt.title('Online Predictive Loss with Change Points', fontsize=27)
    plt.savefig('./results/loss/sim_cp_loss.pdf')
    plt.close()


def plot_rl_cp(T, data, R, pmean, pvar):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    ax1, ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    
    # Plot predictions.
    ax1.plot(range(0, T), pmean, c='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
    ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

    # R is a 2d lower triangular matrix
    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)

    plt.tight_layout()
    plt.savefig('./results/forecast/EIA_rl.pdf')
    plt.close()