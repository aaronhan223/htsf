from pmdarima.compat import pandas
import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, dataset
import logging
import json
import shutil
import hiertsforecaster.models.deepar as net
from hiertsforecaster.preprocess import train_config
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import seaborn as sns
import os
import pdb

logger = logging.getLogger('MECATS.Utils')

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set.
    def __init__(self, train, valid, cuda, horizon, window, dataset, normalize):
        self.tr = train
        self.va = valid
        self.te = train_config.MecatsParams.pred_step
        self.metric_mode = train_config.MecatsParams.metric_mode
        self.cuda = cuda
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
        if self.m != 1:
            self.feats = dataset.columns.values
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        if self.metric_mode:
            self._split(int(self.tr * (self.n - self.te)), int((self.tr + self.va) * (self.n - self.te)))
        else:
            self._split(int(self.tr * self.n), int((self.tr + self.va) * self.n))

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.valid[1] * self.scale.expand(self.valid[1].size(0), self.m)

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

    def _split(self, train, valid):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set)
        if len(valid_set) == 0:
            self.valid = self.train
        else:
            self.valid = self._batchify(valid_set)
            # self.test_prep = self._batchify(valid_set)
        self.test = self._batchify(test_set)
        # if self.method in ['sharq', 'mecats']:
        # self.method = 'base'
        if isinstance(self.dataset, (np.ndarray, np.generic)):
            self.test_data = self.dataset[test_set]
        else:
            self.test_data = self.dataset.iloc[test_set]

    def _batchify(self, idx_set):

        # idx_set is the range for end point, window size is the length for looking back
        n = len(idx_set)
        if self.m != 1:
            X = np.zeros((n, self.P, self.m))
            Y = np.zeros((n, self.m))
        else:
            X = torch.zeros((n, self.P, self.m))
            Y = torch.zeros((n, self.m))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            if self.m != 1:
                X[i, :, :] = self.dat[start:end, :]
                Y[i, :] = self.dat[idx_set[i], :]
            else:
                X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
                Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])

        if self.m != 1:
            return [dict(zip(self.feats, np.split(X, X.shape[2], 2))),
                    dict(zip(self.feats, np.split(Y, Y.shape[1], 1)))]
        else:
            return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle):
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


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor

        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )

    def __call__(self, loss):
        self.lr_scheduler.step(loss)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            logger.info('Early stopping counter {} of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                logger.info('Early stopping.')
                self.early_stop = True


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


class TrainDataset(Dataset):
    def __init__(self, data, window, set_range, config):
        self.data, self.label, _ = prep_deepar_data(window, data, True, set_range, config)
        self.train_len = self.data.shape[0]

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index, :, :], 0, self.label[index])


class ValidDataset(Dataset):
    def __init__(self, data, window, set_range, config):
        self.data, self.label, self.v = prep_deepar_data(window, data, False, set_range, config)
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


def prep_deepar_data(window, data, train, set_range, config):
    data = np.expand_dims(data, 1)
    params = config.DeeparParams
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


def prepare_data(set_range, data, window, h=1):
    n = len(set_range)
    rawdat = data.values
    if len(rawdat.shape) == 1:
        rawdat = np.expand_dims(rawdat, 1)
    X = torch.zeros((n, window, rawdat.shape[1]))
    for i in range(n):
        end = set_range[i] - h + 1
        start = end - window
        X[i, :, :] = torch.from_numpy(rawdat[start:end, :])
    return X


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


def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    print(f'Checkpoint saved to {filepath}')
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
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

    return model


def check_directory(plot, unit_test, realm_id):
    if not unit_test:
        if not os.path.exists(os.getcwd() + f'/hiertsforecaster/save/{realm_id}'):
            os.mkdir(os.getcwd() + f'/hiertsforecaster/save/{realm_id}')
            os.mkdir(os.getcwd() + f'/hiertsforecaster/save/{realm_id}/lstnet')
        if plot and not os.path.exists(os.getcwd() + '/hiertsforecaster/plots'):
            os.mkdir(os.getcwd() + '/hiertsforecaster/plots')
        if plot and not os.path.exists(os.getcwd() + '/hiertsforecaster/results'):
            os.mkdir(os.getcwd() + '/hiertsforecaster/results')
            os.mkdir(os.getcwd() + '/hiertsforecaster/results/deepar_fig')
            os.mkdir(os.getcwd() + '/hiertsforecaster/results/forecast')
            os.mkdir(os.getcwd() + '/hiertsforecaster/results/loss')
            os.mkdir(os.getcwd() + '/hiertsforecaster/results/weights')


def set_seed(params, cuda, seed):
    if cuda:
        torch.cuda.manual_seed(seed)
        os.environ['MASTER_ADDR'] = params.ip_address
        os.environ['MASTER_PORT'] = params.port
        logger.info('Using CUDA | {} GPU | {} nodes'.format(params.n_gpu, params.n_nodes))
    else:
        logger.info('NOT using CUDA')
    torch.manual_seed(seed)
    np.random.seed(seed)


def write_data_to_dir(result, quantile, metric_mode, loss, crps) -> None:
    '''
    This is for unit test.
    '''
    saved_point_pred = pd.DataFrame(data=np.vstack([*result.values()]).T, columns=['0', '1', '2'])
    saved_quant_pred = pd.DataFrame(data=np.hstack([*quantile.values()]), columns=['95_0', '70_0', '30_0', '5_0', '95_1', '70_1', '30_1', '5_1', '95_2', '70_2', '30_2', '5_2'])
    if metric_mode == 0:
        date_series = pd.date_range(start=pd.Timestamp('2019-09-01 00:00:00'), periods=saved_point_pred.shape[0], freq='MS')
        saved_point_pred.insert(0, 'ds', date_series)
        saved_quant_pred.insert(0, 'ds', date_series)
        saved_point_pred.to_csv('./pred_res/point_prediction_metrics_false.csv')
        saved_quant_pred.to_csv('./pred_res/quant_prediction_metrics_false.csv')
    else:
        assert loss is not None, 'loss cannot be None type!'
        assert crps is not None, 'crps cannot be None type!'
        date_series = pd.date_range(start=pd.Timestamp('2018-09-01 00:00:00'), periods=saved_point_pred.shape[0], freq='MS')
        saved_point_pred.insert(0, 'ds', date_series)
        saved_quant_pred.insert(0, 'ds', date_series)
        saved_point_pred.to_csv('./pred_res/point_prediction_metrics_true.csv')
        saved_quant_pred.to_csv('./pred_res/quant_prediction_metrics_true.csv')
        np.savez('./pred_res/metrics.npz', mape=loss, crps=crps) 


def plot_weights(weights, num_epochs, level, node, legend, experts):
    logger.info('Plot weights for node {} | level {}'.format(node, level))
    keys = [str(i) for i in range(len(experts))]
    color_list = ['#D05A6E', '#3A8FB7', '#24936E', '#939650', '#43341B']
    colors = dict(zip(keys, color_list[:len(keys)]))
    legends = dict(zip(keys, experts[:len(keys)]))
    # markers = {'d': 'o', 'a': 'o', 'f': 'o', 'l': 'o', 'da': 'o'}
    # linestyles = {'d': 'solid', 'a': 'solid', 'f': 'solid', 'l': 'solid', 'da': 'solid'}
    plt.rcParams["figure.figsize"] = (8.5, 7.5)

    x = np.arange(num_epochs)
    weight_list = [weights[:, i] for i in weights.shape[1]]
    data = dict(zip(keys, weight_list))

    for key in keys:
        plt.plot(x, data[key], label=legends[key], color=colors[key], lw=8, ls='solid', zorder=1)
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
    plt.savefig('./hiertsforecaster/results/weights/{}_{}.pdf'.format(level, node))
    plt.close()


def forecast_plot(train, pred, target, quantiles, level, node, legend):
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
    if legend:
        plt.legend(('Training Samples', 'True Value', 'Point Prediction', '95-5 Quantile', '70-30 Quantile'), loc='upper left', fontsize=18, fancybox=True)
    # for i in range(n_quantiles):
    #     plt.plot(t_test, np.concatenate((np.array([train_plot[-1]]), quantiles[:, i])), color='#C1328E', linewidth=1)
    plt.axvline(x=t_train[-1], color='k', linestyle='-', linewidth=3)
    plt.tick_params(labelsize=20)
    plt.grid()
    plt.xlabel('Time Stamp', fontsize=24)
    plt.title('Level {} Vertex {}'.format(level, node), fontsize=30)
    plt.savefig('./hiertsforecaster/results/forecast/{}_{}.pdf'.format(level, node))
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
    plt.savefig('./hiertsforecaster/results/forecast/qplot_{}_{}_{}.pdf'.format(dataset, level, node))
    plt.close()

    
def plot_weights_cp(training_weights, online_weights, epochs_precp, epochs_aftcp, experts):
    logger.info('Plot weights under change points.')
    keys = [str(i) for i in range(len(experts))]
    color_list = ['#D05A6E', '#3A8FB7', '#24936E', '#939650', '#43341B']
    colors = dict(zip(keys, color_list[:len(keys)]))
    legends = dict(zip(keys, experts[:len(keys)]))
    # linestyles = {'d': 'solid', 'a': 'solid', 'f': 'solid', 'l': 'solid', 'da': 'solid'}
    plt.rcParams["figure.figsize"] = (18, 8)

    weights = np.concatenate((training_weights, online_weights))
    t = np.arange(epochs_precp + epochs_aftcp)
    weight_list = [weights[:, i] for i in weights.shape[1]]
    data = dict(zip(keys, weight_list))

    for key in keys:
        plt.plot(t, data[key], label=legends[key], color=colors[key], lw=8, ls='solid', zorder=1)
    
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
    plt.savefig('./hiertsforecaster/results/weights/weights.pdf')
    plt.close()


def plot_loss_cp(epoch_loss_no_cp, epoch_loss_cp, epoch):
    logger.info('Plot loss under change points.')
    keys = ['c', 'n']
    colors = {'c': '#FFC408', 'n': '#43341B'}
    legends = {'c': 'With CP Detection', 'n': 'Without CP Detection'}
    linestyles = {'c': 'solid', 'n': 'solid'}
    plt.rcParams["figure.figsize"] = (10, 8.5)

    start, end = train_config.gating_network_params.num_epochs, train_config.gating_network_params.num_epochs + epoch * train_config.online_params.epoch
    t = np.arange(start, end, train_config.online_params.epoch)
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
    plt.savefig('./hiertsforecaster/results/loss/sim_cp_loss.pdf')
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
    plt.savefig('./hiertsforecaster/results/forecast/EIA_rl.pdf')
    plt.close()