import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from algorithms import Models
from algorithms.train_LSTNet import train_lstn
from algorithms.learning import Learner
from algorithms.runHTS import hts_train_and_pred
from evaluation.plot import ts_plot, forecast_plot, forecast_plot_single
from preprocess import simulation, utils
from preprocess.AEdemand import get_ts
from preprocess.M3 import extract_ts
from preprocess.hierarchical import TreeNodes
from preprocess.wiki import get_agg_data
import pdb


def main(data, nodes, params, recon, H, h, quantile, dataset, hierarchical, verbose, cuda, gpu, plot=False):
    l, results = len(nodes) + 1 if hierarchical else 1, {}
    algs = params['alg']
    num_epochs = params['num_epoch']
    lr = params['lr']
    hidden_dim = params['hidden_dim']
    layer_dim = params['layer_dim']
    nonlinearity = params['nonlinearity']

    if algs == 'htsprophet':
        data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        training, testing = data[:-8], data[-8:]
        order_level = ['date'] + TreeNodes(nodes).col_order()
        training, testing = training[order_level], testing[order_level]
        hts_train_and_pred(train=training, test=testing, steps_ahead=h, node_list=order_level[1:])

    else:
        if hierarchical:
            data = data[TreeNodes(nodes).col_order()]
            node_list = data.columns.values
        else:
            node_list = ['0']

        if algs == 'LSTNet':
            Data = utils.Data_utility(0.6, 0.2, cuda, h, 24 * 7, data, recon, normalize=2)
            training, validation, testing = Data.train, Data.valid, Data.test
        else:
            Data = None
            training, validation, testing = data[:-2 * H], data[H: -H], data[2 * H:]

        if plot:
            ts_plot(training, l, dataset)
        models, quantile_models, optimizers, quantile_optimizers, combined_optimizers, lag = \
            Models.get_models_optimizers(node_list, algs, cuda, lr, hidden_dim, layer_dim, nonlinearity, Data)

        learner = Learner(models=models, optimizers=optimizers, quantile_models=quantile_models,
                          quantile_optimizers=quantile_optimizers, combined_optimizers=combined_optimizers,
                          num_epochs=num_epochs, train=training, validation=validation, test=testing, nodes=nodes,
                          node_list=node_list, recon=recon, alg=algs, lag=lag, H=H, h=h, l=l, hierarchical=hierarchical,
                          cuda=cuda, gpu=gpu, verbose=verbose, Data=Data)

        if recon == 'sharq' or recon == 'base':
            training_start = time.time()
            median_pred = learner.fit_median_recon() if recon == 'sharq' else learner.base_forecast()
            training_time = time.time() - training_start
            if quantile:
                learner.fit_quantile(median_pred)
            inference_start = time.time()
            results, loss = learner.predict(median_pred, quantile)
            inference_time = time.time() - inference_start
        elif recon == 'BU':
            training_start = time.time()
            median_pred = learner.bottom_forecast()
            training_time = time.time() - training_start

            inference_start = time.time()
            _, loss = learner.predict(median_pred, quantile)
            inference_time = time.time() - inference_start
        else:
            training_start = time.time()
            median_pred = learner.base_forecast()
            training_time = time.time() - training_start

            inference_start = time.time()
            _, loss = learner.predict(median_pred, quantile)
            inference_time = time.time() - inference_start

        print('Training time: {}; inference time {}'.format(training_time, inference_time))
        return results, training, loss


class sharq:

    def __init__(self, DATASET='User_specified', DATA=None, IF_TIME_SERIES=True, FORECAST_GRANULARITY='D', IS_HIERARCHICAL=False,
                 FORECAST_HORIZON=1, HIERARCHY_GRAPH=None, MODEL_HYPER_PARAMS=None, TRAINING_METHOD='base',
                 CATEGORICAL_FEATURES=None, QUANTILE=True, VERBOSE=False, EPOCH=1000, BATCH=128, GPU=0, SEED=78712):
        '''

        :param DATASET: Specify which dataset to use for experiment (default User specified if use your own data).
        :param DATA: User specified time series data (default None, cannot be None together with DATASET).
        :param IF_TIME_SERIES: Whether the user specified data is time series or sequence data (default False).
        :param FORECAST_GRANULARITY: Forecasting granularity for time series data (default daily).
        :param IS_HIERARCHICAL: Whether the user specified data contains hierarchical structure (default False).
        :param HIERARCHY_GRAPH: Hierarchical structure of user specified data in 2D array (default None).
        :param FORECAST_HORIZON: Forecasting horizon (default 1).
        :param TRAINING_METHOD: Reconciliation method (default base).
        :param CATEGORICAL_FEATURES: Categorical features to formulate hts (default None).
        :param MODEL_HYPER_PARAMS: Hyper-parameters of forecasting model, dictionary.
        Example parameter input (default):
        {'alg': 'rnn', 'num_epoch': 1000, 'lr': 0.1, 'hidden_dim': 5, 'layer_dim': 2, 'nonlinearity': 'tanh'}
        '''
        self.IF_TIME_SERIES = IF_TIME_SERIES
        self.FORECAST_GRANULARITY = FORECAST_GRANULARITY
        self.IS_HIERARCHICAL = IS_HIERARCHICAL
        self.h = FORECAST_HORIZON
        self.nodes = HIERARCHY_GRAPH
        self.CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
        self.DATASET = DATASET
        self.quantile = QUANTILE
        self.verbose = VERBOSE
        self.num_epoch = EPOCH
        self.data = DATA
        self.cuda = torch.cuda.is_available()
        self.gpu = GPU
        self.batch_size = BATCH
        self.params = {'alg': 'rnn', 'num_epoch': 1000, 'lr': 0.1, 'hidden_dim': 5, 'layer_dim': 2,
                       'nonlinearity': 'tanh'}
        if self.cuda:
            torch.cuda.set_device(self.gpu)
            torch.cuda.manual_seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        if MODEL_HYPER_PARAMS is not None and 'alg' in MODEL_HYPER_PARAMS:
            self.params['alg'] = MODEL_HYPER_PARAMS['alg']
        if MODEL_HYPER_PARAMS is not None and 'num_epoch' in MODEL_HYPER_PARAMS:
            self.params['num_epoch'] = MODEL_HYPER_PARAMS['num_epoch']
        if MODEL_HYPER_PARAMS is not None and 'lr' in MODEL_HYPER_PARAMS:
            self.params['lr'] = MODEL_HYPER_PARAMS['lr']
        if MODEL_HYPER_PARAMS is not None and 'hidden_dim' in MODEL_HYPER_PARAMS:
            self.params['hidden_dim'] = MODEL_HYPER_PARAMS['hidden_dim']
        if MODEL_HYPER_PARAMS is not None and 'layer_dim' in MODEL_HYPER_PARAMS:
            self.params['layer_dim'] = MODEL_HYPER_PARAMS['layer_dim']
        if MODEL_HYPER_PARAMS is not None and 'nonlinearity' in MODEL_HYPER_PARAMS:
            self.params['nonlinearity'] = MODEL_HYPER_PARAMS['nonlinearity']

        if not self.IS_HIERARCHICAL:
            self.TRAINING_METHOD = 'base'
        else:
            self.TRAINING_METHOD = TRAINING_METHOD
            if self.nodes is None and self.DATASET == 'User_specified':
                raise ValueError('Need to specify time series hierarchical structure.')

    def fit_and_predict(self):

        if self.DATASET == 'M5':
            self.data = pd.read_csv('./data/hierarchical_data.csv').drop(['Unnamed: 0'], axis=1)
            self.data.index = list(self.data['date'])
            self.data.drop(['date'], axis=1, inplace=True)
            self.data.columns = [str(i) for i in range(self.data.shape[1])]
            self.nodes = [[60]]

        elif self.DATASET == 'labour':
            self.data = pd.read_csv('./data/labour_force.csv').drop(['Unnamed: 0'], axis=1)
            self.nodes = [[2], [2, 2], [8, 8, 8, 8]]

        elif self.DATASET == 'sim_small':
            self.data, S, nbts = simulation.simulate('small')
            self.data = pd.DataFrame(self.data, columns=[str(i) for i in range(np.array(S).shape[0])])
            self.nodes = [[2], [2, 2]]

        elif self.DATASET == 'sim_large':
            self.data, S, nbts = simulation.simulate('large')
            self.data = pd.DataFrame(self.data, columns=[str(i) for i in range(np.array(S).shape[0])])
            self.nodes = [[10], [4] * 10, [4] * 40]

        elif self.DATASET == 'wiki':
            self.data = pd.read_csv('./data/wiki.csv').rename(columns={'Unnamed: 0': 'date'})
            self.data, self.nodes = get_agg_data(self.data)
            self.data = self.data.astype(float)

        elif self.DATASET == 'M3':
            self.data = extract_ts()
            self.data = pd.DataFrame(self.data, columns=[str(i) for i in range(19)])
            self.nodes = [[2], [2, 2], [3, 3, 3, 3]]

        elif self.DATASET == 'AEdemand':
            self.data = get_ts()
            self.data = pd.DataFrame(self.data, columns=[str(i) for i in range(22)])
            self.nodes = [[3], [2, 2, 2], [2, 2, 2, 2, 2, 2]]

        elif self.data is None:
            raise ValueError('Need to specify input data or pick an existing data set.')

        else:
            if self.IF_TIME_SERIES:
                self.data = self.data.resample(self.FORECAST_GRANULARITY).sum()
            # assume here we have the input data formatted as DataFrame and corresponding features

        horizons = np.arange(self.h)
        if not os.path.exists(os.getcwd() + '/runs'):
            os.mkdir(os.getcwd() + '/runs')
        if not os.path.exists(os.getcwd() + '/plots'):
            os.mkdir(os.getcwd() + '/plots')
        if not os.path.exists(os.getcwd() + '/save'):
            os.mkdir(os.getcwd() + '/save')

        if self.params['alg'] == 'LSTNet':
            multilevel_loss = train_lstn(self.TRAINING_METHOD, self.nodes, self.data, self.cuda, self.h, self.num_epoch,
                                         self.batch_size, self.params, self.verbose)

        elif self.params['alg'] in ['rnn', 'lstm', 'ar']:
            multi_step_pred = []
            multilevel_loss = np.zeros((self.h, len(self.nodes) + 1))
            global train
            for h in horizons:
                results, train, loss = main(data=self.data, nodes=self.nodes, params=self.params,
                                            recon=self.TRAINING_METHOD, h=h, quantile=self.quantile, H=self.h,
                                            dataset=self.DATASET, hierarchical=self.IS_HIERARCHICAL,
                                            verbose=self.verbose, cuda=self.cuda, gpu=self.gpu)
                multi_step_pred.append(results)
                multilevel_loss[h, :] = loss
            multilevel_loss = multilevel_loss.mean(axis=0)
            if self.IS_HIERARCHICAL:
                forecast_plot(train, multi_step_pred, self.DATASET)
            else:
                forecast_plot_single(train, multi_step_pred, self.DATASET)
        else:
            raise ValueError('Algorithm name not defined.')

        print('Dataset={}, \nalg={}, \ntraining method={}, \nmulti-level '
              'MAPE={}\n'.format(self.DATASET, self.params['alg'], self.TRAINING_METHOD, multilevel_loss))
        print('Program done!')
