import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import logging
from models.train_LSTNet import train_lstn
from models.Models import get_moe_optimizers
from recon.learning_moe import MoE_Learner
from evaluation.plot import ts_plot
from evaluation.metrics import compute_level_loss, compute_crps
from preprocess import utils
from preprocess.AEdemand import get_ts
from preprocess.EIA import get_ts_eia
from preprocess.sim_cp import get_cp_ts
from preprocess.M3 import extract_ts
from preprocess.hierarchical import TreeNodes
from preprocess.wiki import get_agg_data

logger = logging.getLogger('MECATS.mecats')


def mecats_main(data, parameter, nodes, dataset, online, cuda):
    date = pd.Series(pd.date_range(start='1/1/2000', periods=data.shape[0]), name='ds')
    if len(nodes) == 0:
        data = pd.DataFrame(data=data, columns=['0'])
    else:
        data = data[TreeNodes(nodes).col_order()]
    l = len(nodes) + 1
    node_list = data.columns.values
    Quantiles = False
    logger.info('<------------Pre-train model bank start------------>')
    if parameter['mecats'].QUANTILE and parameter['mecats'].RECON == 'sharq':
        Quantiles = True
        models, gns, optimizers, q_nets, q_optimizers = get_moe_optimizers(data=data, params=parameter, node_list=node_list, date=date, dataset=dataset, quantile=True, cuda=cuda)
        learner = MoE_Learner(models=models, optimizers=optimizers, gns=gns, params=parameter, data=data, date=date, nodes=nodes,
                              node_list=node_list, dataset=dataset, cuda=cuda, q_nets=q_nets, q_optimizers=q_optimizers)
    else:
        models, gns, optimizers = get_moe_optimizers(data=data, params=parameter, node_list=node_list, date=date, dataset=dataset, quantile=False, cuda=cuda)
        learner = MoE_Learner(models=models, optimizers=optimizers, gns=gns, params=parameter, data=data, date=date, nodes=nodes,
                              node_list=node_list, dataset=dataset, cuda=cuda)
    learner.fit_point_recon()
    if online:
        logger.info('<------------Online update using test data------------>')
        logger.info('Change point status is {}'.format(parameter['online'].enable_cp))
        step_loss_no_cp, l_range = learner.online_update(params=parameter['online'])
        
        if dataset == 'sim_cp':
            parameter['online'].enable_cp = 1 - parameter['online'].enable_cp
            logger.info('Change point status is {}'.format(parameter['online'].enable_cp))
            step_loss_cp, _ = learner.online_update(params=parameter['online'])
            utils.plot_loss_cp(step_loss_no_cp, step_loss_cp, l_range, parameter)
        logger.info('Program done!')
        sys.exit()
    else:
        logger.info('<------------Prediction on test data------------>')
        result, test, quantile = learner.test_pred()
        logger.info('Computing multi-level loss')
        loss = compute_level_loss(node_list, nodes, result, test, l)
        if Quantiles:
            crps = compute_crps(test, quantile, node_list, nodes, l)
            return loss, crps
        else:
            return loss, None


class mecats:

    def __init__(self, DATASET='User_specified', DATA=None, IF_TIME_SERIES=True, FORECAST_GRANULARITY='D', IS_HIERARCHICAL=False,
                 FORECAST_HORIZON=1, HIERARCHY_GRAPH=None, MODELS=None, RECON='base', ONLINE=0,
                 CATEGORICAL_FEATURES=None, QUANTILE=True, VERBOSE=False, GPU=0, SEED=78712):
        
        self.IF_TIME_SERIES = IF_TIME_SERIES
        self.FORECAST_GRANULARITY = FORECAST_GRANULARITY
        self.IS_HIERARCHICAL = IS_HIERARCHICAL
        self.h = FORECAST_HORIZON
        self.DATASET = DATASET
        self.quantile = QUANTILE
        self.verbose = VERBOSE
        self.cuda = torch.cuda.is_available()
        self.gpu = GPU
        self.algs = MODELS
        self.online = ONLINE
        
        if isinstance(DATA, str):
            self.data = None
        else:
            self.data = DATA
        if isinstance(HIERARCHY_GRAPH, str):
            self.nodes = None
        else:
            self.nodes = HIERARCHY_GRAPH
        if isinstance(CATEGORICAL_FEATURES, str):
            self.CATEGORICAL_FEATURES = None
        else:
            self.CATEGORICAL_FEATURES = CATEGORICAL_FEATURES

        if self.cuda:
            torch.cuda.set_device(self.gpu)
            torch.cuda.manual_seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        if not self.IS_HIERARCHICAL:
            self.RECON = 'base'
        else:
            self.RECON = RECON
            if self.nodes is None and self.DATASET == 'User_specified':
                raise ValueError('Need to specify time series hierarchical structure.')
        
        if not os.path.exists(os.getcwd() + '/plots'):
            os.mkdir(os.getcwd() + '/plots')
        if not os.path.exists(os.getcwd() + '/save'):
            os.mkdir(os.getcwd() + '/save')
            os.mkdir(os.getcwd() + '/save/lstnet')
        if not os.path.exists(os.getcwd() + '/results'):
            os.mkdir(os.getcwd() + '/results')
            os.mkdir(os.getcwd() + '/results/deepar_fig')
            os.mkdir(os.getcwd() + '/results/forecast')
            os.mkdir(os.getcwd() + '/results/loss')
            os.mkdir(os.getcwd() + '/results/weights')

        utils.set_logger(os.path.join('./save', 'history_{}_{}_{}.log'.format(self.DATASET, self.algs, self.RECON)))
        logger.info('\n\n<---------------NEW RUN--------------->')
        logger.info('Method: {} | Model: {}'.format(self.RECON, self.algs))
        logger.info('Parameter settings:\n')

        self.parameter_set = {}
        path = os.path.join('parameters', 'mecats.json')
        self.parameter_set['mecats'] = utils.Params(path)
        with open(path) as f:
            mecats_params = json.load(f)
        logger.info('\n MECATS: \n {}'.format(mecats_params))

        if self.algs == 'moe' or self.algs == 'lstnet':
            path = os.path.join('parameters', 'lstnet_params.json')
            self.parameter_set['lstnet'] = utils.Params(path)
            with open(path) as f:
                lstnet_params = json.load(f)
            logger.info('\n LSTNet: \n {}'.format(lstnet_params))

        if self.algs == 'rnn':
            path = os.path.join('parameters', 'rnn_params.json')
            self.parameter_set['rnn'] = utils.Params(path)
            with open(path) as f:
                rnn_params = json.load(f)
            logger.info('\n RNN: \n {}'.format(rnn_params))

        if self.algs == 'moe':
            path = os.path.join('parameters', 'pydlm.json')
            self.parameter_set['pydlm'] = utils.Params(path)
            with open(path) as f:
                pydlm_params = json.load(f)
            logger.info('\n PyDLM: \n {}'.format(pydlm_params))
            
            path = os.path.join('parameters', 'deepar_params.json')
            self.parameter_set['deepar'] = utils.Params(path)
            with open(path) as f:
                deepar_params = json.load(f)
            logger.info('\n DeepAR: \n {}'.format(deepar_params))

            path = os.path.join('parameters', 'gating_network_params.json')
            self.parameter_set['gating_network'] = utils.Params(path)
            with open(path) as f:
                gating_network_params = json.load(f)
            logger.info('\n Gating Network: \n {}'.format(gating_network_params))

            if mecats_params["QUANTILE"]:   
                path = os.path.join('parameters', 'quantile_net.json')
                self.parameter_set['quantile_net'] = utils.Params(path)
                with open(path) as f:
                    quantile_net_params = json.load(f)
                assert quantile_net_params['num_layers'] == len(quantile_net_params['layer_dims'])
                # assert quantile_net_params['d'] == quantile_net_params['layer_dims'][-1]
                logger.info('\n Uncertainty Wrapper Network: \n {}'.format(quantile_net_params))

            if self.online:
                path = os.path.join('parameters', 'online_params.json')
                self.parameter_set['online'] = utils.Params(path)
                with open(path) as f:
                    online_params = json.load(f)
                if DATASET == 'EIA':
                    assert online_params['enable_cp'] == 1
                logger.info('\n Online parameters: \n {}'.format(online_params))

    def fit_and_predict(self):

        if self.DATASET == 'M5':
            self.data = pd.read_csv('./data/hierarchical_data.csv').drop(['Unnamed: 0'], axis=1)
            self.data.index = list(self.data['date'])
            self.data.drop(['date'], axis=1, inplace=True)
            self.data.columns = [str(i) for i in range(self.data.shape[1])]
            self.nodes = [[60]]

        elif self.DATASET == 'labour':
            if self.algs != 'moe':
                self.data = pd.read_csv('./data/labour_force.csv').drop(['Unnamed: 0'], axis=1)
            else:
                self.data = pd.read_csv('./data/labour_force.csv').rename(columns={'Unnamed: 0': 'ds'})
            self.nodes = [[2], [2, 2], [8, 8, 8, 8]]

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

        elif self.DATASET == 'EIA':
            self.data = get_ts_eia()
            self.nodes = []

        elif self.DATASET == 'sim_cp':
            self.data = get_cp_ts()
            self.nodes = []

        elif self.data is None:
            raise ValueError('Need to specify input data or pick an existing data set.')

        else:
            if self.IF_TIME_SERIES:
                self.data = self.data.resample(self.FORECAST_GRANULARITY).sum()
            # assume here we have the input data formatted as DataFrame and corresponding features
        if self.cuda:
            logger.info('Using CUDA | GPU {}'.format(self.gpu))
        else:
            logger.info('NOT using CUDA')
        logger.info('Dataset {} loaded!'.format(self.DATASET))

        if self.algs == 'moe':
            multilevel_loss, crps = mecats_main(data=self.data, parameter=self.parameter_set, nodes=self.nodes, 
                                                online=self.online, dataset=self.DATASET, cuda=self.cuda)
            if crps is not None:
                logger.info('CRPS by level: {}'.format(crps))

        elif 'mint' in self.RECON:
            self.data = self.data[TreeNodes(self.nodes).col_order()]
            multilevel_loss = train_lstn(self.RECON, self.nodes, self.data, self.cuda, self.parameter_set, 
                                         self.DATASET, self.verbose)
        else:
            raise ValueError('Algorithm name not defined.')

        logger.info('\nDataset={}, \nalg={}, \ntraining method={}, \nmulti-level '
                    'MAPE={}\n'.format(self.DATASET, self.algs, self.RECON, multilevel_loss))
        logger.info('Program done!')
        print('\nDataset={}, \nalg={}, \ntraining method={}, \nmulti-level MAPE={}\n'.format(self.DATASET, self.algs, self.RECON, multilevel_loss))
        print('Program done!')
