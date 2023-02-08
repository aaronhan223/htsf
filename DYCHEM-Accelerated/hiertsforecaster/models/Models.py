from collections import OrderedDict
from hiertsforecaster.models.quantile import Qunatile_Network
from hiertsforecaster.models.gating_network import Gating_Network
from hiertsforecaster.models.train_LSTNet import train_lstn
from hiertsforecaster.models.train_deepar import train_deepar
from hiertsforecaster.models.deepar import Net
from hiertsforecaster.preprocess import utils
from hiertsforecaster.preprocess import train_config
import numpy as np
import pmdarima as pm
from pmdarima.arima import StepwiseContext
import pandas as pd
from pydlm import dlm, trend, seasonality
from prophet import Prophet
import concurrent.futures
import os
import torch
import torch.optim as optim
import logging
import pickle
from time import perf_counter
import json
from prophet.serialize import model_to_json
import pdb

p = 5
logger = logging.getLogger('MECATS.Models')


class PretrainExpts:
    '''
    Pre-train each model in the model bank offline using training data.
    '''
    def __init__(self, data, name, cuda, plot, experts):
        self.cuda = cuda
        self.name = name
        self.fbprophet = Prophet()
        self.plot = plot
        self.experts = experts
        self.config = train_config
        train_split = 1 - train_config.MecatsParams.valid_ratio
        self.val = train_config.DeeparParams().val
        assert train_config.MecatsParams.valid_ratio >= self.val
        self.tr = train_split - self.val
        self.data = data.dropna() #
        length = len(self.data)
        if train_config.MecatsParams.metric_mode:
            self.train = self.data[:int(train_split * (length - train_config.MecatsParams.pred_step))]
        else:
            self.train = self.data[:int(train_split * length)]
        self.auto_arima_params = train_config.AutoArimaParams
        Trend = trend(degree=train_config.PydlmParams().degree_linear, discount=train_config.PydlmParams().discount_linear, name='trend', w=train_config.PydlmParams().w_linear)
        seasonal_yearly = seasonality(period=train_config.PydlmParams().seasonality_period, discount=train_config.PydlmParams().discount_seasonal, name='weeklyY', w=train_config.PydlmParams().w_seasonal)
        seasonal_quarterly = seasonality(period=int(train_config.PydlmParams().seasonality_period / 4), discount=train_config.PydlmParams().discount_seasonal, name='weeklyQ', w=train_config.PydlmParams().w_seasonal / 4)
        self.dlm = dlm(self.train.to_list()) + Trend + seasonal_yearly + seasonal_quarterly
        self.deepar_train_range = range(train_config.DeeparParams().train_window, int(length * self.tr))
        self.deepar_valid_range = range(int(length * self.tr), int(length * train_split))
        self.deepar = Net(train_config.DeeparParams())
        self.lstnet_params = train_config.LstNetParams()

    def pretrain(self, name, realm_id):

        if 'auto_arima' in self.experts:
            try:
                logger.info('Training Auto-Arima Models...')
                arima_d = pm.arima.ndiffs(self.train)
                with StepwiseContext(max_dur=self.auto_arima_params.max_dur):
                    self.auto_arima_model = pm.auto_arima(self.train, seasonal=self.auto_arima_params.seasonal, m=self.auto_arima_params.m, stepwise=True, d=arima_d)
                with open(f'./hiertsforecaster/save/{realm_id}/arima_{name}.pkl', 'wb') as pkl:
                    pickle.dump(self.auto_arima_model, pkl)
            except:
                logger.info('Auto-arima failed due to data issue, switching to average prediction.')
                np.savez(f'./hiertsforecaster/save/{realm_id}/average_{name}.npz', average=np.average(self.train.to_numpy()))

        if 'average' in self.experts:
            logger.info('Computing Training Set Average...')
            np.savez(f'./hiertsforecaster/save/{realm_id}/average_{name}.npz', average=np.average(self.train.to_numpy()))

        if 'deepar' in self.experts:
            logger.info('Training DeepAR Models...')
            train_deepar(model=self.deepar, data=self.data, params=self.deepar_params, train_range=self.deepar_train_range, 
                         valid_range=self.deepar_valid_range, config=self.config, plot=self.plot)
            logger.info('Restoring DeepAR parameters...')
            optimizer = optim.Adam(self.deepar.parameters(), lr=self.deepar_params.learning_rate)
            utils.load_checkpoint(os.path.join(f'./hiertsforecaster/save/{realm_id}/deepar', 'best.pth.tar'), self.deepar, optimizer)

        if 'fbprophet' in self.experts:
            full_data = pd.DataFrame(data={'ds': self.train.index, 'y': self.train.values})
            logger.info('Training FB Prophet Models...')
            m1 = self.fbprophet.fit(full_data)
            with open(f'./hiertsforecaster/save/{realm_id}/fbprophet_{name}.json', 'w') as fout:
                json.dump(model_to_json(m1), fout)

        if 'lstnet' in self.experts:
            logger.info('Training LSTNet Models...')
            train_lstn(data=self.data, lstn_params=self.lstnet_params, tr=self.tr, val=self.val,
                       cuda=self.cuda, verbose=True, unit_test=self.config.MecatsParams.unit_test)

        if 'pydlm' in self.experts:
            try:
                logger.info('Training PyDLM Models...')
                self.dlm.fit()
                with open(f'./hiertsforecaster/save/{realm_id}/pydlm_{name}.pkl', 'wb') as pkl:
                    pickle.dump(self.dlm, pkl)
            except:
                logger.info('PyDLM failed due to data issue, switching to average prediction.')
                np.savez(f'./hiertsforecaster/save/{realm_id}/average_{name}.npz', average=np.average(self.train.to_numpy()))

def Vertex_PreTrain(name, data, experts, plot, cuda, realm_id):
    model = PretrainExpts(data, name, cuda, plot, experts)
    model.pretrain(name, realm_id)
    return f'Finished pretraining at vertex {name}.'


def get_moe_optimizers(hier_ts_data, cuda, plot, experts, realm_id):
    '''
    Return dictionary of offline-trained MoE models, gating networks, as well as optimizers for the gating network.
    '''
    gns, optimizers = OrderedDict(), OrderedDict()
    if train_config.MecatsParams.quantile:
        q_nets, q_optimizers = OrderedDict(), OrderedDict()

    vertex_list = hier_ts_data.node_list
    with concurrent.futures.ProcessPoolExecutor() as executer:
        results = [executer.submit(Vertex_PreTrain, vertex, hier_ts_data.ts_data[vertex], experts, plot, cuda, realm_id) for vertex in vertex_list]
        for f in concurrent.futures.as_completed(results):
            logger.info(f.result())
    for name in vertex_list:
        logger.info('Preparing model for node {}'.format(name))
        if len(hier_ts_data.ts_data[name].shape) == 1:
            input_dim = 1
        else:
            input_dim = hier_ts_data.ts_data[name].shape[1]

        gating_params = train_config.GatingNetParams()
        gn = Gating_Network(input_dim, gating_params, experts)
        gns[name] = gn
        optimizers[name] = torch.optim.Adam(gn.parameters(), lr=gating_params.learning_rate)
        if train_config.MecatsParams.quantile:
            net = Qunatile_Network(train_config.QuantileNetParams)
            Optim = torch.optim.Adam(net.parameters(), lr=train_config.QuantileNetParams.learning_rate)
            q_nets[name], q_optimizers[name] = net, Optim
    if train_config.MecatsParams.quantile:
        return gns, optimizers, q_nets, q_optimizers
    return gns, optimizers