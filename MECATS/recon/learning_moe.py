'''
Name: learning_moe.py
Learning mixture of multiple time series forecasting models in offline setting.
'''
import pandas as pd
import torch
import torch.nn as nn
import pdb
import numpy as np
import torch.nn.functional as F
import os
import math
import logging
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
import torch_dct as dct
from fbprophet import Prophet
from pydlm import dlm, trend
import pmdarima as pm
from preprocess import utils
from preprocess.hierarchical import TreeNodes
from models import LSTNet, Optim
from models.quantile import QuantileLoss
from models.train_LSTNet import fit_and_pred
from models.train_deepar import train_and_evaluate
from models.deepar import Net, loss_fn
from models.bocpd_model import GaussianUnknownMean, is_changepoint, bocd
from recon.MinT import mint_py

logger = logging.getLogger('MECATS.learning_moe')


class Gating_Network(nn.Module):
    '''
    Gating network with recurrent and fully connected layer.
    '''
    def __init__(self, input_dim, params):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, params.recurrent_hidden_dim, params.layer_dim, batch_first=True)
        self.l1 = nn.Linear(params.batch_size, params.linear_hidden_dim)
        self.nonlinear = nn.Tanh()
        self.l2 = nn.Linear(params.linear_hidden_dim, 5)

    def forward(self, data):
        o1 = self.lstm(data)[0][:, -1, :]
        o2 = self.l1(torch.t(o1))
        o3 = self.nonlinear(o2)
        o4 = self.l2(o3)
        return F.softmax(o4, dim=1)


class Qunatile_Network(nn.Module):
    '''
    Uncertainty wrapper with densely connected layer.
    '''
    def __init__(self, params):
        super().__init__()
        self.dense_net = torch.nn.Sequential()
        self.d = params.d
        self.type = params.type
        layer_dims = params.layer_dims
        input_dim = params.window + 1
        for i in range(params.num_layers):
            if i == 0:
                self.dense_net.add_module('dense_{}'.format(i), nn.Linear(input_dim, layer_dims[i]))
            else:
                self.dense_net.add_module('dense_{}'.format(i), nn.Linear(layer_dims[i - 1], layer_dims[i]))
            if i < params.num_layers - 1:
                self.dense_net.add_module('nonlinear_{}'.format(i), nn.ReLU())
        model_set = [self.dense_net for _ in range(self.d)]
        self.model_set = nn.ModuleList(model_set)
        if torch.cuda.is_available():
            self.device = 'cuda:{}'.format(torch.cuda.current_device())
        else:
            self.device = 'cpu'
        self.softplus = nn.Softplus()
        self.t_k = 0.5 * torch.cos(math.pi * (torch.arange(self.d) + 0.5) / self.d) + 0.5
        self.quantiles = params.quantiles

    def forward(self, data, pred):
        C_0, C_k = self.get_coeff(data, self.d, self.model_set, pred)
        quantile_output = torch.empty(len(pred), len(self.quantiles)).to(self.device)
        for i, tau in enumerate(self.quantiles):
            quantile_output[:, i] = self.eval_cheb(tau, C_0, C_k, self.d)
        return quantile_output

    def get_coeff(self, X, d, nets, pred):
        t_k = self.t_k.repeat(math.ceil(X.shape[0] / d) + 1)
        t_ks = [t_k[i: i + X.shape[0]].unsqueeze(1).to(self.device) for i in range(d)]
        pred = torch.tensor(pred).to(self.device)
        full_output = torch.empty(X.shape[0], d).to(self.device)
        C_k = torch.empty(X.shape[0], d).to(self.device)
        for i, net in enumerate(nets):
            full_input = torch.cat((t_ks[i], X), 1)
            logit = net(full_input)[:, 0]
            logit = self.softplus(logit + 1e-5) + 1e-3
            full_output[:, i] = logit

        c_k = dct.dct(full_output)
        for i in range(1, d - 1):
            C_k[:, i] = (c_k[:, i - 1] - c_k[:, i + 1]) / (4 * i)
        C_k[:, d - 1] = c_k[:, d - 2] / 4 * (d - 1)
        if self.type == 'median':
            indices = torch.tensor(range(2, d, 2))
            C_0 = 2 * pred - 2 * torch.sum(C_k[:, indices] * torch.pow(-1, indices / 2).repeat(X.shape[0], 1).to(self.device), dim=1)
        elif self.type == 'mean':
            indices = list(range(1, d, 2))
            C_0 = 2 * pred - 2 * torch.sum(C_k[:, indices] / (torch.pow(torch.tensor(indices).to(self.device), 2) - 4).repeat(X.shape[0], 1), dim=1)
        else:
            raise ValueError('Point prediction must be mean or median.')
        return C_0, C_k

    def eval_cheb(self, tau, C_0, C_k, d):
        T_1, T_2, T_3 = 0, 0, 0
        sigma = 2 * tau - 1
        for i in range(d - 1, 0, -1):
            T_3 = T_1
            T_1 = 2 * sigma * T_1 - T_2 + C_k[:, i]
            T_2 = T_3
        return sigma * T_1 - T_2 + 0.5 * C_0


class MoE_Learner:
    '''
    MECATS model class, contains online/offline training and prediction functions.
    '''
    def __init__(self, models, optimizers, gns, params, data, date, nodes, node_list, dataset, cuda, **kwargs) -> None:
        self.models = models
        self.optimizers = optimizers
        self.gns = gns
        self.data = data
        self.date = date
        self.nodes = nodes
        self.node_list = node_list
        self.val = params['sharq'].valid_split
        self.te = params['sharq'].test_split
        self.params_set = params
        self.params = params['gating_network']
        self.window = self.params.window
        self.l = len(nodes) + 1
        self.preds = {}
        self.weights = {}
        self.dataset = dataset
        self.cuda = cuda
        if 'q_nets' in kwargs:
            self.q_nets = kwargs.pop('q_nets')
            self.q_optimizers = kwargs.pop('q_optimizers')
            self.q_net_params = params['quantile_net']
            self.q_net_params.window = self.window
        length = self.data.shape[0]
        self.valid_range = range(int((1 - self.te - self.val) * length), int((1 - self.te) * length))
        self.test_range = range(int((1 - self.te) * length), length)
        valid_start = range(self.valid_range[0] - params['sharq'].FORECAST_HORIZON, self.valid_range[0])
        if cuda:
            self.device = 'cuda:{}'.format(torch.cuda.current_device())
        else:
            self.device = 'cpu'
        logger.info('<-------------Model Bank Prediction------------->')
        for node in node_list:
            logger.info('Predict on validation set for node {}'.format(node))
            self.preds[node] = torch.tensor(self.predict(models[node], data[node], self.valid_range, valid_start), requires_grad=False).to(self.device)
 
    def fit_point_recon(self):
        logger.info('<-------------Train Gating Network------------->')
        node_list = TreeNodes(self.nodes).nodes_by_level(self.l)

        for name in node_list:
            self.fit(self.optimizers, self.gns, name)

        if len(self.nodes) > 0:
            logger.info('Train Aggregated Level Gating Network')
            for i in range(self.l - 1, 0, -1):
                nodes = TreeNodes(self.nodes).nodes_by_level(i)
                for name in nodes:
                    childs = TreeNodes(self.nodes, name=name).get_child()
                    self.fit(self.optimizers, self.gns, name, childs=childs)
            
            if self.params_set['sharq'].QUANTILE and self.params_set['sharq'].RECON == 'sharq':
                logger.info('<-------------Train Unvertainty Wrapper------------->')
                quantiles = self.q_net_params.quantiles
                qloss = QuantileLoss(quantiles).to(self.device)
                q_nets = OrderedDict()

                for node in self.node_list:
                    logger.info('Uncertainty wrapper for node {}'.format(node))
                    preds = self.preds[node].cpu().numpy()
                    weights = np.expand_dims(self.weights[node], 0)
                    S_pred = np.matmul(weights, preds)[0]

                    net, data, optimizer = self.q_nets[node], self.data[node], self.q_optimizers[node]
                    set_range = range(self.q_net_params.window, int((1 - self.te - self.val) * len(data)))
                    q_net_input = torch.squeeze(prepare_data(set_range, data, self.q_net_params.window))
                    dataset = utils.QuantileDataset(q_net_input)
                    train_loader = DataLoader(dataset, batch_size=S_pred.shape[0], num_workers=4, shuffle=False, drop_last=True)
                    
                    y = torch.tensor(data[int((1 - self.te - self.val) * len(data)): int((1 - self.te) * len(data))].to_list(), dtype=torch.float32)
                    y = torch.unsqueeze(y, 1).to(self.device)
                    for epoch in range(self.q_net_params.num_epochs):
                        total_loss = 0.
                        for train_batch in train_loader:
                            train_batch = train_batch.to(self.device)
                            quantile_output = net(train_batch, S_pred)
                            loss = qloss(quantile_output, y)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.data
                        if epoch % 100 == 0:
                            logger.info('Epoch [{}] | Node {} | Quantile Loss {}'.format(epoch, node, total_loss))
                    q_nets[node] = net
                self.q_nets = q_nets

        models = OrderedDict()
        logger.info('<-------------Update Model Bank------------->')
        for name in self.node_list:
            logger.info('Updating for node {}'.format(name))
            model_dict = self.update_model_bank(self.models[name], self.data[name], name)
            models[name] = model_dict
        self.models = models

    def fit(self, optimizers, gns, name, **kwargs):
        logger.info('Train gating network at node {}'.format(name))
        MSE, regularization = nn.MSELoss(), False
        gn, optimizer, data = gns[name].to(self.device), optimizers[name], self.data[name]
        # gating network input data is the same as training data range
        set_range = range(self.window, int((1 - self.te - self.val) * len(data)))
        gating_params = self.params_set['gating_network']

        if 'childs' in kwargs and self.params_set['sharq'].RECON == 'sharq':
            childs = kwargs.pop('childs')
            gn_input = prepare_data(set_range, self.data[childs], self.window)
            regularization = True
        else:
            gn_input = prepare_data(set_range, data, self.window)
        
        # Don't set batch size too large (>128) for short time seris.
        dataset = utils.GatingDataset(gn_input)
        train_loader = DataLoader(dataset, batch_size=gating_params.batch_size, num_workers=4, shuffle=False, drop_last=True)
        preds = self.preds[name]
        y = torch.tensor(data[int((1 - self.te - self.val) * len(data)): int((1 - self.te) * len(data))].to_list(), dtype=torch.float32)
        y = torch.unsqueeze(y, 0).to(self.device)
        
        self.all_weights = torch.zeros((self.params.num_epochs, 5))
        for epoch in range(self.params.num_epochs):
            total_loss, count = 0., 0
            epoch_weights = torch.zeros((1, 5), dtype=torch.float32, requires_grad=False)
            for train_batch in train_loader:
                if regularization:
                    batch = torch.unsqueeze(train_batch[:, :, 0], 2)
                else:
                    batch = train_batch
                
                batch = batch.to(self.device)
                weights = gn(batch)
                weighted_preds = torch.matmul(weights, preds)
                optimizer.zero_grad()

                if regularization:
                    loss = MSE(weighted_preds, y) + self.get_reg(childs, gns, train_batch, self.device, preds.shape[1], self.params.Lambda)
                else:
                    loss = MSE(weighted_preds, y)

                loss.backward()
                optimizer.step()
                total_loss += loss.data
                epoch_weights += weights.cpu().data
                count += 1

            self.all_weights[epoch] = (epoch_weights / count)[0]
            if epoch % 100 == 0:
                logger.info('Epoch [{}] | Node {} | MSE {}'.format(epoch, name, total_loss))
        
        level = TreeNodes(self.nodes, name=name).get_levels()
        self.weights[name] = self.all_weights[-1, :].numpy()
        legend = True if name == '0' else False
        utils.plot_weights(self.all_weights.numpy(), self.params.num_epochs, level, name, self.dataset, self.params_set['sharq'].RECON, legend)
    
    def online_update(self, params):
        model_dict, gn, optimizer = self.models['0'], self.gns['0'], self.optimizers['0']
        test = self.data['0'].iloc[self.test_range]
        # larger learning rate since fewer epochs...
        # self.params.learning_rate = self.params.learning_rate * 10
        MSE = nn.MSELoss()
        h = self.params_set['sharq'].FORECAST_HORIZON
        test_length, m = len(test), 0
        length = len(self.data['0'])
        test_start = range(self.test_range[m] - h, self.test_range[m])
        train_loader = self.get_data_loader(m)
        step_range = range(params.batch_size, test_length + 1, params.batch_size)

        step_weights = torch.zeros((len(step_range) * params.epoch, 5))
        step_loss = torch.zeros(len(step_range))
        log_R = -np.inf * np.ones((len(step_range) + 1, len(step_range) + 1))
        log_R[0, 0] = 0
        p_mean, p_var = np.empty(len(step_range)), np.empty(len(step_range))
        cnt, ave_weights = np.inf, 0.2 * torch.ones((1, 5)).to(self.device)
        log_message = np.array([0])
        log_H = np.log(params.hazard)
        log_1mH = np.log(1 - params.hazard)
        cp_model = GaussianUnknownMean(params.prior_mean, params.prior_var, params.observe_var)
        
        for i, n in enumerate(step_range):
            preds = torch.tensor(self.predict(model_dict, self.data['0'], self.test_range[m: m + params.batch_size], test_start), requires_grad=False).to(self.device)
            y = torch.tensor(test[m: n].to_list(), dtype=torch.float32)
            y = torch.unsqueeze(y, 0).to(self.device)
            all_weights = torch.zeros((params.epoch, 5))
            
            model_weights = torch.zeros((1, 5))
            for batch in train_loader:
                model_weights = gn(batch.to(self.device))
            if params.enable_cp and self.dataset == 'sim_cp':
                cp = is_changepoint(float(y - torch.matmul(model_weights, preds)), i + 1, cp_model, log_R, log_message, log_H, log_1mH)
                if cp:
                    cnt = 0
                model_weights = (1 - math.exp(-cnt / params.tau)) * model_weights + math.exp(-cnt / params.tau) * ave_weights
                if math.exp(-cnt / params.tau) < 0.1:
                    cnt = np.inf
                else:
                    cnt += 1

            elif params.enable_cp and self.dataset == 'EIA':
                p_mean, p_var = bocd(float(y - torch.matmul(model_weights, preds)), i + 1, cp_model, log_R, log_message, log_H, log_1mH, p_mean, p_var)
                p_mean[i] += float(torch.matmul(model_weights, preds))

            plot_loss = MSE(torch.matmul(model_weights, preds), y).cpu().detach()
            step_loss[i] = plot_loss

            for epoch in range(params.epoch):
                total_loss, count = 0., 0
                epoch_weights = torch.zeros((1, 5), dtype=torch.float32, requires_grad=False)
                for batch in train_loader:
                    batch = batch.to(self.device)
                    weights = gn(batch)
                    weighted_preds = torch.matmul(weights, preds)
                    optimizer.zero_grad()
                    loss = MSE(weighted_preds, y)

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data
                    epoch_weights += weights.cpu().data
                    count += 1
                all_weights[epoch] = (epoch_weights / count)[0]
                logger.info('Step [{}] | Epoch {} | MSE {}'.format(n, epoch, total_loss))
            
            step_weights[i * params.epoch: (i + 1) * params.epoch] = all_weights
            if n < test_length:
                # new_obs = test[m: n].to_list()
                m = n
                test_start = range(self.test_range[m] - h, self.test_range[m])
                train_loader = self.get_data_loader(m)

                # online update non-deep learning models
                # logger.info('Online update models at step {}'.format(i))
                # model_dict['pydlm'].append(new_obs, component='main')
                # model_dict['pydlm'].fitForwardFilter(useRollingWindow=True, windowLength=self.params_set['pydlm'].window)
                # model_dict['auto_arima'] = model_dict['auto_arima'].update(new_obs)
                # date = self.date[:int((1 - self.te) * length) + i]
                # data = self.data['0'][:int((1 - self.te) * length) + i]
                # full_data = pd.concat([date, data], axis=1)
                # full_data.rename(columns={'0': 'y'}, inplace=True)
                # model_dict['fbprophet'] = Prophet().fit(full_data, init=stan_init(model_dict['fbprophet']))

        if not params.enable_cp and self.dataset == 'sim_cp':
            utils.plot_weights_cp(self.all_weights.numpy(), step_weights.numpy(), self.params.num_epochs, len(step_range) * params.epoch, self.dataset)
        elif params.enable_cp and self.dataset == 'EIA':
            utils.plot_rl_cp(len(step_range), test, np.exp(log_R), p_mean, p_var)
        return step_loss.numpy(), len(step_range)

    def get_data_loader(self, m):
        gating_range = range(self.test_range[m] - self.params_set['sharq'].FORECAST_HORIZON - self.params_set['gating_network'].batch_size + 1, self.test_range[m])
        gn_input = prepare_data(gating_range, self.data['0'], self.window)
        dataset = utils.GatingDataset(gn_input)
        train_loader = DataLoader(dataset, batch_size=self.params_set['gating_network'].batch_size, num_workers=4, shuffle=False, drop_last=True)
        return train_loader

    def predict(self, model_bank, data, set_range, pred_start):
        '''
        Model bank prediction on the specified validation set.
        '''
        dlm = model_bank['pydlm']
        fbprophet = model_bank['fbprophet']
        arima = model_bank['auto_arima']
        lstnet = model_bank['lstnet']
        deepar = model_bank['deepar']
        deepar_params = model_bank['deepar_params']
        lstnet_params = model_bank['lstn_params']
        n = len(set_range)

        logger.info('PyDLM prediction.')
        preds_pydlm = np.array(dlm.predictN(N=n)[0], dtype=np.float32)
        logger.info('Auto-ARIMA prediction.')
        preds_auto_arima = np.array(arima.predict(n_periods=n, return_conf_int=False), dtype=np.float32)
        logger.info('Prophet prediction.')
        future_df = fbprophet.make_future_dataframe(periods=n)
        preds_fbprophet = np.array(fbprophet.predict(future_df)['yhat'][-n:].to_list(), dtype=np.float32)
        logger.info('LSTNet prediction.')
        preds_lstnet = pred_lstn(data, lstnet_params.h, pred_start, lstnet, lstnet_params.window, n, self.cuda).squeeze(1).cpu().detach().numpy()
        logger.info('DeepAR prediction.')
        preds_deepar = pred_deepar(data, set_range, deepar, deepar_params).squeeze(0)
        preds = np.stack((preds_pydlm, preds_auto_arima, preds_fbprophet, preds_lstnet, preds_deepar))
        return preds

    def get_reg(self, childs, gns, train_batch, device, h, Lambda=.5):
        reg = torch.zeros((1, h), requires_grad=False, device=device)
        for i, node in enumerate(childs):
            weights = gns[node](torch.unsqueeze(train_batch[:, :, i], 2).to(device))
            preds = self.preds[node]
            if i == 0:
                reg += torch.matmul(weights, preds)
            else:
                reg -= torch.matmul(weights, preds)
        return Lambda * torch.sum(torch.pow(reg, 2))

    def test_pred(self):
        test = self.data.iloc[self.test_range]
        pred_dict, quantile_dict = {}, {}
        method = self.params_set['sharq'].RECON
        test_start = range(self.test_range[0] - self.params_set['sharq'].FORECAST_HORIZON, self.test_range[0])
        if method == 'sharq':
            for node in self.node_list:
                logger.info('Predict on test set for node {}'.format(node))
                preds = self.predict(self.models[node], self.data[node], self.test_range, test_start)
                weights = np.expand_dims(self.weights[node], 0)
                combined_pred = np.matmul(weights, preds)[0]
                pred_dict[node] = combined_pred
                if self.params_set['sharq'].QUANTILE:
                    net, data = self.q_nets[node], self.data[node]
                    # here we use the most recent records for quantile predictions
                    pred_length, test_start_point = combined_pred.shape[0], int((1 - self.te) * len(data))
                    set_range = range(test_start_point - pred_length, test_start_point)
                    q_net_input = torch.squeeze(prepare_data(set_range, data, self.q_net_params.window))
                    dataset = utils.QuantileDataset(q_net_input)
                    test_loader = DataLoader(dataset, batch_size=len(dataset), num_workers=4, shuffle=False)
                    for batch in test_loader:
                        batch = batch.to(self.device)
                        quantile_dict[node] = net(batch, combined_pred).cpu().detach().numpy()
                    level = TreeNodes(self.nodes, name=node).get_levels()
                    legend = False if int(node) > 6 else True
                    utils.forecast_plot(np.array(self.data.iloc[self.valid_range][node]), combined_pred, 
                                        np.array(test[node]), quantile_dict[node], level, node, self.dataset, legend)
        else:
            output, result = np.zeros((test.shape)), np.zeros((test.shape))
            for i, node in enumerate(self.node_list):
                logger.info('Predict on test set for node {}'.format(node))
                preds = self.predict(self.models[node], self.data[node], self.test_range, test_start)
                weights = np.expand_dims(self.weights[node], 0)
                output[:, i] = np.matmul(weights, preds)[0]
            logger.info('MinT reconciliation for MoE prediction...')
            for j in range(output.shape[0]):
                row_pred = output[j, :]
                result[j, :] = mint_py(TreeNodes(self.nodes).get_s_matrix(), row_pred, method)
            for i, node in enumerate(self.node_list):
                pred_dict[node] = result[:, i]
        return pred_dict, test, quantile_dict

    def update_model_bank(self, model_bank, data, name):
        '''
        Update original model bank with observations used to train gating network.
        Use this function after training the gating network.
        '''
        dlm = model_bank['pydlm']
        fbprophet = model_bank['fbprophet']
        arima = model_bank['auto_arima']
        lstnet = model_bank['lstnet']
        deepar = model_bank['deepar']
        deepar_params = model_bank['deepar_params']
        lstnet_params = model_bank['lstn_params']
        pydlm_params = self.params_set['pydlm']

        length = len(data)
        train_valid_range = range(deepar_params.train_window, int((1 - self.te) * length))
        valid_range = range(int((1 - self.te - self.val) * length), int((1 - self.te) * length))
        data_valid = data.iloc[valid_range]

        logger.info('Updating Auto-Arima Models...')
        arima = arima.update(data_valid)

        logger.info('Updating PyDLM Models...')
        dlm.append(data_valid.to_list(), component='main')
        dlm.fitForwardFilter(useRollingWindow=True, windowLength=pydlm_params.window)

        logger.info('Updating Prophet Models...')
        date = self.date[:int((1 - self.te) * length)]
        train_valid_data = data[:int((1 - self.te) * length)]
        full_data = pd.concat([date, train_valid_data], axis=1)
        full_data.rename(columns={name: 'y'}, inplace=True)
        m2 = Prophet().fit(full_data, init=stan_init(fbprophet))

        logger.info('Updating DeepAR Models...')
        train_deepar(model=deepar, data=data, params=deepar_params, train_range=train_valid_range, 
                     valid_range=valid_range)

        logger.info('Updating LSTNet Models...')
        train_lstn(data=data, lstn_params=lstnet_params, tr=(1 - self.te), val=0., 
                   cuda=self.cuda, method=self.params_set['sharq'].RECON, verbose=True, init=lstnet)

        logger.info('Load updated LSTNet model...')
        with open(os.path.join(lstnet_params.model_dir, '{}_{}_LSTNet.pt'.format(self.params_set['sharq'].RECON, lstnet_params.dataset)), 'rb') as f:
            LSTNet = torch.load(f)

        logger.info('Restoring updated DeepAR parameters...')
        optimizer = optim.Adam(deepar.parameters(), lr=deepar_params.learning_rate)
        utils.load_checkpoint(os.path.join(deepar_params.model_dir, '{}_{}_best.pth.tar'.format(deepar_params.dataset, deepar_params.method)), deepar, optimizer)

        return {'auto_arima': arima, 'fbprophet': m2, 'pydlm': dlm, 'lstnet': LSTNet, 
                'deepar': deepar, 'deepar_params': deepar_params, 'lstn_params': lstnet_params}


class fitModel:
    '''
    Pre-train each model in the model bank offline using training data.
    '''
    def __init__(self, data, date, params, name, dataset, cuda):
        self.cuda = cuda
        self.name = name
        self.fbprophet = Prophet()
        self.sharq_params = params['sharq']
        train_split = self.sharq_params.train_split
        self.val = 0.1
        assert params['sharq'].valid_split >= self.val
        assert params['sharq'].valid_split >= params['sharq'].test_split
        self.tr = train_split - self.val
        self.train = pd.Series(data[:int((self.tr + self.val) * len(data))], name='y')
        self.date_train = date[:int((self.tr + self.val) * len(date))]
        self.data = data
        self.pydlm_params = params['pydlm']
        Trend = trend(degree=self.pydlm_params.trend_degree, discount=self.pydlm_params.discount, name='trend', w=self.pydlm_params.prior_cov)
        self.dlm = dlm(self.train.to_list()) + Trend

        self.deepar_params = params['deepar']
        self.deepar_params.relative_metrics = False
        self.deepar_params.dataset = dataset
        self.deepar_params.method = self.sharq_params.RECON
        self.deepar_params.train_window = int(1.2 * params['sharq'].valid_split * len(data))
        self.deepar_params.valid_window = self.deepar_params.train_window
        self.deepar_params.predict_steps = int(self.val * len(data))
        self.deepar_params.predict_start = self.deepar_params.valid_window - self.deepar_params.predict_steps
        self.deepar_params.test_predict_start = self.deepar_params.predict_start

        self.lstnet_params = params['lstnet']
        self.lstnet_params.dataset = dataset
        self.lstnet_params.window = self.deepar_params.train_window
        self.pydlm_params.window = self.deepar_params.train_window
        params['gating_network'].window = self.deepar_params.train_window
        if 'quantile_net' in params.keys():
            self.q_net_params = params['quantile_net']
            self.q_net_params.window = self.deepar_params.train_window
        self.train_range = range(self.deepar_params.train_window, int(len(self.data) * self.tr))
        self.valid_range = range(int(len(self.data) * self.tr), int(len(self.data) * (self.tr + self.val)))

        if cuda:
            self.deepar_params.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
            self.deepar = Net(self.deepar_params).to(self.deepar_params.device)
        else:
            self.deepar = Net(self.deepar_params)
            self.deepar_params.device = torch.device('cpu')

    def pretrain(self):
        logger.info('Training Auto-Arima Models...')
        self.auto_arima_model = pm.auto_arima(self.train, seasonal=True, m=12)

        full_data = pd.concat([self.date_train, self.train], axis=1)
        if self.name is not None:
            full_data.rename(columns={self.name: 'y'}, inplace=True)
        logger.info('Training FB Prophet Models...')
        m1 = self.fbprophet.fit(full_data)

        logger.info('Training PyDLM Models...')
        self.dlm.fitForwardFilter(useRollingWindow=True, windowLength=self.pydlm_params.window)

        logger.info('Training DeepAR Models...')
        train_deepar(model=self.deepar, data=self.data, params=self.deepar_params,
                     train_range=self.train_range, valid_range=self.valid_range)

        logger.info('Training LSTNet Models...')
        train_lstn(data=self.data, lstn_params=self.lstnet_params, tr=self.tr, val=self.val,
                   cuda=self.cuda, method=self.sharq_params.RECON, verbose=True)

        logger.info('Load saved LSTNet model...')
        with open(os.path.join(self.lstnet_params.model_dir, '{}_{}_LSTNet.pt'.format(self.sharq_params.RECON, self.lstnet_params.dataset)), 'rb') as f:
            LSTNet = torch.load(f)

        logger.info('Resotring DeepAR parameters...')
        optimizer = optim.Adam(self.deepar.parameters(), lr=self.deepar_params.learning_rate)
        utils.load_checkpoint(os.path.join(self.deepar_params.model_dir, '{}_{}_best.pth.tar'.format(self.deepar_params.dataset, self.deepar_params.method)), self.deepar, optimizer)
        return {'auto_arima': self.auto_arima_model, 'fbprophet': m1, 'pydlm': self.dlm, 'lstnet': LSTNet, 
                'deepar': self.deepar, 'deepar_params': self.deepar_params, 'lstn_params': self.lstnet_params}


def train_lstn(data, lstn_params, tr, val, cuda, method, verbose, **kwargs):
    Data = utils.Data_utility(tr, val, cuda, lstn_params.h, lstn_params.window, data, method, normalize=lstn_params.normalize)
    data_dim = 1 if len(data.shape) == 1 else data.shape[1]
    if 'init' in kwargs:
        model = kwargs.pop('init')
    else:
        model = LSTNet.Model(data_dim, lstn_params.window)
    criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    if cuda:
        model.cuda()
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
    optim = Optim.Optim(model.parameters(), lstn_params.optimizer, lstn_params.learning_rate, 10.)
    fit_and_pred(Data, model, lstn_params.h, lstn_params.num_epoch, lstn_params.batch_size, optim, method,
                 criterion, evaluateL2, evaluateL1, lstn_params.model_dir, lstn_params.dataset, None, verbose, cuda)


def train_deepar(model, data, params, train_range, valid_range):
    train_set = utils.TrainDataset(data, params.train_window, train_range)
    train_loader = DataLoader(train_set, batch_size=params.batch_size, num_workers=4)
    valid_set = utils.ValidDataset(data, params.valid_window, valid_range)
    valid_loader = DataLoader(valid_set, batch_size=params.batch_size, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    print('Starting training for {} epoch(s)'.format(params.num_epochs))
    train_and_evaluate(model, train_loader, valid_loader, optimizer, loss_fn, params)


def pred_lstn(data, h, val, model, window, n, cuda):
    preds = torch.zeros((n, 1))
    X = prepare_data(val, data, window, h)
    X = X.cuda() if cuda else X
    # use prediction from previous step as inputs for next step
    for i in range(n):
        output = model(X).detach()[0]
        preds[i, :] = output
        X = torch.cat((X[:, h:, :], output.unsqueeze(1).unsqueeze(2)), 1)
    return preds


def pred_deepar(data, val, model, params):
    params.predict_steps = len(val)
    params.predict_start = params.valid_window - params.predict_steps
    params.test_predict_start = params.valid_window - params.predict_steps
    valid_set = utils.ValidDataset(data, params.valid_window, val)

    input, id, v, label = valid_set[len(valid_set) - 1]
    input = torch.unsqueeze(torch.tensor(input, dtype=torch.float32).to(params.device), 1)
    id = torch.tensor([id]).unsqueeze(0).to(params.device)
    v = torch.unsqueeze(torch.tensor(v, dtype=torch.float32).to(params.device), 0)
    label = torch.unsqueeze(torch.tensor(label, dtype=torch.float32).to(params.device), 0)
    input_mu = torch.zeros(1, params.test_predict_start, device=params.device) # scaled
    input_sigma = torch.zeros(1, params.test_predict_start, device=params.device) # scaled
    hidden = model.init_hidden(1)
    cell = model.init_cell(1)

    # insample prediction
    for t in range(params.test_predict_start):
        mu, sigma, hidden, cell = model(input[t].unsqueeze(0), id, hidden, cell)
        input_mu[:, t] = v[:, 0] * mu + v[:, 1]
        input_sigma[:, t] = v[:, 0] * sigma

    sample_mu, _ = model.test(input, v, id, hidden, cell)
    return sample_mu.cpu().detach().numpy()


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


def stan_init(m):
    """
    Retrieve parameters from a trained Prophet model.
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res