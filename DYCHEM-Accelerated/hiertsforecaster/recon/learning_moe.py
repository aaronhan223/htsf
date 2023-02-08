'''
Name: learning_moe.py
Learning to combine heterogeneous forecasting models and perform model-free quantile estimation.
'''
import sys
import pandas as pd
from time import perf_counter, sleep
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import pdb
import numpy as np
import os
import math
import copy
import logging
import pickle
import json
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
from prophet import Prophet
from pydlm import dlm, trend, seasonality
import concurrent.futures
from hiertsforecaster.models.deepar import Net
from hiertsforecaster.preprocess import utils, train_config
from hiertsforecaster.preprocess.hierarchical import TreeNodes
from hiertsforecaster.models.quantile import QuantileLoss
from hiertsforecaster.models.train_LSTNet import train_lstn, pred_lstn
from hiertsforecaster.models.train_deepar import train_deepar, pred_deepar
from hiertsforecaster.models.bocpd_model import GaussianUnknownMean, is_changepoint
from prophet.serialize import model_from_json, model_to_json


logger = logging.getLogger('MECATS.learning_moe')


class MoE_Learner:
    '''
    MECATS model class, contains online/offline training and prediction functions.
    '''
    def __init__(self, optimizers, gns, hier_ts_data, cuda, plot, experts, realm_id, **kwargs) -> None:
        self.optimizers = optimizers
        self.gns = gns
        self.hier_ts_data = hier_ts_data
        self.data = hier_ts_data.ts_data.dropna() #
        self.nodes = hier_ts_data.nodes
        self.node_list = hier_ts_data.node_list
        self.config = train_config
        self.val = self.config.MecatsParams.valid_ratio
        self.te = self.config.MecatsParams.pred_step
        self.params = self.config.GatingNetParams()
        self.window = self.params.window
        self.l = len(hier_ts_data.nodes) + 1
        self.preds = {}
        self.weights = {}
        self.cuda = cuda
        self.plot = plot
        self.experts = experts
        self.realm_id = realm_id
        self.path = f'./hiertsforecaster/save/{self.realm_id}/'
        if 'q_nets' in kwargs:
            self.q_nets = kwargs.pop('q_nets')
            self.q_optimizers = kwargs.pop('q_optimizers')
            self.q_net_params = self.config.QuantileNetParams
        length = self.data.shape[0]
        if not self.config.MecatsParams.metric_mode:
            self.split_point = int((1 - self.val) * length)
            self.valid_range = range(self.split_point, length)
            self.test_range = range(length, length + self.te)
        else:
            self.split_point = int((1 - self.val) * (length - self.te))
            self.valid_range = range(self.split_point, length - self.te)
            self.test_range = range(length - self.te, length)
        if cuda:
            self.device = 'cuda:'
        else:
            self.device = 'cpu'
        self.valid_start = range(self.valid_range[0] - self.config.MecatsParams.horizon, self.valid_range[0])
        logger.info('<-------------Model Bank Prediction------------->')
        self._model_bank_pred()

    def _model_bank_pred(self):
        vertex_list = self.hier_ts_data.node_list
        with concurrent.futures.ProcessPoolExecutor() as executer:
            results = [executer.submit(predict, vertex, self.data[vertex], self.valid_range, self.valid_start, self.experts, self.cuda, self.path) for vertex in vertex_list]
            for f in concurrent.futures.as_completed(results):
                self.preds[f.result()[0]] = torch.tensor(f.result()[1], requires_grad=False)

    def fit_moe(self):
        self.train_gating_net()
        if self.config.MecatsParams.quantile:
            self.train_uncertainty_wrapper()
        self.update_all_experts()

    def train_gating_net(self):
        logger.info('<-------------Train Gating Network------------->')
        node_list = TreeNodes(self.nodes).nodes_by_level(self.l)

        logger.info('Train Bottom Level Gating Network')
        for name in node_list:
            start_time = perf_counter()
            mp.spawn(fit_gn, nprocs=self.config.ParallelParams.n_gpu, 
                     args=(self.data, self.params, self.config.ParallelParams, self.optimizers, self.gns, name, self.window, self.split_point,
                           self.device, self.experts, self.valid_range, self.preds,
                           self.plot, self.config.MecatsParams, self.nodes, self.path))
            print(f"Train GN time: {int(perf_counter() - start_time)}")
            self.load_and_remove_weights(self.path, name)
            
        if len(self.nodes) > 0:
            logger.info('Train Aggregated Level Gating Network')
            for i in range(self.l - 1, 0, -1):
                nodes = TreeNodes(self.nodes).nodes_by_level(i)
                for name in nodes:
                    childs = TreeNodes(self.nodes, name=name).get_child()
                    mp.spawn(fit_gn, nprocs=self.config.ParallelParams.n_gpu, 
                             args=(self.data, self.params, self.config.ParallelParams, self.optimizers, self.gns, name, self.window, self.split_point,
                                   self.device, self.experts, self.valid_range, self.preds, 
                                   self.plot, self.config.MecatsParams, self.nodes, self.path, childs))
                    self.load_and_remove_weights(self.path, name)

    def load_and_remove_weights(self, path, name):
        file_path = [path + 'gn_weights_p{}_n{}.npz'.format(i, name) for i in range(self.config.ParallelParams.n_gpu)]
        process_weights = [np.load(fp)['weights'] for fp in file_path]
        self.weights[name] = np.mean(np.vstack(process_weights), axis=0)
        for path in file_path:
            os.remove(path)

    def train_uncertainty_wrapper(self):
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
            lr_scheduler, early_stopping = utils.LRScheduler(optimizer), utils.EarlyStopping()
            set_range = range(self.q_net_params.window, self.split_point)
            q_net_input = torch.squeeze(utils.prepare_data(set_range, data, self.q_net_params.window))
            dataset = utils.QuantileDataset(q_net_input)
            train_loader = DataLoader(dataset, batch_size=S_pred.shape[0], num_workers=4, shuffle=False, drop_last=True)
            
            y = torch.tensor(data[self.valid_range].to_list(), dtype=torch.float32)
            y = torch.unsqueeze(y, 1).to(self.device)
            for epoch in range(self.q_net_params.num_epochs):
                epoch_loss, cnt = 0., 0
                for train_batch in train_loader:
                    train_batch = train_batch.to(self.device)
                    quantile_output = net(train_batch, S_pred)
                    loss = qloss(quantile_output, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.data
                    cnt += 1
                epoch_loss /= cnt
                if epoch % 100 == 0:
                    logger.info('Epoch [{}] | Node {} | Quantile Loss {}'.format(epoch, node, epoch_loss))

                if self.q_net_params.lr_schedule:
                    lr_scheduler(epoch_loss)
                if self.q_net_params.early_stop:
                    early_stopping(epoch_loss)
                    if early_stopping.early_stop:
                        break
            q_nets[node] = net
        self.q_nets = q_nets

    def update_all_experts(self):
        logger.info('<-------------Update Model Bank------------->')
        for name in self.node_list:
            logger.info('Updating for node {}'.format(name))
            self.update_model_bank(name)

    def online_update(self, params):
        # TODO: replace self.models
        model_dict, gn, optimizer = self.models['0'], self.gns['0'], self.optimizers['0']
        test = self.data['0'].iloc[self.test_range]
        # larger learning rate since fewer epochs...
        # self.params.learning_rate = self.params.learning_rate * 10
        MSE = nn.MSELoss()
        h = self.config.mecats_params.horizon
        test_length, m = len(test), 0
        length = len(self.data['0'])
        test_start = range(self.test_range[m] - h, self.test_range[m])
        train_loader = self.get_data_loader(m)
        step_range = range(params.batch_size, test_length + 1, params.batch_size)

        step_weights = torch.zeros((len(step_range) * params.epoch, len(self.experts)))
        step_loss = torch.zeros(len(step_range))
        log_R = -np.inf * np.ones((len(step_range) + 1, len(step_range) + 1))
        log_R[0, 0] = 0
        cnt, ave_weights = np.inf, 0.2 * torch.ones((1, len(self.experts))).to(self.device)
        log_message = np.array([0])
        log_H = np.log(params.hazard)
        log_1mH = np.log(1 - params.hazard)
        cp_model = GaussianUnknownMean(params.prior_mean, params.prior_var, params.observe_var)
        
        for i, n in enumerate(step_range):
            preds = torch.tensor(self.predict(model_dict, self.data['0'], self.test_range[m: m + params.batch_size], test_start), requires_grad=False).to(self.device)
            y = torch.tensor(test[m: n].to_list(), dtype=torch.float32)
            y = torch.unsqueeze(y, 0).to(self.device)
            all_weights = torch.zeros((params.epoch, len(self.experts)))
            
            model_weights = torch.zeros((1, len(self.experts)))
            for batch in train_loader:
                model_weights = gn(batch.to(self.device))
            if params.enable_cp:
                cp = is_changepoint(float(y - torch.matmul(model_weights, preds)), i + 1, cp_model, log_R, log_message, log_H, log_1mH)
                if cp:
                    cnt = 0
                model_weights = (1 - math.exp(-cnt / params.tau)) * model_weights + math.exp(-cnt / params.tau) * ave_weights
                if math.exp(-cnt / params.tau) < 0.1:
                    cnt = np.inf
                else:
                    cnt += 1

            plot_loss = MSE(torch.matmul(model_weights, preds), y).cpu().detach()
            step_loss[i] = plot_loss

            for epoch in range(params.epoch):
                total_loss, count = 0., 0
                epoch_weights = torch.zeros((1, len(self.experts)), dtype=torch.float32, requires_grad=False)
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
                # model_dict['pydlm'].fitForwardFilter(useRollingWindow=True, windowLength=config.pydlm_params.window)
                # model_dict['auto_arima'] = model_dict['auto_arima'].update(new_obs)
                # date = self.date[:length - self.te + i]
                # data = self.data['0'][:length - self.te + i]
                # full_data = pd.concat([date, data], axis=1)
                # full_data.rename(columns={'0': 'y'}, inplace=True)
                # model_dict['fbprophet'] = Prophet().fit(full_data, init=stan_init(model_dict['fbprophet']))

        if self.plot and not params.enable_cp:
            utils.plot_weights_cp(self.all_weights.numpy(), step_weights.numpy(), self.params.num_epochs, len(step_range) * params.epoch)
        return step_loss.numpy(), len(step_range)

    def get_data_loader(self, m):
        gating_range = range(self.test_range[m] - self.config.MecatsParams.horizon - self.config.GatingNetParams().batch_size + 1, self.test_range[m])
        gn_input = utils.prepare_data(gating_range, self.data['0'], self.window)
        dataset = utils.GatingDataset(gn_input)
        train_loader = DataLoader(dataset, batch_size=self.config.GatingNetParams().batch_size, num_workers=4, shuffle=False, drop_last=True)
        return train_loader

    def inference(self):
        if not self.config.MecatsParams.metric_mode:
            test = np.zeros((len(self.test_range), 1))
        else:
            test = self.data.iloc[self.test_range]

        pred_dict, quantile_dict = {}, {}
        test_start = range(self.test_range[0] - self.config.MecatsParams.horizon, self.test_range[0])
        
        with concurrent.futures.ProcessPoolExecutor() as executer:
            if self.config.MecatsParams.metric_mode:
                results = [executer.submit(predict, vertex, self.data[vertex], self.test_range, test_start, self.experts, self.cuda, self.path) for vertex in self.node_list]
            else:
                results = [executer.submit(predict, vertex, self.data[vertex], self.test_range, test_start, self.experts, self.cuda, self.path, True) for vertex in self.node_list]
            for f in concurrent.futures.as_completed(results):
                node, preds = f.result()[0], f.result()[1]
                logger.info('Predict on test set for node {}'.format(node))
                weights = np.expand_dims(self.weights[node], 0)
                np.savez(self.path + f'results_{node}.npz', weights=weights, preds=preds)
                combined_pred = np.matmul(weights, preds)[0]
                pred_dict[node] = combined_pred
                if self.config.MecatsParams.quantile:
                    net, data = self.q_nets[node], self.data[node]
                    # here we use the most recent records for quantile predictions
                    if not train_config.MecatsParams.metric_mode:
                        test_start_point = len(data)
                    else:
                        test_start_point = len(data) - self.te
                    pred_length = combined_pred.shape[0]
                    set_range = range(test_start_point - pred_length, test_start_point)
                    q_net_input = torch.squeeze(utils.prepare_data(set_range, data, self.q_net_params.window))
                    dataset = utils.QuantileDataset(q_net_input)
                    test_loader = DataLoader(dataset, batch_size=len(dataset), num_workers=4, shuffle=False)
                    for batch in test_loader:
                        batch = batch.to(self.device)
                        quantile_dict[node] = net(batch, combined_pred).cpu().detach().numpy()
                    
                    if self.plot and train_config.MecatsParams.metric_mode and not self.config.MecatsParams.unit_test:
                        level = TreeNodes(self.nodes, name=node).get_levels()
                        legend = True
                        utils.forecast_plot(np.array(self.data.iloc[self.valid_range][node]), combined_pred, 
                                            np.array(test[node]), quantile_dict[node], level, node, legend)
        return pred_dict, test, quantile_dict

    def update_model_bank(self, name):
        '''
        Update original model bank with observations used to train gating network.
        Use this function after training the gating network.
        '''
        data = self.data[name].dropna() #
        length = len(data)
        if self.config.MecatsParams.metric_mode:
            train_valid_data = data[:length - self.te]
        else:
            train_valid_data = data[:length]    

        if 'auto_arima' in self.experts:
            try:
                logger.info('Updating Auto-Arima Models...')
                with open(self.path + f'arima_{name}.pkl', 'rb') as file:
                    auto_arima = pickle.load(file)
                data_valid = data.iloc[self.valid_range]
                arima = auto_arima.update(data_valid)
                with open(self.path + f'arima_{name}.pkl', 'wb') as pkl:
                    pickle.dump(arima, pkl)
            except:
                logger.info('Updating Auto-Arima (average) Models...')
                np.savez(self.path + f'average_{name}.npz', average=np.average(train_valid_data.to_numpy()))

        if 'average' in self.experts:
            logger.info('Updating Average Models...')
            np.savez(self.path + f'average_{name}.npz', average=np.average(train_valid_data.to_numpy()))

        if 'deepar' in self.experts:
            logger.info('Updating DeepAR Models...')
            deepar = Net(self.config.DeeparParams())
            deepar = utils.load_checkpoint(os.path.join(self.config.DeeparParams().model_dir, 'best.pth.tar'), deepar)
            deepar_params = self.config.DeeparParams()
            if self.config.MecatsParams.metric_mode:
                train_valid_range = range(deepar_params.train_window, length - self.te)
            else:
                train_valid_range = range(deepar_params.train_window, length)
            train_deepar(model=deepar, data=data, params=deepar_params, train_range=train_valid_range, 
                         valid_range=self.valid_range, config=self.config, plot=self.plot)

        if 'fbprophet' in self.experts:
            logger.info('Updating Prophet Models...')
            with open(self.path + f'fbprophet_{name}.json', 'r') as file:
                fbprophet = model_from_json(json.load(file))
            full_data = pd.DataFrame(data={'ds': train_valid_data.index, 'y': train_valid_data.values})
            try:
                m2 = Prophet().fit(full_data, init=stan_init(fbprophet))
            except:
                m2 = Prophet().fit(full_data)
            with open(self.path + f'fbprophet_{name}.json', 'w') as fout:
                json.dump(model_to_json(m2), fout)

        if 'lstnet' in self.experts:
            logger.info('Updating LSTNet Models...')
            with open(os.path.join(self.path, 'LSTNet.pt'), 'rb') as f:
                lstnet = torch.load(f)
            lstnet_params = self.config.LstNetParams()
            train_lstn(data=data, lstn_params=lstnet_params, tr=1., val=0., 
                       cuda=self.cuda, verbose=True, unit_test=self.config.MecatsParams.unit_test, init=lstnet)
            
        if 'pydlm' in self.experts:
            try:
                logger.info('Updating PyDLM Models...')
                pydlm_params = self.config.PydlmParams()
                Trend = trend(degree=pydlm_params.degree_linear, discount=pydlm_params.discount_linear, name='trend', w=pydlm_params.w_linear)
                seasonal_yearly = seasonality(period=pydlm_params.seasonality_period, discount=pydlm_params.discount_seasonal, name='weeklyY', w=pydlm_params.w_seasonal)
                seasonal_quarterly = seasonality(period=int(pydlm_params.seasonality_period / 4), discount=pydlm_params.discount_seasonal, name='weeklyQ', w=pydlm_params.w_seasonal / 4)
                dlm_model = dlm(train_valid_data.to_list()) + Trend + seasonal_yearly + seasonal_quarterly
                dlm_model.fit()
                with open(self.path + f'pydlm_{name}.pkl', 'wb') as pkl:
                    pickle.dump(dlm_model, pkl)
            except:
                logger.info('Updating PyDLM (average) Models...')
                np.savez(self.path + f'average_{name}.npz', average=np.average(train_valid_data.to_numpy()))


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


def predict(name, data, set_range, pred_start, experts, cuda, path, test_pred=False):
    '''
    Model bank prediction on the specified validation set.
    '''
    n, pred = len(set_range), []
    if 'auto_arima' in experts:
        try:
            logger.info('Auto-ARIMA prediction.')
            with open(path + f'arima_{name}.pkl', 'rb') as file:
                auto_arima = pickle.load(file)
            preds_auto_arima = np.array(auto_arima.predict(n_periods=n, return_conf_int=False), dtype=np.float32)
        except:
            logger.info('Auto-ARIMA (average) prediction.')
            preds_auto_arima = np.array(np.repeat(np.load(path + f'average_{name}.npz')['average'], n), dtype=np.float32)
        pred.append(preds_auto_arima)
    if 'average' in experts:
        logger.info('Average prediction.')
        ave = np.array(np.repeat(np.load(path + f'average_{name}.npz')['average'], n), dtype=np.float32)
        pred.append(ave)
    if 'deepar' in experts:
        logger.info('DeepAR prediction.')
        deepar = Net(train_config.DeeparParams())
        deepar = utils.load_checkpoint(os.path.join(train_config.DeeparParams().model_dir, 'best.pth.tar'), deepar)
        deepar_params = train_config.DeeparParams()
        length = data.shape[0]
        split_point = int((1 - train_config.MecatsParams.valid_ratio) * length)
        valid_range = range(split_point, length)
        if test_pred and not train_config.MecatsParams.metric_mode:
            preds_deepar = pred_deepar(data, valid_range, deepar, deepar_params, test_pred, train_config.MecatsParams.pred_step, train_config).squeeze(0)
        else:
            preds_deepar = pred_deepar(data, set_range, deepar, deepar_params, test_pred, len(set_range), train_config).squeeze(0)
        pred.append(preds_deepar)
    if 'fbprophet' in experts:
        logger.info('Prophet prediction.')
        with open(path + f'fbprophet_{name}.json', 'r') as file:
            fbprophet = model_from_json(json.load(file))
        future_df = fbprophet.make_future_dataframe(periods=n)
        preds_fbprophet = np.array(fbprophet.predict(future_df)['yhat'][-n:].to_list(), dtype=np.float32)
        pred.append(preds_fbprophet)
    if 'lstnet' in experts:
        logger.info('LSTNet prediction.')
        with open(os.path.join(path, 'LSTNet.pt'), 'rb') as f:
            lstnet = torch.load(f)
        lstnet_params = train_config.LstNetParams()
        preds_lstnet = pred_lstn(data, lstnet_params.h, pred_start, lstnet, lstnet_params.window, n, cuda).squeeze(1).cpu().detach().numpy()
        pred.append(preds_lstnet)
    if 'pydlm' in experts:
        try:
            logger.info('PyDLM prediction.')
            with open(path + f'pydlm_{name}.pkl', 'rb') as file:
                dlm = pickle.load(file)
            preds_pydlm = np.array(dlm.predictN(N=n)[0], dtype=np.float32)
        except:
            logger.info('PyDLM (average) prediction.')
            preds_pydlm = np.array(np.repeat(np.load(path + f'average_{name}.npz')['average'], n), dtype=np.float32)
        pred.append(preds_pydlm)
    return (name, np.stack(pred, axis=0))


def fit_gn(gpu, data, gn_params, parallel_params, optimizers, gns, name, window, split_point, 
           device, experts, valid_range, all_preds, plot, mecats_params, nodes, path, childs=None):

    print('Train gating network at node {}'.format(name))
    rank = parallel_params.nr * parallel_params.n_gpu + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=parallel_params.world_size, rank=rank)
    torch.cuda.set_device(gpu)

    MSE, regularization = nn.MSELoss().to(device + str(gpu)), False
    gn, optimizer, col_data = gns[name].to(device + str(gpu)), optimizers[name], data[name]
    gn = nn.parallel.DistributedDataParallel(gn, device_ids=[gpu])
    set_range = range(window, split_point)
    if gn_params.lr_schedule:
        lr_scheduler = utils.LRScheduler(optimizer)
    if gn_params.early_stop:
        early_stopping = utils.EarlyStopping()

    if childs is not None:
        gn_input = utils.prepare_data(set_range, data[childs], window)
        regularization = True
    else:
        gn_input = utils.prepare_data(set_range, col_data, window)
        
    # Don't set batch size too large (>128) for short time series.
    dataset = utils.GatingDataset(gn_input)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=parallel_params.world_size, rank=rank)
    train_loader = DataLoader(dataset=dataset, batch_size=gn_params.batch_size, num_workers=0, shuffle=False, pin_memory=True, sampler=train_sampler, drop_last=True)
    preds = all_preds[name].to(device + str(gpu))
    y = torch.tensor(col_data[valid_range].to_list(), dtype=torch.float32)
    y = torch.unsqueeze(y, 0).to(device + str(gpu))
    all_weights = torch.zeros((gn_params.num_epochs, len(experts)))

    for epoch in range(gn_params.num_epochs):
        epoch_loss, count = 0., 0
        epoch_weights = torch.zeros((1, len(experts)), dtype=torch.float32, requires_grad=False)
        for train_batch in train_loader:
            if regularization:
                batch = torch.unsqueeze(train_batch[:, :, 0], 2)
            else:
                batch = train_batch
            
            batch = batch.to(device + str(gpu))
            weights = gn(batch)
            weighted_preds = torch.matmul(weights, preds)
            optimizer.zero_grad()

            if regularization:
                loss = MSE(weighted_preds, y) + get_reg(all_preds, childs, gns, batch, device + str(gpu), gn_params.Lambda)
            else:
                loss = MSE(weighted_preds, y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.data
            epoch_weights += weights.cpu().data
            count += 1

        epoch_loss /= count
        all_weights[epoch] = (epoch_weights / count)[0]
        if epoch % 10 == 0:
            print('Epoch [{}] | Node {} | MSE {}'.format(epoch, name, epoch_loss))
        if gn_params.lr_schedule:
            lr_scheduler(epoch_loss)
        if gn_params.early_stop:
            early_stopping(epoch_loss)
            if early_stopping.early_stop:
                break
    np.savez(path + 'gn_weights_p{}_n{}.npz'.format(gpu, name), weights=all_weights[-1, :].numpy())
    if plot and not mecats_params.unit_test:
        level = TreeNodes(nodes, name=name).get_levels()
        utils.plot_weights(all_weights.numpy(), gn_params.num_epochs, level, name, True, experts)


def get_reg(all_preds, childs, gns, batch_data, device, Lambda=.5):
    reg = torch.zeros((1, all_preds['0'].shape[1]), requires_grad=False, device=device)
    for i, node in enumerate(childs):
        gn = gns[node].to(device)
        weights = gn(batch_data)
        preds = all_preds[node].to(device)
        if i == 0:
            reg += torch.matmul(weights, preds)
        else:
            reg -= torch.matmul(weights, preds)
    return Lambda * torch.sum(torch.pow(reg, 2))