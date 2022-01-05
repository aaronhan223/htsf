import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tensorboardX import SummaryWriter
import queue
from preprocess.hierarchical import TreeNodes
from evaluation.metrics import compute_level_loss, get_recon_error
from algorithms.quantile import QuantileLoss
from algorithms.MinT import recon_base_forecast
from algorithms.ERM import unbiased_recon
from torch.autograd import Variable
from tqdm import tqdm
import pdb

writer_median = SummaryWriter('runs/exp-1')
writer_quantile = SummaryWriter('runs/exp-2')
combined_writer = SummaryWriter('runs/exp-3')


class Learner:
    def __init__(self, models, optimizers, quantile_models, quantile_optimizers, combined_optimizers, num_epochs, train,
                 validation, test, nodes, node_list, recon, alg, lag, H, h, l, hierarchical, cuda, gpu, Data,
                 verbose=False):

        self.models = models
        self.optimizers = optimizers
        self.quantile_models = quantile_models
        self.quantile_optimizers = quantile_optimizers
        self.combined_optimizers = combined_optimizers
        self.nodes = nodes
        self.train = train
        self.validation = validation
        self.test = test
        self.node_list = node_list
        self.num_epochs = num_epochs
        self.lr_decay = .1
        self.check_points = {'normal': [100, .5], 'combined': [1000, 5]}
        self.recon = recon
        self.alg = alg
        self.lag = lag
        self.H = H
        self.h = h
        self.l = l
        self.hierarchical = hierarchical
        self.verbose = verbose
        self.Data = Data
        self.cuda = cuda
        if cuda:
            self.device = 'cuda:' + str(gpu)
        else:
            self.device = 'cpu'
        self.loss = QuantileLoss([0.5]).to(self.device)
        self.qloss = QuantileLoss([0.05, 0.95]).to(self.device)

    def bottom_forecast(self):
        node_list = TreeNodes(self.nodes).nodes_by_level(self.l)
        for name in node_list:
            self.fit(self.models, self.optimizers, self.loss, name, writer_median)
        return self.bottom_median_predict(node_list)

    def base_forecast(self):
        for name in self.node_list:
            self.fit(self.models, self.optimizers, self.loss, name, writer_median)
        return self.median_predict()

    def fit_median_recon(self):
        node_list = TreeNodes(self.nodes).nodes_by_level(self.l)
        for name in node_list:
            self.fit(self.models, self.optimizers, self.loss, name, writer_median)
        for i in range(self.l - 1, 0, -1):
            nodes = TreeNodes(self.nodes).nodes_by_level(i)
            for name in nodes:
                childs = TreeNodes(self.nodes, name=name).get_child()
                self.fit(self.models, self.optimizers, self.loss, name, writer_median, childs=childs)
        return self.median_predict()

    def fit_quantile(self, median_forecast):
        for name in self.node_list:
            self.fit(self.quantile_models, self.quantile_optimizers, self.qloss, name, writer_quantile)

        if self.hierarchical:
            # do the reconciliation in the bottom-up fashion
            for i in range(self.l - 1, 0, -1):
                node_list = TreeNodes(self.nodes).nodes_by_level(i)
                for name in node_list:
                    childs = TreeNodes(self.nodes, name=name).get_child()
                    updated_models = OrderedDict()
                    q_train, q_valid = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
                    for node in childs:
                        updated_models[node] = self.quantile_models[node]

                    if self.verbose:
                        print('Reconciling node {} at level {}'.format(name, i))
                    for epoch in range(self.num_epochs):
                        self.combined_optimizers[name].zero_grad()
                        combined_loss = self.get_normalized_qloss(self.train, updated_models, median_forecast)
                        combined_loss.backward()
                        self.combined_optimizers[name].step()
                        combined_writer.add_scalar(tag=name, scalar_value=combined_loss.item(), global_step=epoch)
                        if self.verbose:
                            print('Epoch [{}]: training combined loss in node {}: {}'.format(epoch, name,
                                                                                             combined_loss.item()))

                        combined_loss_valid = self.get_normalized_qloss(self.validation, updated_models, median_forecast)
                        if self.early_stop(q_train, q_valid, combined_loss.item(), combined_loss_valid.item(), 1,
                                           self.combined_optimizers[name], type='combined'):
                            break
                    print('\n')
            combined_writer.close()

    def get_data(self, name, models):
        if self.alg == 'LSTNet':
            X, y = torch.tensor(self.train[0][name], dtype=torch.float32), torch.tensor(self.train[1][name], dtype=torch.float32)
            X_valid, y_valid = torch.tensor(self.validation[0][name], dtype=torch.float32), torch.tensor(self.validation[1][name], dtype=torch.float32)
            if self.cuda:
                X, y, X_valid, y_valid = X.cuda(), y.cuda(), X_valid.cuda(), y_valid.cuda()
            return Variable(X), Variable(y), Variable(X_valid), Variable(y_valid)

        if self.hierarchical:
            series, series_valid = np.array(self.train[name]), np.array(self.validation[name])
        else:
            series, series_valid = np.array(self.train), np.array(self.validation)
        X, y = list(models.values())[0].get_x_y(series[:-self.H], [[series[-self.H + self.h]]])
        X_valid, y_valid = list(models.values())[0].get_x_y(series_valid[:-self.H], [[series_valid[-self.H + self.h]]])
        return X, y, X_valid, y_valid

    def fit(self, models, optimizers, loss_func, name, writer, warm_start=True, **kwargs):
        '''
        Train quantile loss of each node individually before reconciliation. Keep a validation set to monitor the
        convergence of each node while training.
        '''
        if self.verbose:
            print('Training node {} '.format(name))
        q_train, q_valid, regularization, MSE = queue.Queue(maxsize=2), queue.Queue(maxsize=5), False, nn.MSELoss()
        X, y, X_valid, y_valid = self.get_data(name, models)

        if 'childs' in kwargs:
            childs = kwargs.pop('childs')
            regularization = True

        for epoch in range(self.num_epochs):
            models[name].train()
            if self.alg == 'LSTNet':
                models[name].zero_grad()
                output = models[name](X)

            else:
                optimizers[name].zero_grad()
                output = models[name](X)

            if regularization:
                if warm_start:
                    loss = MSE(output, y) + self.get_reg(self.train, childs)
                else:
                    loss = loss_func(output, y) + self.get_reg(self.train, childs)
            else:
                if warm_start:
                    loss = MSE(output, y)
                else:
                    loss = loss_func(output, y)
            loss.backward()
            optimizers[name].step()
            if loss.item() < self.check_points['normal'][0]:
                warm_start = False
            writer.add_scalar(tag=name, scalar_value=loss.item(), global_step=epoch)
            if self.verbose:
                print('Epoch [{}]: training {} quantile loss in node {}: {}'.format(epoch, loss_func.get_q_length(),
                                                                                    name, loss.item()))

            output_valid = models[name](X_valid)
            if regularization:
                if warm_start:
                    loss_valid = MSE(output_valid, y_valid) + self.get_reg(self.validation, childs)
                    loss_valid = loss_valid.item()
                else:
                    loss_valid = loss_func(output_valid, y_valid) + self.get_reg(self.validation, childs)
                    loss_valid = loss_valid.item()
            else:
                if warm_start:
                    loss_valid = MSE(output_valid, y_valid).item()
                else:
                    loss_valid = loss_func(output_valid, y_valid).item()

            if self.early_stop(q_train, q_valid, loss.item(), loss_valid, 0.05, optimizers[name]):
                break
        if self.verbose:
            print('\n')
        writer.close()

    def get_reg(self, data, childs, epsilon=.1):
        reg = torch.tensor([[0.]], requires_grad=False, device=self.device)
        test = np.array(data[childs])[:-self.H, :]
        for i, node in enumerate(childs):
            if i == 0:
                reg += self.models[node].forward(self.models[node].get_x_y(test[:, i]))
            else:
                reg -= self.models[node].forward(self.models[node].get_x_y(test[:, i]))
        return epsilon * torch.pow(reg, 2)

    def get_normalized_qloss(self, data, models, median_forecast):
        loss_lower = torch.tensor([[0.]], requires_grad=False, device=self.device)
        loss_upper = torch.tensor([[0.]], requires_grad=False, device=self.device)
        for i, name in enumerate(models.keys()):
            X = models[name].get_x_y(np.array(data[name])[:-self.H])
            median = torch.tensor(median_forecast[name])
            if i == 0:
                loss_lower += torch.pow((models[name](X)[:, 0] - median), 2)
                loss_upper += torch.pow((models[name](X)[:, 1] - median), 2)
            else:
                loss_lower -= torch.pow((models[name](X)[:, 0] - median), 2)
                loss_upper -= torch.pow((models[name](X)[:, 1] - median), 2)
        return torch.pow(loss_lower, 2) + torch.pow(loss_upper, 2)

    def early_stop(self, q_train, q_valid, train_loss, valid_loss, epsilon, optimizer, type='normal'):
        if train_loss == 0.:
            return True

        if q_train.full():
            _ = q_train.get()
        q_train.put(train_loss)

        if train_loss < self.check_points[type][1] and self.is_increase(list(q_train.queue)):
            return True

        if train_loss < self.check_points[type][0] and self.is_increase(list(q_train.queue)):
            optimizer.param_groups[0]['lr'] *= self.lr_decay

        if q_valid.full():
            _ = q_valid.get()
        q_valid.put(valid_loss)

        if q_valid.full():
            if self.is_increase(list(q_valid.queue)) or valid_loss < epsilon:
                return True
        return False

    def is_increase(self, list_q):
        bol = True
        for i in range(len(list_q) - 1):
            bol = bol and (list_q[i] <= list_q[i + 1])
        return bol

    def bottom_median_predict(self, node_list):
        test_pred = []
        for name in node_list:
            series = np.array(self.test[name])
            X, y = self.models[name].get_x_y(series[:-self.H], [[series[-self.H + self.h]]])
            output = self.models[name](X)
            test_pred.append(float(output))
        S = TreeNodes(self.nodes).get_s_matrix()
        full_pred = np.dot(S, np.array(test_pred))
        return dict(zip(TreeNodes(self.nodes).col_order(), full_pred))

    def median_predict(self):
        test_pred = {}
        for name in self.node_list:
            series = np.array(self.test[name]) if self.hierarchical else np.array(self.test)
            X, y = self.models[name].get_x_y(series[:-self.H], [[series[-self.H + self.h]]])
            output = self.models[name](X)
            test_pred[name] = float(output)
        return test_pred

    def predict(self, median_forecast, quantile):
        results = {}
        if 'mint' in self.recon:
            if self.alg == 'ar':
                median_forecast = recon_base_forecast(self.node_list, self.nodes, median_forecast, self.models,
                                                      self.train, self.lag, self.recon, self.alg)
            else:
                median_forecast = recon_base_forecast(self.node_list, self.nodes, median_forecast, self.models,
                                                      self.train, self.h, self.recon, self.alg)
        elif self.recon == 'erm':
            y_hat = np.expand_dims(np.fromiter(median_forecast.values(), dtype=float), axis=0)
            y = np.expand_dims(self.test.values[-self.H + self.h, :], axis=0)
            median_forecast = unbiased_recon(self.nodes, y, y_hat)

        if quantile:
            level_llh = dict(zip(range(self.l), np.zeros(self.l)))
            upper_loss, lower_loss, median_loss = QuantileLoss([0.95]).to(self.device), QuantileLoss([0.05]).to(self.device), QuantileLoss([0.5]).to(self.device)
            for name in self.node_list:
                if self.hierarchical:
                    series, train = np.array(self.test[name]), np.array(self.train[name])
                else:
                    series, train = np.array(self.test), np.array(self.train)

                X, y = list(self.models.values())[0].get_x_y(series[:-self.H], [[series[-self.H + self.h]]])
                median = torch.tensor(np.array([[median_forecast[name]]]), requires_grad=False, device=self.device)
                trivial_upper = torch.tensor(np.array([[np.percentile(train, 95)]]), requires_grad=False, device=self.device)
                trivial_lower = torch.tensor(np.array([[np.percentile(train, 5)]]), requires_grad=False, device=self.device)
                trivial_median = torch.tensor(np.array([[np.percentile(train, 50)]]), requires_grad=False, device=self.device)

                lower = self.quantile_models[name](X)[:, 0].item()
                upper = self.quantile_models[name](X)[:, 1].item()
                if upper <= median_forecast[name]:
                    upper = 2 * median_forecast[name] - lower
                upper = torch.tensor(np.array([[upper]]), requires_grad=False, device=self.device)
                lower = torch.tensor(np.array([[lower]]), requires_grad=False, device=self.device)

                Q = (upper_loss.q_loss_1d(upper, y) + lower_loss.q_loss_1d(lower, y) +
                     median_loss.q_loss_1d(median, y)) / 3
                results[name] = (upper.item(), median.item(), lower.item(), y.item())
                q = (upper_loss.q_loss_1d(trivial_upper, y) + lower_loss.q_loss_1d(trivial_lower, y) +
                     median_loss.q_loss_1d(trivial_median, y)) / 3
                if q == 0.:
                    q += 1e-10

                if self.hierarchical:
                    node = TreeNodes(self.nodes, name=name)
                    level_llh[node.get_levels()] += Q / q
                else:
                    level_llh[0] += Q / q

            for l in range(1, len(level_llh) + 1):
                if self.hierarchical:
                    print('[{}, h={}] likelihood ratio at level {} is {}'.format(self.recon, self.h + 1, l - 1, level_llh[l - 1] / len(TreeNodes(self.nodes).nodes_by_level(l))))
                else:
                    print('[{}, h={}] likelihood ratio is {}'.format(self.recon, self.h + 1, level_llh[l - 1]))

        loss = compute_level_loss(self.node_list, self.nodes, median_forecast, self.test, self.l, self.hierarchical,
                                  -self.H + self.h)
        if self.hierarchical:
            get_recon_error(self.node_list, self.nodes, median_forecast, self.h, self.recon)
        return results, loss
