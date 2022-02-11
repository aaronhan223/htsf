import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import queue
import logging
from hiertsforecaster.preprocess.hierarchical import TreeNodes
from hiertsforecaster.evaluation.metrics import compute_level_loss, get_recon_error
from hiertsforecaster.models.quantile import QuantileLoss
from hiertsforecaster.preprocess import utils
from torch.autograd import Variable
import pdb

logger = logging.getLogger('MECATS.sharq')


class Learner:
    def __init__(self, models, optimizers, quantile_models, quantile_optimizers, combined_optimizers, params, sharq_params, train,
                 validation, test, nodes, node_list, recon, alg, lag, h, l, hierarchical, cuda, gpu, Data, verbose=False):

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
        self.sharq_params = sharq_params
        self.params = params
        self.lr_decay = .1
        self.check_points = {'normal': [100, .5], 'combined': [1000, 5]}
        self.recon = recon
        self.alg = alg
        self.lag = lag
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
        self.loss = QuantileLoss([sharq_params.mid_quantile]).to(self.device)
        self.qloss = QuantileLoss(sharq_params.other_quantiles).to(self.device)

    def bottom_forecast(self):
        node_list = TreeNodes(self.nodes).nodes_by_level(self.l)
        for name in node_list:
            self.fit(self.models, self.optimizers, self.loss, name)
        return self.bottom_median_predict(node_list)

    def base_forecast(self):
        for name in self.node_list:
            self.fit(self.models, self.optimizers, self.loss, name)
        return self.median_predict()

    def fit_median_recon(self):
        node_list = TreeNodes(self.nodes).nodes_by_level(self.l)
        for name in node_list:
            self.fit(self.models, self.optimizers, self.loss, name)
        for i in range(self.l - 1, 0, -1):
            nodes = TreeNodes(self.nodes).nodes_by_level(i)
            for name in nodes:
                childs = TreeNodes(self.nodes, name=name).get_child()
                self.fit(self.models, self.optimizers, self.loss, name, childs=childs)
        return self.median_predict()

    def fit_quantile(self):
        # first fit quantile models at each node independently
        for name in self.node_list:
            self.fit(self.quantile_models, self.quantile_optimizers, self.qloss, name)

        if self.hierarchical:
            # do the reconciliation in the bottom-up fashion
            for i in range(self.l - 1, 0, -1):
                node_list = TreeNodes(self.nodes).nodes_by_level(i)
                for name in node_list:
                    childs = TreeNodes(self.nodes, name=name).get_child()
                    X, y, X_valid, _ = self.get_data(name, childs=childs)
                    updated_models = OrderedDict()
                    q_train, q_valid = queue.Queue(maxsize=2), queue.Queue(maxsize=3)
                    for node in childs:
                        updated_models[node] = self.quantile_models[node]

                    if self.verbose:
                        logger.info('Reconciling node {} at level {}'.format(name, i))
                    for epoch in range(self.params.num_epoch):
                        combined_loss = self.train_combined_loss(updated_models, X, y, name)
                        if self.verbose and epoch % 20 == 0:
                            logger.info('Epoch [{}]: training combined loss in node {}: {}'.format(epoch, name, combined_loss))

                        combined_loss_valid = self.get_normalized_qloss(X_valid, updated_models)
                        if self.early_stop(q_train, q_valid, combined_loss, combined_loss_valid, 1, self.combined_optimizers[name], type='combined'):
                            break
                    print('\n')

    def train_combined_loss(self, model, X, y, name):
        total_loss, n_samples = 0, 0
        for inputs, _ in self.Data.get_batches(X, y, self.params.batch_size):
            self.combined_optimizers[name].zero_grad()
            combined_loss = self.get_normalized_qloss(inputs, model)
            combined_loss.backward()
            self.combined_optimizers[name].step()
            total_loss += combined_loss.data.cpu().numpy()
            n_samples += inputs.shape[0]
        return total_loss / n_samples

    def get_data(self, name, **kwargs):
        if 'childs' in kwargs:
            childs = kwargs.pop('childs')
            assert name == childs[0]
            for i, node in enumerate(childs):
                if i == 0:
                    X = torch.tensor(self.train[0][node], dtype=torch.float32)
                    y = torch.tensor(self.train[1][node], dtype=torch.float32)
                    X_valid = torch.tensor(self.validation[0][node], dtype=torch.float32)
                    y_valid = torch.tensor(self.validation[1][node], dtype=torch.float32)
                else:
                    X = torch.cat((X, torch.tensor(self.train[0][node], dtype=torch.float32)), 2)
                    y = torch.cat((y, torch.tensor(self.train[1][node], dtype=torch.float32)), 1)
                    X_valid = torch.cat((X_valid, torch.tensor(self.validation[0][node], dtype=torch.float32)), 2)
                    y_valid = torch.cat((y_valid, torch.tensor(self.validation[1][node], dtype=torch.float32)), 1)
        else:
            X, y = torch.tensor(self.train[0][name], dtype=torch.float32), torch.tensor(self.train[1][name], dtype=torch.float32)
            X_valid, y_valid = torch.tensor(self.validation[0][name], dtype=torch.float32), torch.tensor(self.validation[1][name], dtype=torch.float32)
        if self.cuda:
            X, y, X_valid, y_valid = X.cuda(), y.cuda(), X_valid.cuda(), y_valid.cuda()
        return Variable(X), Variable(y), Variable(X_valid), Variable(y_valid)

    def fit(self, models, optimizers, loss_func, name, warm_start=True, **kwargs):
        '''
        Train quantile loss of each node individually before reconciliation. Keep a validation set to monitor the
        convergence of each node while training.
        '''
        if self.verbose:
            print('Training / reconciling model at vertex {} '.format(name))
        q_train, q_valid, regularization, MSE = queue.Queue(maxsize=2), queue.Queue(maxsize=5), False, nn.MSELoss()

        if 'childs' in kwargs:
            childs = kwargs.pop('childs')
            X, y, X_valid, y_valid = self.get_data(name, childs=childs)
            regularization = True
        else:
            X, y, X_valid, y_valid = self.get_data(name)

        for epoch in range(self.params.num_epoch):
            models[name].train()
            total_loss, n_samples = 0., 0
            for inputs, labels in self.Data.get_batches(X, y, self.params.batch_size):
                models[name].zero_grad()
                if regularization:
                    batch_input = torch.unsqueeze(inputs[:, :, 0], 2)
                    batch_label = torch.unsqueeze(labels[:, 0], 1)
                else:
                    batch_input = inputs
                    batch_label = labels
                output = models[name](batch_input)

                if regularization:
                    if warm_start:
                        loss = MSE(output, batch_label) + self.get_reg(childs, inputs, self.sharq_params.Lambda)
                    else:
                        loss = loss_func(output, batch_label) + self.get_reg(childs, inputs, self.sharq_params.Lambda)
                else:
                    if warm_start:
                        loss = MSE(output, batch_label)
                    else:
                        loss = loss_func(output, batch_label)
                loss.backward()
                optimizers[name].step()
                total_loss += loss.data
                n_samples += output.shape[0]

            total_loss /= n_samples
            if total_loss < self.check_points['normal'][0]:
                warm_start = False
            if self.verbose and epoch % 100 == 0:
                logger.info('Epoch [{}]: training {} quantile loss in node {}: {}'.format(epoch, loss_func.get_q_length(), name, total_loss))
            
            if regularization:
                valid_input = torch.unsqueeze(X_valid[:, :, 0], 2)
                valid_label = torch.unsqueeze(y_valid[:, 0], 1)
            else:
                valid_input = X_valid
                valid_label = y_valid

            output_valid = models[name](valid_input)
            if regularization:
                if warm_start:
                    loss_valid = MSE(output_valid, valid_label) + self.get_reg(childs, X_valid, self.sharq_params.Lambda)
                    loss_valid = loss_valid.item()
                else:
                    loss_valid = loss_func(output_valid, valid_label) + self.get_reg(childs, X_valid, self.sharq_params.Lambda)
                    loss_valid = loss_valid.item()
            else:
                if warm_start:
                    loss_valid = MSE(output_valid, valid_label).item()
                else:
                    loss_valid = loss_func(output_valid, valid_label).item()

            if self.early_stop(q_train, q_valid, total_loss.item(), loss_valid, 0.05, optimizers[name]):
                break
        if self.verbose:
            print('\n')

    def get_reg(self, childs, inputs, Lambda=.1):
        reg = torch.zeros((inputs.shape[0], 1), requires_grad=False, device=self.device)
        for i, node in enumerate(childs):
            model_input = torch.unsqueeze(inputs[:, :, i], 2).to(self.device)
            if i == 0:
                reg += self.models[node](model_input)
            else:
                reg -= self.models[node](model_input)
        return Lambda * torch.sum(torch.pow(reg, 2))

    def get_normalized_qloss(self, data, models):
        loss_lower = torch.zeros((data.shape[0], 1), requires_grad=False, device=self.device)
        loss_upper = torch.zeros((data.shape[0], 1), requires_grad=False, device=self.device)
        for i, name in enumerate(models.keys()):
            X = torch.unsqueeze(data[:, :, i], 2).to(self.device)
            median = self.models[name](X).detach()
            if i == 0:
                loss_lower += torch.pow((torch.unsqueeze(models[name](X)[:, 0], 1) - median), 2)
                loss_upper += torch.pow((torch.unsqueeze(models[name](X)[:, 1], 1) - median), 2)
            else:
                loss_lower -= torch.pow((torch.unsqueeze(models[name](X)[:, 0], 1) - median), 2)
                loss_upper -= torch.pow((torch.unsqueeze(models[name](X)[:, 1], 1) - median), 2)
        return torch.sum(torch.pow(loss_lower, 2) + torch.pow(loss_upper, 2))

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
            X = torch.tensor(self.test[0][name], dtype=torch.float32, device=self.device)
            output = self.models[name](X).detach()
            test_pred.append(output)
        S = TreeNodes(self.nodes).get_s_matrix()
        full_pred = np.dot(S, np.array(test_pred))
        return dict(zip(TreeNodes(self.nodes).col_order(), full_pred))

    def median_predict(self):
        test_pred = {}
        for name in self.node_list:
            X = torch.tensor(self.validation[0][name][-1, :, :], dtype=torch.float32, device=self.device).unsqueeze(0)
            n = self.test[0][name].shape[0]
            output = torch.zeros((n, X.shape[2]))
            for i in range(n):
                pred = self.models[name](X).detach()[0]
                output[i, :] = pred
                X = torch.cat((X[:, self.h:, :], pred.unsqueeze(0).unsqueeze(1)), 1)
            test_pred[name] = output.squeeze(1).cpu().detach().numpy()
        return test_pred

    def predict(self, median_forecast, data, quantile):

        if quantile:
            quantiles = self.sharq_params.other_quantiles
            num_quantile = len(quantiles)
            val, te = self.sharq_params.valid_split, self.sharq_params.test_split
            length = data.shape[0]
            valid_range = range(int((1 - te - val) * length), int((1 - te) * length))
            test_range = range(int((1 - te) * length), length)
            test = data.iloc[test_range]

            for name in self.node_list:
                inputs = torch.tensor(np.repeat(np.expand_dims(self.validation[0][name][-1, :, :], axis=0), num_quantile, 0), dtype=torch.float32, device=self.device)
                n = test.shape[0]
                quantile_preds = torch.zeros((n, num_quantile))
                level = TreeNodes(self.nodes, name=name).get_levels()
                for i in range(n):
                    for q in range(num_quantile):
                        Input = inputs[q, :, :].unsqueeze(0)
                        q_pred = self.quantile_models[name](Input)[:, q].detach()
                        quantile_preds[i, q] = q_pred
                        inputs[q, :, :] = torch.cat((Input[:, self.h:, :], q_pred.unsqueeze(0).unsqueeze(1)), 1)

                median = median_forecast[name]
                quantile_preds = quantile_preds.cpu().detach().numpy()
                utils.quantile_forecast_plot(np.array(data.iloc[valid_range][name]), median, np.array(test[name]),
                                             quantile_preds, level, name, self.sharq_params.DATASET)

        loss = compute_level_loss(self.node_list, self.nodes, median_forecast, self.Data.test_data, self.l)
        if self.hierarchical:
            get_recon_error(self.node_list, self.nodes, median_forecast, self.recon)
        return loss
