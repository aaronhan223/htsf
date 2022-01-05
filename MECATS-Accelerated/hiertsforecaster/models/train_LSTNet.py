from hiertsforecaster.preprocess import utils
from hiertsforecaster.models import LSTNet
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import os
import pdb

logger = logging.getLogger('MECATS.LstNet')


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)

        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).data.cpu().numpy()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).data.cpu().numpy()
        n_samples += (output.size(0) * data.m)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation


def train(data, X, Y, model, criterion, optim, batch_size, shuffle):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, shuffle):
        optim.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        optim.step()
        total_loss += loss.data.cpu().numpy()
        n_samples += (output.size(0) * data.m)
    return total_loss / n_samples


def fit_and_pred(Data, model, params, optim, criterion, evaluateL2, evaluateL1, verbose, unit_test, realm_id):

    best_val = np.inf
    lr_scheduler, early_stopping = utils.LRScheduler(optim), utils.EarlyStopping()
    for epoch in range(1, params.num_epoch + 1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, params.batch_size, 1 - unit_test)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, params.batch_size)
        if verbose:
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae '
                        '{:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        if val_loss < best_val:
            with open(os.path.join(f'./hiertsforecaster/save/{realm_id}/lstnet', 'LSTNet.pt'), 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if params.lr_schedule:
            lr_scheduler(val_loss)
        if params.early_stop:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break


def train_lstn(data, lstn_params, tr, val, cuda, verbose, unit_test, **kwargs):
    Data = utils.Data_utility(tr, val, cuda, lstn_params.h, lstn_params.window, data, normalize=lstn_params.normalize)
    data_dim = 1 if len(data.shape) == 1 else data.shape[1]
    if 'init' in kwargs:
        model = kwargs.pop('init')
    else:
        model = LSTNet.Model(data_dim, lstn_params.window, test=unit_test)
    criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    if cuda:
        model.cuda()
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lstn_params.learning_rate)
    fit_and_pred(Data, model, lstn_params, optimizer, criterion, evaluateL2, evaluateL1, verbose, unit_test)


def pred_lstn(data, h, val, model, window, n, cuda):
    preds = torch.zeros((n, 1))
    X = utils.prepare_data(val, data, window, h)
    # logger.info('window size {}'.format(window))
    # logger.info('X shape {}'.format(X.shape))
    X = X.cuda() if cuda else X
    # use prediction from previous step as inputs for next step
    for i in range(n):
        output = model(X).detach()[0]
        preds[i, :] = output
        X = torch.cat((X[:, h:, :], output.unsqueeze(1).unsqueeze(2)), 1)
    return preds
