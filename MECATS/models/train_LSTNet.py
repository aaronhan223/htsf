import numpy as np
import pandas as pd
from preprocess.hierarchical import TreeNodes
from preprocess import utils
from evaluation.metrics import compute_level_loss
from recon.MinT import mint_py
from models import LSTNet, Optim
import torch
import torch.nn as nn
import math
import time
from tqdm import trange
import os
from itertools import chain
import pdb


def evaluate(data, X, Y, model, h, evaluateL2, evaluateL1, batch_size, part, nodes, method, cuda):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    if part == 'test':
        inputs = data.valid[0][-1, :, :].unsqueeze(0)
        inputs = inputs.cuda() if cuda else inputs
        n = X.shape[0]
        output = torch.zeros((n, X.shape[2]))
        for i in range(n):
            pred = model(inputs).detach()[0]
            output[i, :] = pred
            inputs = torch.cat((inputs[:, h:, :], pred.unsqueeze(0).unsqueeze(1)), 1)
        col_names = TreeNodes(nodes).col_order()
        result = None
        for j in trange(output.shape[0]):
            row_pred = output[j, :].cpu().detach().numpy()
            recon_pred = mint_py(TreeNodes(nodes).get_s_matrix(), row_pred, method)
            result = recon_pred if j == 0 else np.concatenate((result, recon_pred), axis=0)
        pred_dict = {}
        for i, node in enumerate(col_names):
            pred_dict[node] = result[:, i]
        full_test = pd.DataFrame(data=Y.cpu().detach().numpy(), columns=col_names)
        loss = compute_level_loss(pred_dict.keys(), nodes, pred_dict, full_test, len(nodes) + 1)
        return loss

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


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.data.cpu().numpy()
        n_samples += (output.size(0) * data.m)
    return total_loss / n_samples


def fit_and_pred(Data, model, h, num_epoch, batch_size, optim, TRAINING_METHOD, criterion, evaluateL2, evaluateL1,
                 model_dir, dataset, nodes, verbose, cuda):

    best_val = 10000000
    for epoch in range(1, num_epoch + 1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, batch_size)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, h, evaluateL2,
                                               evaluateL1, batch_size, 'valid', nodes, TRAINING_METHOD, cuda)
        if verbose:
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae '
                '{:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        if val_loss < best_val:
            with open(os.path.join(model_dir, '{}_{}_LSTNet.pt'.format(TRAINING_METHOD, dataset)), 'wb') as f:
                torch.save(model, f)
            best_val = val_loss


def train_lstn(TRAINING_METHOD, nodes, data, cuda, params, dataset, verbose):

    if TRAINING_METHOD == 'BU':
        start = sum(list(chain(*nodes[:-1]))) + 1
        end = sum(list(chain(*nodes))) + 1
        feat_list = [str(i) for i in range(start, end)]
        data = data[feat_list]
    sharq_params = params['sharq']
    lstnet_params = params['lstnet']

    data_dim = 1 if len(data.shape) == 1 else data.shape[1]
    Data = utils.Data_utility(sharq_params.train_split, sharq_params.valid_split, cuda, 
                              sharq_params.FORECAST_HORIZON, lstnet_params.window, data, 
                              TRAINING_METHOD, normalize=lstnet_params.normalize)
    model = LSTNet.Model(data_dim, lstnet_params.window)
    criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    if cuda:
        model.cuda()
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()

    optim = Optim.Optim(model.parameters(), lstnet_params.optimizer, lstnet_params.learning_rate, 10.)
    fit_and_pred(Data, model, sharq_params.FORECAST_HORIZON, lstnet_params.num_epoch, lstnet_params.batch_size, optim, 
                 TRAINING_METHOD, criterion, evaluateL2, evaluateL1, lstnet_params.model_dir, dataset, nodes, verbose, cuda)

    with open(os.path.join(lstnet_params.model_dir, '{}_{}_LSTNet.pt'.format(TRAINING_METHOD, dataset)), 'rb') as f:
        model = torch.load(f)
    multilevel_loss = evaluate(Data, Data.test[0], Data.test[1], model, sharq_params.FORECAST_HORIZON, evaluateL2, 
                               evaluateL1, lstnet_params.batch_size, 'test', nodes, TRAINING_METHOD, cuda)
    print('Multi-level MAPE is:', multilevel_loss)
    return multilevel_loss
