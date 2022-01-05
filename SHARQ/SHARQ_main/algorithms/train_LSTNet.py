import numpy as np
from preprocess.hierarchical import TreeNodes
from preprocess import utils
from evaluation.labour import compute_level_loss
from algorithms.MinT import recon_base_forecast
from algorithms.ERM import unbiased_recon
from algorithms import LSTNet, Optim
import torch
import torch.nn as nn
import math
import time
from itertools import chain
import pdb


def evaluate(data, X, Y, model, h, evaluateL2, evaluateL1, batch_size, part, nodes, method, alg, cuda):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    if part == 'test':
        output = model(X.cuda()) if cuda else model(X)
        result = np.zeros((output.shape[0], len(nodes) + 1))
        if method == 'erm':
            i = 0
            recon_pred = unbiased_recon(nodes, Y.numpy(), output.cpu().detach().numpy())
            for pred in recon_pred:
                result[i, :] = compute_level_loss(pred.keys(), nodes, pred, Y[i, :].numpy(), len(nodes) + 1, True, h)
                i += 1
        else:
            for i in range(output.shape[0]):
                test_pred = output[i, :].cpu().detach().numpy()
                full_test = Y[i, :].cpu().detach().numpy()
                if method == 'BU':
                    S = TreeNodes(nodes).get_s_matrix()
                    full_test = np.dot(S, full_test)
                    test_pred = np.dot(S, test_pred)
                pred_dict = dict(zip(TreeNodes(nodes).col_order(), test_pred))
                if 'mint' in method:
                    pred_dict = recon_base_forecast(pred_dict.keys(), nodes, pred_dict, model,
                                                    data, data.P + data.h - 1, method, alg)
                result[i, :] = compute_level_loss(pred_dict.keys(), nodes, pred_dict, full_test, len(nodes) + 1, True, h)
        result = result.mean(axis=0)
        return result

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


def fit_and_pred(Data, model, h, num_epoch, batch_size, params, optim, TRAINING_METHOD, criterion, evaluateL2, evaluateL1,
                 nodes, verbose, cuda):

    best_val = 10000000
    for epoch in range(1, num_epoch + 1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, 64)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, h, evaluateL2,
                                               evaluateL1, batch_size, 'valid', nodes,
                                               TRAINING_METHOD, params['alg'], cuda)
        if verbose:
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae '
                '{:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        if val_loss < best_val:
            with open('./save/LSTNet.pt', 'wb') as f:
                torch.save(model, f)
            best_val = val_loss


def train_lstn(TRAINING_METHOD, nodes, data, cuda, h, num_epoch, batch_size, params, verbose):

    if TRAINING_METHOD == 'BU':
        start = sum(list(chain(*nodes[:-1]))) + 1
        end = sum(list(chain(*nodes))) + 1
        feat_list = [str(i) for i in range(start, end)]
        data = data[feat_list]

    Data = utils.Data_utility(0.6, 0.2, cuda, h, 24 * 7, data, TRAINING_METHOD, normalize=2)
    model = LSTNet.Model(Data)
    criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    if cuda:
        model.cuda()
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()

    optim = Optim.Optim(model.parameters(), 'adam', 1e-3, 10.)
    # optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit_and_pred(Data, model, h, num_epoch, batch_size, params, optim, TRAINING_METHOD, criterion, evaluateL2,
                 evaluateL1, nodes, verbose, cuda)

    with open('./save/LSTNet.pt', 'rb') as f:
        model = torch.load(f)
    multilevel_loss = evaluate(Data, Data.test[0], Data.test[1], model, h, evaluateL2, evaluateL1, batch_size,
                               'test', nodes, TRAINING_METHOD, params['alg'], cuda)
    return multilevel_loss
