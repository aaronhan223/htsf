from collections import OrderedDict
from algorithms.RNN import RNNModel
from algorithms.AR import AutoRegressive
from algorithms.LSTM import LSTMModel
from algorithms import LSTNet, Optim
import torch

p = 5


def get_models_optimizers(node_list, algs, cuda, lr, hidden_dim, layer_dim, nonlinearity, Data):
    models, quantile_models, optimizers = OrderedDict(), OrderedDict(), OrderedDict()
    quantile_optimizers, combined_optimizers = OrderedDict(), OrderedDict()

    for name in node_list:
        model_dict = {'rnn': [RNNModel(input_dim=1, hidden_dim=hidden_dim, layer_dim=layer_dim, quantiles=[0.5],
                                       nonlinearity=nonlinearity),
                              RNNModel(input_dim=1, hidden_dim=hidden_dim, layer_dim=layer_dim, quantiles=[0.05, 0.95],
                                       nonlinearity=nonlinearity)],
                      'lstm': [LSTMModel(input_dim=1, hidden_dim=hidden_dim, layer_dim=layer_dim, quantiles=[0.5]),
                               LSTMModel(input_dim=1, hidden_dim=hidden_dim, layer_dim=layer_dim, quantiles=[0.05, 0.95])],
                      'ar': [AutoRegressive(quantiles=[0.5], p=p), AutoRegressive(quantiles=[0.05, 0.95], p=p)],
                      'LSTNet': [LSTNet.Model(Data, method='sharq', quantiles=[0.5]),
                                 LSTNet.Model(Data, method='sharq', quantiles=[0.05, 0.95])]}
        model, quantile_model = model_dict[algs][0], model_dict[algs][1]
        if cuda:
            models[name], quantile_models[name] = model.cuda(), quantile_model.cuda()
        else:
            models[name], quantile_models[name] = model, quantile_model
        optimizer_dict = {'rnn': [torch.optim.SGD(models[name].parameters(), lr=lr),
                                  torch.optim.SGD(quantile_models[name].parameters(), lr=lr),
                                  torch.optim.Adam(quantile_models[name].parameters(), lr=lr)],
                          'lstm': [torch.optim.SGD(models[name].parameters(), lr=lr),
                                   torch.optim.SGD(quantile_models[name].parameters(), lr=lr),
                                   torch.optim.Adam(quantile_models[name].parameters(), lr=lr)],
                          'ar': [torch.optim.SGD(models[name].parameters(), lr=lr),
                                 torch.optim.SGD(quantile_models[name].parameters(), lr=lr),
                                 torch.optim.Adam(quantile_models[name].parameters(), lr=lr)],
                          'LSTNet': [torch.optim.Adam(models[name].parameters(), lr=lr),
                                     torch.optim.Adam(quantile_models[name].parameters(), lr=lr),
                                     torch.optim.Adam(quantile_models[name].parameters(), lr=lr)]}
        optimizers[name] = optimizer_dict[algs][0]
        quantile_optimizers[name] = optimizer_dict[algs][1]
        combined_optimizers[name] = optimizer_dict[algs][2]
    return models, quantile_models, optimizers, quantile_optimizers, combined_optimizers, p
