from collections import OrderedDict
from models.RNN import RNNModel
from models.AR import AutoRegressive
from models.LSTM import LSTMModel
from models import LSTNet
from recon.learning_moe import fitModel
from recon.learning_moe import Gating_Network, Qunatile_Network
import torch
import logging

p = 5
logger = logging.getLogger('MECATS.Models')
    

def get_moe_optimizers(data, params, node_list, date, dataset, quantile, cuda):
    '''
    Return dictionary of offline-trained MoE models, gating networks, as well as optimizers for the gating network.
    '''
    models, gns, optimizers = OrderedDict(), OrderedDict(), OrderedDict()
    if quantile:
        q_nets, q_optimizers = OrderedDict(), OrderedDict()
    for name in node_list:
        logger.info('Preparing model for node {}'.format(name))
        gating_params = params['gating_network']
        if len(data[name].shape) == 1:
            input_dim = 1
        else:
            input_dim = data[name].shape[1]

        model = fitModel(data[name], date, params, name, dataset, cuda)
        model_dict = model.pretrain()
        gn = Gating_Network(input_dim, gating_params)
        gns[name] = gn
        models[name] = model_dict
        optimizers[name] = torch.optim.Adam(gn.parameters(), lr=gating_params.learning_rate)
        if quantile:
            q_net_params = params['quantile_net']
            if cuda:
                net = Qunatile_Network(q_net_params).to('cuda:{}'.format(torch.cuda.current_device()))
            else:
                net = Qunatile_Network(q_net_params)
            Optim = torch.optim.Adam(net.parameters(), lr=q_net_params.learning_rate)
            q_nets[name], q_optimizers[name] = net, Optim
    if quantile:
        return models, gns, optimizers, q_nets, q_optimizers
    return models, gns, optimizers