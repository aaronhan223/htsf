import torch
import torch.nn as nn
import math
import torch_dct as dct
import pdb


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, qu in enumerate(self.quantiles):
            errors = target - preds[:, i].unsqueeze(1)
            losses.append(torch.max((qu - 1) * errors, qu * errors))
        loss = torch.sum(torch.cat(losses, dim=1))
        return loss

    def get_quantile(self):
        return self.quantiles

    def get_q_length(self):
        if len(self.quantiles) == 1:
            return self.quantiles[0]
        else:
            return 'boundary'

    def q_loss_1d(self, preds, target):
        errors = target.item() - preds.item()
        loss = max((self.quantiles[0] - 1) * errors, self.quantiles[0] * errors)
        return loss


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