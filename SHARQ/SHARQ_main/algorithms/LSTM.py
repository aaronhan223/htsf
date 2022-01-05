import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import chain
import pdb

cuda = torch.cuda.is_available()


class LSTMModel(nn.Module):
    '''
    RNN model with hidden and multiple stacked layer for time series prediction.
    '''
    def __init__(self, input_dim, hidden_dim, layer_dim, quantiles):
        super(LSTMModel, self).__init__()
        self.quantiles = quantiles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = len(quantiles)
        rnns = [nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) for _ in range(len(self.quantiles))]
        final_layers = [nn.Linear(hidden_dim, 1) for _ in range(len(self.quantiles))]
        self.rnn = nn.ModuleList(rnns)
        self.final_layers = nn.ModuleList(final_layers)
        self.init_weights()

    def init_weights(self):
        for m in chain(self.final_layers):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def RNN_input(self, X):
        X = torch.Tensor(X)
        X = torch.unsqueeze(X, 0)
        X = torch.unsqueeze(X, 2)
        if cuda:
            X = Variable(X.cuda())
        return X

    def get_x_y(self, data, target=None):
        X = self.RNN_input(np.array(data, dtype=np.float32))
        if target is None:
            return X
        if cuda:
            y = torch.tensor(np.array(target, dtype=np.float32), requires_grad=False).cuda()
        else:
            y = torch.tensor(np.array(target, dtype=np.float32), requires_grad=False)
        return X, y

    def forward(self, x):
        if cuda:
            h0s = [Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()) for _ in range(len(self.quantiles))]
            c0s = [Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()) for _ in range(len(self.quantiles))]
        else:
            h0s = [Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)) for _ in range(len(self.quantiles))]
            c0s = [Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)) for _ in range(len(self.quantiles))]
        outputs = []
        for rnn, h0, c0, layer in zip(self.rnn, h0s, c0s, self.final_layers):
            out, (_, _) = rnn(x, (h0, c0))
            output = layer(out[:, -1, :])
            outputs.append(output)
        return torch.cat(outputs, dim=1)
