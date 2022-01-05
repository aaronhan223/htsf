import torch
import torch.nn as nn
from itertools import chain
from torch.autograd import Variable
import numpy as np

cuda = torch.cuda.is_available()


class AutoRegressive(nn.Module):
    def __init__(self, quantiles, p=2, hidden=10):
        super(AutoRegressive, self).__init__()
        # FIXME: make sure p <= H?
        self.p = p
        self.quantiles = quantiles
        linear_layers = [nn.Linear(p, hidden) for _ in range(len(self.quantiles))]
        relu = [nn.ReLU() for _ in range(len(self.quantiles))]
        final_layers = [nn.Linear(hidden, 1) for _ in range(len(self.quantiles))]
        self.linear_layers = nn.ModuleList(linear_layers)
        self.relu = nn.ModuleList(relu)
        self.final_layers = nn.ModuleList(final_layers)
        self.init_weights()

    def init_weights(self):
        for m in chain(self.linear_layers):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def get_x_y(self, data, target=None):
        X = torch.Tensor(data)
        X = torch.unsqueeze(X, 0)
        if cuda:
            X = Variable(X.cuda())
        if target is None:
            return X
        if cuda:
            y = torch.tensor(np.array(target, dtype=np.float32), requires_grad=False).cuda()
        else:
            y = torch.tensor(np.array(target, dtype=np.float32), requires_grad=False)
        return X, y

    def forward(self, x):
        input = x[:, -self.p - 1: -1]
        output = []
        for layer_1, relu, layer_2 in zip(self.linear_layers, self.relu, self.final_layers):
            output.append(layer_2(relu(layer_1(input))))
        return torch.cat(output, dim=1)
