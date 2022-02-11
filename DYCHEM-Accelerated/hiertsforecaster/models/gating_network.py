import torch
import torch.nn as nn
import torch.nn.functional as F


class Gating_Network(nn.Module):
    '''
    Gating network with recurrent and fully connected layer.
    '''
    def __init__(self, input_dim, params, experts):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, params.recurrent_hidden_dim, params.layer_dim, batch_first=True)
        self.l1 = nn.Linear(params.batch_size, params.linear_hidden_dim)
        self.nonlinear = nn.Tanh()
        self.l2 = nn.Linear(params.linear_hidden_dim, len(experts))

    def forward(self, data):
        o1 = self.lstm(data)[0][:, -1, :]
        o2 = self.l1(torch.t(o1))
        o3 = self.nonlinear(o2)
        o4 = self.l2(o3)
        return F.softmax(o4, dim=1)