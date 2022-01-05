import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Model(nn.Module):
    def __init__(self, data, method=None, quantiles=None):
        super(Model, self).__init__()
        self.use_cuda = True
        self.quantiles = quantiles
        self.P = 24 * 7
        if method == 'sharq':
            self.m = 1
        else:
            self.m = data.m
        self.hidR = 50
        self.hidC = 50
        self.hidS = 5
        self.Ck = 6
        self.skip = 24
        self.pt = (self.P - self.Ck) / self.skip
        self.hw = 24
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=0.2)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        # if (args.output_fun == 'sigmoid'):
        self.output = F.sigmoid
        if self.quantiles is not None:
            final_layers = [nn.Linear(self.hidR + self.skip * self.hidS, self.m) for _ in range(len(self.quantiles))]
            self.final_layers = nn.ModuleList(final_layers)
        # if (args.output_fun == 'tanh'):
        #     self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt) * self.skip:].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(int(self.pt), batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)
        # add multiple quantile here
        res = []
        if self.quantiles is not None:
            for layer in self.final_layers:
                res.append(layer(r))
        else:
            res = self.linear1(r)

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            if self.quantiles is not None:
                res = [e + z for e in res]
            else:
                res = res + z

        if (self.output):
            if self.quantiles is not None:
                res = [self.output(e) for e in res]
                return torch.cat(res, dim=1)
            else:
                res = self.output(res)
                return res
