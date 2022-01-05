import torch
import numpy as np
from torch.autograd import Variable
import pdb


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, train, valid, cuda, horizon, window, dataset, method, normalize=2):
        self.tr = train
        self.va = valid
        self.cuda = cuda
        self.method = method
        self.feats = dataset.columns.values
        self.P = window
        self.h = horizon
        self.rawdat = dataset.values
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(self.tr * self.n), int((self.tr + self.va) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test_prep[1] * self.scale.expand(self.test_prep[1].size(0), self.m)

        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)
        if self.method == 'sharq':
            self.method = 'base'
        self.test_prep = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):

        # idx_set is the range for end point, window size is the length for looking back
        n = len(idx_set)
        if self.method == 'sharq':
            X = np.zeros((n, self.P, self.m))
            Y = np.zeros((n, self.m))
        else:
            X = torch.zeros((n, self.P, self.m))
            Y = torch.zeros((n, self.m))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            if self.method == 'sharq':
                X[i, :, :] = self.dat[start:end, :]
                Y[i, :] = self.dat[idx_set[i], :]
            else:
                X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
                Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])

        if self.method == 'sharq':
            return [dict(zip(self.feats, np.split(X, X.shape[2], 2))),
                    dict(zip(self.feats, np.split(Y, Y.shape[1], 1)))]
        else:
            return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size
