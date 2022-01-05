from itertools import chain
import numpy as np


class TreeNodes:
    '''
    Define hierarchical structure of time series data. Given a particular node in a graph, return its child / levels.
    '''
    def __init__(self, levels, **kwargs):
        self.graph = levels
        if 'name' in kwargs:
            name = kwargs.pop('name')
            self.name = name

    def get_child(self):
        if int(self.name) > sum(list(chain(*self.graph))) - sum(self.graph[-1]):
            return None
        elif self.name == '0':
            return [str(i) for i in range(self.graph[0][0] + 1)]
        else:
            idx, l = self.get_idx(self.name)
            start = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx - 1])
            end = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx])
            return [self.name] + [str(i) for i in range(start + 1, end + 1, 1)]

    def get_idx(self, node):
        total = 0
        for i, l in enumerate(self.graph):
            for e in l:
                total += e
                if total >= int(node):
                    idx = int(node) - sum(list(chain(*self.graph[:i])))
                    return idx, i + 1

    def get_levels(self):
        if self.name == '0':
            return 0
        _, level = self.get_idx(self.name)
        return level

    def col_order(self):
        return [str(i) for i in range(sum(list(chain(*self.graph))) + 1)]

    def nodes_by_level(self, l):
        if l == 1:
            return ['0']
        else:
            start, end = sum(list(chain(*self.graph[:l - 2]))), sum(list(chain(*self.graph[:l - 1])))
            return [str(i) for i in range(start + 1, end + 1)]

    def get_leaf_idx(self, node, aggts):
        if node > sum(list(chain(*self.graph))) - sum(self.graph[-1]):
            return node - aggts
        elif node == 0:
            return [self.get_leaf_idx(i, aggts) for i in range(1, self.graph[0][0] + 1)]
        else:
            idx, l = self.get_idx(node)
            start = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx - 1])
            end = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx])
            return [self.get_leaf_idx(i, aggts) for i in range(start + 1, end + 1, 1)]

    def get_s_matrix(self):
        nts, nbts = sum(list(chain(*self.graph))) + 1, sum(self.graph[-1])
        S = np.zeros((nts, nbts))
        S[-nbts:] = np.eye(nbts)
        aggts = nts - nbts
        for i in range(aggts):
            S[i, self.ravel_multi_d_list(self.get_leaf_idx(i, aggts))] = 1
        return S

    def ravel_multi_d_list(self, lists):
        while isinstance(lists[0], list):
            lists = list(chain(*lists))
        return lists
