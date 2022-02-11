from itertools import chain
import numpy as np


class TreeNodes:
    '''
    Define hierarchical structure of time series data. Given a particular node in a graph, return its child / levels.
    '''
    def __init__(self, levels, **kwargs):
        self.graph = levels
        self.balance = True
        if 'name' in kwargs:
            name = kwargs.pop('name')
            self.name = name
        if 0 in list(chain(*self.graph)):
            self.balance = False
        if not self.balance:
            self.bottom_tup = self._process_nodes()

    def _process_nodes(self):
        bottom_tup, stack = [], [0, ]
        while stack:
            root = stack.pop()
            if root is not None:
                if self._get_child(root) is not None:
                    childs = self._get_child(root)
                    childs.reverse()
                    for e in childs:
                        stack.append(e)
                else:
                    bottom_tup.append(root)
        return bottom_tup

    def _get_child(self, name):
        if int(name) > sum(list(chain(*self.graph))) - sum(self.graph[-1]):
            return None
        elif name == 0:
            return [i for i in range(1, self.graph[0][0] + 1)]
        else:
            idx, l = self._get_idx(name)
            start = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx - 1])
            end = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx])
            if start == end:
                return None
            return [i for i in range(start + 1, end + 1, 1)]

    def _is_leaf(self, node):
        for idx, n in enumerate(self.bottom_tup):
            if n == node:
                return idx
        return False

    def _get_leaf_idx_unbalance(self, node):
        if not isinstance(self._is_leaf(node), bool):
            return self._is_leaf(node)
        elif node == 0:
            return [self._get_leaf_idx_unbalance(i) for i in range(1, self.graph[0][0] + 1)]
        else:
            idx, l = self._get_idx(node)
            start = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx - 1])
            end = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx])
            return [self._get_leaf_idx_unbalance(i) for i in range(start + 1, end + 1, 1)]

    def get_child(self):
        if int(self.name) > sum(list(chain(*self.graph))) - sum(self.graph[-1]):
            return None
        elif self.name == '0':
            return [str(i) for i in range(self.graph[0][0] + 1)]
        else:
            idx, l = self._get_idx(self.name)
            start = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx - 1])
            end = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx])
            if start == end:
                return None
            return [self.name] + [str(i) for i in range(start + 1, end + 1, 1)]

    def _get_idx(self, node):
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
        _, level = self._get_idx(self.name)
        return level

    def col_order(self):
        return [str(i) for i in range(sum(list(chain(*self.graph))) + 1)]

    def nodes_by_level(self, l):
        if l == 1:
            return ['0']
        else:
            start, end = sum(list(chain(*self.graph[:l - 2]))), sum(list(chain(*self.graph[:l - 1])))
            return [str(i) for i in range(start + 1, end + 1)]

    def _is_flatten(self, lists):
        for lis in lists:
            if isinstance(lis, list):
                return False
        return True

    def _flatten(self, lists):
        list_element = []
        for lis in lists:
            if isinstance(lis, list):
                list_element.append(lis)
                lists.remove(lis)
        for lis in list_element:
            lists += lis
        return lists

    def _get_leaf_idx(self, node, aggts):
        if node > sum(list(chain(*self.graph))) - sum(self.graph[-1]):
            return node - aggts
        elif node == 0:
            return [self._get_leaf_idx(i, aggts) for i in range(1, self.graph[0][0] + 1)]
        else:
            idx, l = self._get_idx(node)
            start = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx - 1])
            end = sum(list(chain(*self.graph[:l]))) + sum(self.graph[l][:idx])
            return [self._get_leaf_idx(i, aggts) for i in range(start + 1, end + 1, 1)]

    def get_s_matrix(self):
        nts, nbts = sum(list(chain(*self.graph))) + 1, sum(self.graph[-1])
        if not self.balance:
            for l in self.graph:
                nbts += l.count(0)
        S = np.zeros((nts, nbts))
        if self.balance:
            S[-nbts:] = np.eye(nbts)
            aggts = nts - nbts
            for i in range(aggts):
                S[i, self._ravel_multi_d_list(self._get_leaf_idx(i, aggts))] = 1
        else:
            for i in range(nts):
                indices = self._get_leaf_idx_unbalance(i)
                if isinstance(indices, list):
                    S[i, self._ravel_multi_d_list(self._get_leaf_idx_unbalance(i))] = 1
                else:
                    S[i, self._get_leaf_idx_unbalance(i)] = 1
        return S

    def _ravel_multi_d_list(self, lists):
        if self.balance:
            while isinstance(lists[0], list):
                lists = list(chain(*lists))
        else:
            while not self._is_flatten(lists):
                lists = self._flatten(lists)
        return lists
