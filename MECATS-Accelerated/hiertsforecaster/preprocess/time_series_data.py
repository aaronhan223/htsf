from hiertsforecaster.preprocess.hierarchical import TreeNodes
from hiertsforecaster.preprocess import train_config
import pdb


class HierTimeSeriesData:
    '''
    A class wrapper for all properties of input time series.
    '''
    def __init__(self, data, nodes):
        self._ts_data = data
        self._nodes = nodes
        self._node_list = TreeNodes(nodes).col_order()
        self._level = len(nodes) + 1
        self._univariate = data.shape[1] == 1
    
    @property
    def ts_data(self):
        return self._ts_data

    @property
    def ts_train_data(self):
        return self._ts_data[:-train_config.MecatsParams.pred_step]

    @property
    def nodes(self):
        return self._nodes
    
    @property
    def node_list(self):
        return self._node_list

    @property
    def level(self):
        return self._level

    @property
    def univariate(self):
        return self._univariate