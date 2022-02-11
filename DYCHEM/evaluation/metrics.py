import numpy as np
from preprocess.hierarchical import TreeNodes
from operator import add
import logging
import properscoring as ps
import pdb

logger = logging.getLogger('HTSF.labour')

# the first three functions are for htsprophet evaluation
def consistency_error_multistep(parent, childs, test_pred, h):
    '''
    Calculate the absolute difference between parent and child forecasts.
    :param parent: name of parent node
    :param childs: name of child nodes
    :param test_pred: forecasting results for each node
    :return: consistency error for a single parent-childs group
    '''
    child_pred = [0] * h
    for child in childs:
        child_pred = list(map(add, child_pred, test_pred[child][-h:]['yhat']))
    consistency_error = np.absolute(np.array(test_pred[parent][-h:]['yhat']) - child_pred)
    return consistency_error


def get_multistep_consistency(node_list, test_pred, h):
    '''
    Get consistency error for the whole graph.
    :param node_list: name of all nodes
    :param test_pred: forecasting results for each node
    :return: total consistency error for the graph.
    '''
    total_error = [0] * h
    for name in node_list:
        node = TreeNodes(name)
        if node.get_child() is not None:
            total_error = list(map(add, total_error, consistency_error_multistep(name, node.get_child()[1:], test_pred, h)))
    return total_error


def get_multistep_mae(test, pred, node_list, h):
    level_loss = dict(zip(range(4), [[0] * h] * 4))
    for name in node_list:
        node = TreeNodes(name)
        loss = np.absolute(np.array(pred[name][-h:]['yhat'] - test[name]))
        level_loss[node.get_levels()] = list(map(add, loss, level_loss[node.get_levels()]))
    return level_loss


def consistency_error(parent, childs, test_pred):
    '''
    Calculate the absolute difference between parent and child forecasts.
    :param parent: name of parent node
    :param childs: name of child nodes
    :param test_pred: forecasting results for each node
    :return: consistency error for a single parent-childs group
    '''
    child_pred = 0
    for child in childs:
        child_pred += test_pred[child]
    consistency_error = np.mean(np.absolute(test_pred[parent] - child_pred))
    return consistency_error


def get_consistency(nodes, node_list, test_pred):
    total_error = 0
    for name in node_list:
        node = TreeNodes(nodes, name=name)
        if node.get_child() is not None:
            total_error += consistency_error(name, node.get_child()[1:], test_pred)
    return total_error


def compute_crps(targets, quantiles, node_list, nodes, l):
    level_loss = dict(zip(range(l), np.zeros(l)))
    for col in node_list:
        node = TreeNodes(nodes, name=col)
        y, tau = np.array(targets[col]), quantiles[col]
        crps_score = np.mean(ps.crps_ensemble(y, tau))
        level_loss[node.get_levels()] += float(crps_score)
    for l in range(1, len(level_loss) + 1):
        level_loss[l - 1] = level_loss[l - 1] / len(TreeNodes(nodes).nodes_by_level(l))
    return np.fromiter(level_loss.values(), dtype=float)
    

def compute_level_loss(node_list, nodes, pred, test, l):
    level_loss = dict(zip(range(l), np.zeros(l)))
    for name in node_list:
        node = TreeNodes(nodes, name=name)
        up, lo = np.absolute(np.array(test[name] - pred[name])), np.absolute(np.array(test[name]))
        length = len(up)
        loss = (100 / length) * np.sum(np.divide(up, lo))
        level_loss[node.get_levels()] += loss
    for l in range(1, len(level_loss) + 1):
        level_loss[l - 1] = level_loss[l - 1] / len(TreeNodes(nodes).nodes_by_level(l))
    return np.fromiter(level_loss.values(), dtype=float)


def get_recon_error(node_list, nodes, median, recon, **kwargs):
    if 'upper_pred' in kwargs:
        upper_pred = kwargs.pop('upper_pred')
        lower_pred = kwargs.pop('lower_pred')
        logger.info("Median reconciliation error is {}".format(get_consistency(nodes, node_list, median)))
        logger.info("Upper quantile reconciliation error is {}".format(get_consistency(nodes, node_list, upper_pred)))
        logger.info("Lower quantile reconciliation error is {}".format(get_consistency(nodes, node_list, lower_pred)))
    else:
        logger.info('[{}] Consistency loss is {}'.format(recon, get_consistency(nodes, node_list, median)))

