'''
Implementation is based on this paper: https://robjhyndman.com/papers/MinT.pdf
'''
import numpy as np
import pandas as pd
from itertools import chain
from tqdm import tqdm
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects.packages import importr
from preprocess import r_snippet
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import pdb


def recon_base_forecast(node_list, nodes, median_forecast, models, train, h, recon, alg):
    '''
    Original MinT API, but need to install several R packages first.
    '''
    pred = np.zeros((1, sum(list(chain(*nodes))) + 1))
    for name in node_list:
        pred[0, int(name)] = median_forecast[name]

    rpy2.robjects.numpy2ri.activate()
    nr, nc = pred.shape
    pred_r = ro.r.matrix(pred, nrow=nr, ncol=nc)

    feed_dict = {}
    for i, e in enumerate(nodes):
        if len(e) == 1:
            feed_dict['Level ' + str(i)] = e[0]
        else:
            feed_dict['Level ' + str(i)] = ro.IntVector(e)
    node_structure = ro.ListVector(feed_dict)

    if not isinstance(train, pd.DataFrame):
        training = int(train.tr * train.n)
        train_np = train.dat[:training, :]
        residual = np.zeros_like(train_np)
        residual[:h] = train_np[:h]
    else:
        residual = np.zeros_like(train.values)
        residual[:h] = train.values[:h]
    residual = insample_residual(node_list, models, train, residual, h, alg, recon)
    nr, nc = residual.shape
    residual_r = ro.r.matrix(residual, nrow=nr, ncol=nc)

    hts = importr('hts')
    MinT = hts.MinT
    combinef = hts.combinef
    if recon == 'mint_shr':
        mint = r_snippet.mint_shr()
    elif recon == 'mint_sam':
        mint = r_snippet.mint_sam()
    else:
        mint = r_snippet.mint_ols()

    try:
        powerpack = SignatureTranslatedAnonymousPackage(mint, "powerpack")
        recon_pred = powerpack.mint_recon(pred_r, node_structure, residual_r)
    except:
        print('MinT does not apply here as covariance matrix is not positive definite.')
        return median_forecast

    return np.array(recon_pred)


def insample_residual(node_list, models, train, residual, h, alg, recon):
    if alg == 'rnn' or recon == 'sharq':  # need to think about the data input format for lstnet with sharq
        for i in range(h + 1, train.values.shape[0]):
            for name in tqdm(node_list):
                X = models[name].get_x_y(np.array(train[name])[:i])
                residual[i, int(name)] = train.values[i, int(name)] - models[name](X)
    else:
        residual[h:, :] = train.dat[h:int(train.tr * train.n), :] - models(train.train[0].cuda()).cpu().detach().numpy()
    return residual


def mint_py(S, forecast, method='ols', **kwargs):
    '''
    Python version for MinT, necessary in deployment.
    S: 2D array representing the graph structure
    forecast: 1D numpy array representing the base forecast of each node
    method: reconciliation method
    '''
    lambda_1, lambda_2 = max(0.005 * np.var(forecast), 1),  0
    
    minimum_eigen = 10 ** -3
    minimum_threshold = 10 ** -4
    maximum_reconciliation = 0.50 
    maximum_recon_overall = 0.25

    # TODO: there is an exact way to compute opitimal lambda_d, see http://www.strimmerlab.org/publications/journals/shrinkcov2005.pdf
    k_h, lambda_d = 1.0, 0.6

    forecast = np.array(forecast)
    forecast[np.abs(forecast) <= minimum_threshold] = 0
    forecast[np.abs(forecast) <= minimum_threshold] = 0

    # WLS, Sample, and Shrinkage estimator require calculating in-sample residual
    if method == 'ols':
        sigma = k_h * np.eye(S.shape[0])
    elif method == 'mint_wls':
        # this approach needs in-sample residual for the rolling window with size T, the size of base_error should be (number of nodes * window size)
        if 'base_error' in kwargs:
            base_error = kwargs.pop('base_error')
            one_step_error = base_error
        else:
            # in case in-sample residual is not provided, use T = 10 here
            one_step_error = np.random.normal(loc=1, scale=3, size=(S.shape[0], 10))
        W_1 = np.dot(one_step_error, one_step_error.T) / one_step_error.shape[1]
        sigma = k_h * np.diag(np.diag(W_1))
    elif method == 'group':
        Lambda = np.squeeze(np.dot(S, np.ones((S.shape[1], 1))))
        sigma = k_h * np.diag(Lambda)
    elif method == 'mint_sam':
        if 'base_error' in kwargs:
            base_error = kwargs.pop('base_error')
            one_step_error = base_error
        else:
            one_step_error = np.random.normal(loc=1, scale=3, size=(S.shape[0], 10))
        W_1 = np.dot(one_step_error, one_step_error.T) / one_step_error.shape[1]
        sigma = k_h * W_1
    elif method == 'mint_shr':
        if 'base_error' in kwargs:
            base_error = kwargs.pop('base_error')
            one_step_error = base_error
        else:
            one_step_error = np.random.normal(loc=1, scale=3, size=(S.shape[0], 10))
        W_1 = np.dot(one_step_error, one_step_error.T) / one_step_error.shape[1]
        W_1_diag = np.diag(np.diag(W_1))
        sigma = k_h * (lambda_d * W_1_diag + (1 - lambda_d) * W_1)

    if min(np.abs(np.linalg.eig(sigma)[0])) <= minimum_eigen:
        sigma = sigma + lambda_1 * np.eye(sigma.shape[0])
    inv_sigma = np.linalg.inv(sigma)
    inv_eig_mat = np.dot(np.dot(S.T, inv_sigma), S)
    if min(np.abs(np.linalg.eig(inv_eig_mat)[0])) <= minimum_eigen:
        inv_eig_mat = inv_eig_mat + lambda_2 * np.eye(S.shape[1])
    A_1 = np.linalg.inv(inv_eig_mat)    
    A_2 = np.dot(S.T, inv_sigma)
    P = np.dot(A_1, A_2)
    recon_pred = np.dot(S, np.dot(P, forecast))

    if (sum((recon_pred - forecast) ** 2) >= maximum_reconciliation * sum(forecast ** 2) and
        np.abs(forecast[0]) >= maximum_recon_overall * np.abs(recon_pred[0] - forecast[0])):
        recon_pred = forecast
    return np.expand_dims(recon_pred, 0)