'''
Implementation is based on this paper: https://robjhyndman.com/papers/MinT.pdf
'''
import numpy as np
import pandas as pd
from itertools import chain
import pdb


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