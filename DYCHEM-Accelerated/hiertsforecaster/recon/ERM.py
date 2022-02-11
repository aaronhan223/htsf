import numpy as np
from numpy.linalg import inv
from hiertsforecaster.preprocess.hierarchical import TreeNodes
import pdb


def get_p_matrix(S, Y, Y_hat):
    P = -np.dot(np.dot(inv(np.dot(S.T, S)), np.dot(S.T, Y.T)), np.dot(Y_hat, inv(np.dot(Y_hat.T, Y_hat))))
    return P


def unbiased_recon(nodes, y, y_hat):
    S = TreeNodes(nodes).get_s_matrix()
    P = get_p_matrix(S, y, y_hat)
    result = []
    for i in range(y_hat.shape[0]):
        recon_pred = {}
        y_tilde = np.dot(np.dot(S, P), y_hat[i, :])
        for i, e in enumerate(y_tilde):
            recon_pred[str(i)] = e
        if y_hat.shape[0] == 1:
            return recon_pred
        result.append(recon_pred)
    return result
