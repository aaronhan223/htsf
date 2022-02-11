import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
from   scipy.stats import norm
from   scipy.special import logsumexp
import pdb


class GaussianUnknownMean:
    
    def __init__(self, mean0, var0, varx):
        """Initialize model.
        
        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])
    
    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds  = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)
    
    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params  = self.prec_params + (1 / self.varx)
        self.prec_params = np.append([1 / self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params  = (self.mean_params * self.prec_params[:-1] + (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1. / self.prec_params + self.varx


def is_changepoint(x, t, model, log_R, log_message, log_H, log_1mH):

    # 3. Evaluate predictive probabilities.
    log_pis = model.log_pred_prob(t, x)

    # 4. Calculate growth probabilities.
    log_growth_probs = log_pis + log_message + log_1mH

    # 5. Calculate changepoint probabilities.
    log_cp_prob = logsumexp(log_pis + log_message + log_H)

    # 6. Calculate evidence
    new_log_joint = np.append(log_cp_prob, log_growth_probs)

    # 7. Determine run length distribution.
    log_R[t, :t+1] = new_log_joint
    log_R[t, :t+1] -= logsumexp(new_log_joint)

    is_cp = False
    if np.argmax(np.exp(log_R[t])) == 1 and t != 1:
        is_cp = True
    # 8. Update sufficient statistics.
    model.update_params(t, x)

    # Pass message.
    log_message = new_log_joint

    return is_cp


def bocd(x, t, model, log_R, log_message, log_H, log_1mH, pmean, pvar):

    # Make model predictions.
    pmean[t - 1] = np.sum(np.exp(log_R[t-1, :t]) * model.mean_params[:t])
    pvar[t - 1] = np.sum(np.exp(log_R[t-1, :t]) * model.var_params[:t])

    # 3. Evaluate predictive probabilities.
    log_pis = model.log_pred_prob(t, x)

    # 4. Calculate growth probabilities.
    log_growth_probs = log_pis + log_message + log_1mH

    # 5. Calculate changepoint probabilities.
    log_cp_prob = logsumexp(log_pis + log_message + log_H)

    # 6. Calculate evidence
    new_log_joint = np.append(log_cp_prob, log_growth_probs)

    # 7. Determine run length distribution.
    log_R[t, :t + 1] = new_log_joint
    log_R[t, :t + 1] -= logsumexp(new_log_joint)

    # 8. Update sufficient statistics.
    model.update_params(t, x)

    # Pass message.
    log_message = new_log_joint

    return pmean, pvar
