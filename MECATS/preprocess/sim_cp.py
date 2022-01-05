"""
Simulate 1D sequential data that contains change points.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

n = 2000

def f(x):
    if (x < 0.80) : # offline no changepoint
        y = 3 + 3 * x + 0.1 * np.sin(60 * x) 
    elif 0.8 <= x < 0.9: # online before changepoint
        y = 3 + 3 * x + 0.1 * x ** 2 * np.sin(60 * x) 
    else: # online after changepoint
        y = 10 + 5 * x ** 4 + np.sin(60 * x) 
    return y

def get_cp_ts():
    np.random.seed(0)
    x_train = np.random.random(size=(n, 1))
    x_train = np.sort(x_train, axis=0)
    y_train = np.apply_along_axis(f, 1, x_train).reshape(-1, 1) + np.random.normal(0.0, 0.05, size=x_train.shape)
    plt.plot(np.arange(len(y_train)), y_train)
    plt.savefig('./plots/sim_cp.pdf')
    plt.close()
    return y_train
