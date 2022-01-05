import pandas as pd
import numpy as np
from itertools import chain
import pdb

ts_name = 'Total Emergency Admissions'


def get_ts():
    data = pd.read_csv('./data/AEdemand.csv')
    data = data[data['key'] == ts_name]
    data.index = data['index']
    data.drop(['index', 'key'], axis=1, inplace=True)
    ts = np.squeeze(data.values)
    result = temporal_agg(ts)
    return result


def aggregate(data, n):
    res, cnt, cum_ts = [], 0, np.zeros_like(data[0])
    for ts in data:
        cum_ts += ts
        cnt += 1
        if cnt == n:
            res.append(cum_ts)
            cnt = 0
            cum_ts = np.zeros_like(data[0])
    return res


def temporal_agg(data):
    data = data[:len(data) - len(data) % 12]
    week = [data[i::12] for i in range(12)]
    biweek = aggregate(week, 2)
    month = aggregate(biweek, 2)
    quarter = aggregate(month, 3)
    all_ts, cnt = np.zeros((len(quarter[0]), 22)), 0
    for ts in chain(quarter, month, biweek, week):
        all_ts[:, cnt] = ts
        cnt += 1
    return all_ts
