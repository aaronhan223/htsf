import pandas as pd
import numpy as np
from itertools import chain
import pdb


def extract_ts():
    data = pd.read_excel('./data/M3C.xls', sheet_name='M3Month')
    nts = data.shape[0]
    result = None
    for i in range(nts):
        raw_ts = data.iloc[i].values
        ts_length, h, category = raw_ts[1], raw_ts[2], raw_ts[3].replace(' ', '')
        starting_year, starting_month = raw_ts[4], raw_ts[5]
        ts = raw_ts[6: 6 + ts_length]
        if ts_length == 144:
            result = temporal_agg(ts, h)
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


def temporal_agg(data, h):
    data = data[:len(data) - len(data) % 12]
    month = [data[i::12] for i in range(12)]
    quarter = aggregate(month, 3)
    semi_annual = aggregate(quarter, 2)
    annual = aggregate(semi_annual, 2)
    all_ts, cnt = np.zeros((len(annual[0]), 19)), 0
    for ts in chain(annual, semi_annual, quarter, month):
        all_ts[:, cnt] = ts
        cnt += 1
    return all_ts
