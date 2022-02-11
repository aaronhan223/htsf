import pandas as pd
import numpy as np
from itertools import chain


def get_agg_data(data):
    agg_1 = ['en', 'fr', 'de', 'ja', 'ru', 'zh']
    agg_2 = ['AAC', 'DES', 'MOB']
    agg_3 = ['AAG', 'SPD']
    agg_4 = [str('%04d' % i) for i in range(1, 151)]
    level_1 = agg_1
    level_2 = [i + j for i in agg_1 for j in agg_2]
    level_3 = [i + j + k for i in agg_1 for j in agg_2 for k in agg_3]
    level_4 = [i + j + k + l for i in agg_1 for j in agg_2 for k in agg_3 for l in agg_4]
    all_feats = data.columns.values[1:]
    full_data = pd.DataFrame()
    full_data['0'] = np.sum(data.values[:, 1:], axis=1)

    cnt = 0
    for feat in chain(level_1, level_2, level_3, level_4):
        selected_feats = all_feats[np.flatnonzero(np.core.defchararray.find(all_feats.tolist(), feat) != -1)]
        if len(selected_feats) == 0:
            continue
        full_data[str(cnt + 1)] = np.sum(data[selected_feats].values, axis=1)
        cnt += 1

    nodes = [[len(agg_1)]]
    level_1_feats, level_2_feats = get_feats(all_feats, level_1), get_feats(all_feats, level_2)
    level_3_feats, level_4_feats = get_feats(all_feats, level_3), get_feats(all_feats, level_4)
    nodes.append(get_nodes_by_level(level_1_feats, level_2_feats))
    nodes.append(get_nodes_by_level(level_2_feats, level_3_feats))
    nodes.append(get_nodes_by_level(level_3_feats, level_4_feats))
    return full_data, nodes


def get_feats(all_feats, level):
    selected_feats = []
    for feat in level:
        selected_feat = all_feats[np.flatnonzero(np.core.defchararray.find(all_feats.tolist(), feat) != -1)]
        if len(selected_feat) != 0:
            selected_feats.append(feat)
    return selected_feats


def get_nodes_by_level(upper_level, lower_level):
    lower_level, nodes = np.array(lower_level), []
    for feat in upper_level:
        selected_feat = lower_level[np.flatnonzero(np.core.defchararray.find(lower_level, feat) != -1)]
        nodes.append(len(selected_feat))
    return nodes
