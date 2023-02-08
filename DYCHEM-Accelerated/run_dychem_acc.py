import numpy as np
import pandas as pd
from tqdm import tqdm
from time import perf_counter
import os
import warnings
import pdb


def main(data, nodes, realm_id):

    from hiertsforecaster.mecats import mecats
    from hiertsforecaster.preprocess import train_config
    start_time = perf_counter()
    mecats_model = mecats(data=data, nodes=nodes, config=train_config, realm_id=realm_id)
    results = mecats_model.fit()
    multilevel_loss, crps = results['mape'], results['crps']

    print(f"Total time: {int(perf_counter() - start_time)}")
    print('Pretrain time: {}'.format(results['pretrain_time']))
    print('Gating Network time: {}'.format(results['fit_moe_time']))
    print('Inference time: {}'.format(results['inf_time']))
    print('Multi-level NRMSE:', multilevel_loss)
    np.savez(f'./hiertsforecaster/save/{realm_id}/nrmse.npz', nrmse=multilevel_loss)
    # print('Multi-level CRPS:', crps)
    print(f'Realm_id {realm_id} done!')


if __name__ == '__main__':
    # example on labour force data
    data = pd.read_csv('./hiertsforecaster/data/labour_force.csv').drop(['Unnamed: 0'], axis=1)
    np.savez('./hiertsforecaster/data/data_length.npz', length=data.shape[0])
    nodes = [[2], [2, 2], [8, 8, 8, 8]]
    realm_id = 10000000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(data, nodes, realm_id)
    os.remove('./hiertsforecaster/data/data_length.npz')