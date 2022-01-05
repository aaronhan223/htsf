import numpy as np


def get_multistep_mae(test, pred, h):
    base = 'item_cnt_day'
    feats = [base + '_' + str(i) for i in range(60)]
    top_level_return = np.absolute(np.array(pred[base][-h:]['yhat'] - test[base]))

    # level 1 MAE should not be divided by 60
    level_1_return = np.zeros(h)
    for feat in feats:
        feat_error = np.absolute(np.array(pred[feat][-h:]['yhat'] - test[feat]))
        level_1_return += feat_error
    return top_level_return, level_1_return


def get_multistep_consistency(pred, h):
    base = 'item_cnt_day'
    feats = [base + '_' + str(i) for i in range(60)]
    top_level_pred = np.array(pred[base][-h:]['yhat'])

    level_1_pred = np.zeros(h)
    for feat in feats:
        level_1_pred += np.array(pred[feat][-h:]['yhat'])
    consistency_error = np.absolute(top_level_pred - level_1_pred)
    return consistency_error
