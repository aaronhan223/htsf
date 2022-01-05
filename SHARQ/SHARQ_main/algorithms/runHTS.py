import pandas as pd
from algorithms.htsprophet.hts import hts
from evaluation.labour import get_multistep_consistency, get_multistep_mae
import datetime
import numpy as np
import pdb


def hts_train_and_pred(train, test, steps_ahead, node_list):
    # conduct in-sample forecast here, and measure the MAE and consistency error
    myDict_ols = hts(y=train, h=steps_ahead, nodes=[[2], [2, 2], [8, 8, 8, 8]], method="OLS")
    level_loss_ols = get_multistep_mae(test=test, pred=myDict_ols, node_list=node_list, h=steps_ahead)
    consistency_err_ols = get_multistep_consistency(node_list=node_list, test_pred=myDict_ols, h=steps_ahead)

    myDict_wlss = hts(y=train, h=steps_ahead, nodes=[[2], [2, 2], [8, 8, 8, 8]], method="WLSS")
    level_loss_wlss = get_multistep_mae(test=test, pred=myDict_wlss, node_list=node_list, h=steps_ahead)
    consistency_err_wlss = get_multistep_consistency(node_list=node_list, test_pred=myDict_wlss, h=steps_ahead)

    myDict_wlsv = hts(y=train, h=steps_ahead, nodes=[[2], [2, 2], [8, 8, 8, 8]], method="WLSV")
    level_loss_wlsv = get_multistep_mae(test=test, pred=myDict_wlsv, node_list=node_list, h=steps_ahead)
    consistency_err_wlsv = get_multistep_consistency(node_list=node_list, test_pred=myDict_wlsv, h=steps_ahead)

    myDict_fp = hts(y=train, h=steps_ahead, nodes=[[2], [2, 2], [8, 8, 8, 8]], method="FP")
    level_loss_fp = get_multistep_mae(test=test, pred=myDict_fp, node_list=node_list, h=steps_ahead)
    consistency_err_fp = get_multistep_consistency(node_list=node_list, test_pred=myDict_fp, h=steps_ahead)

    myDict_pha = hts(y=train, h=steps_ahead, nodes=[[2], [2, 2], [8, 8, 8, 8]], method="PHA")
    level_loss_pha = get_multistep_mae(test=test, pred=myDict_pha, node_list=node_list, h=steps_ahead)
    consistency_err_pha = get_multistep_consistency(node_list=node_list, test_pred=myDict_pha, h=steps_ahead)

    myDict_ahp = hts(y=train, h=steps_ahead, nodes=[[2], [2, 2], [8, 8, 8, 8]], method="AHP")
    level_loss_ahp = get_multistep_mae(test=test, pred=myDict_ahp, node_list=node_list, h=steps_ahead)
    consistency_err_ahp = get_multistep_consistency(node_list=node_list, test_pred=myDict_ahp, h=steps_ahead)

    myDict_bu = hts(y=train, h=steps_ahead, nodes=[[2], [2, 2], [8, 8, 8, 8]], method="BU")
    level_loss_bu = get_multistep_mae(test=test, pred=myDict_bu, node_list=node_list, h=steps_ahead)
    consistency_err_bu = get_multistep_consistency(node_list=node_list, test_pred=myDict_bu, h=steps_ahead)
    
    print("OLS: ")
    print("Loss in each level: {}, \n average in each level: {}, \n Consistency: {}.".format(level_loss_ols, dict(
        zip(range(8), np.mean(np.array(list(level_loss_ols.values())), axis=1))), consistency_err_ols))
    print("------------------")

    print("WLSS: ")
    print("Loss in each level: {}, \n average in each level: {}, \n Consistency: {}.".format(level_loss_wlss, dict(
        zip(range(8), np.mean(np.array(list(level_loss_wlss.values())), axis=1))), consistency_err_wlss))
    print("------------------")

    print("WLSV: ")
    print("Loss in each level: {}, \n average in each level: {}, \n Consistency: {}.".format(level_loss_wlsv, dict(
        zip(range(8), np.mean(np.array(list(level_loss_wlsv.values())), axis=1))), consistency_err_wlsv))
    print("------------------")

    print("FP: ")
    print("Loss in each level: {}, \n average in each level: {}, \n Consistency: {}.".format(level_loss_fp, dict(
        zip(range(8), np.mean(np.array(list(level_loss_fp.values())), axis=1))), consistency_err_fp))
    print("------------------")

    print("PHA: ")
    print("Loss in each level: {}, \n average in each level: {}, \n Consistency: {}.".format(level_loss_pha, dict(
        zip(range(8), np.mean(np.array(list(level_loss_pha.values())), axis=1))), consistency_err_pha))
    print("------------------")

    print("AHP: ")
    print("Loss in each level: {}, \n average in each level: {}, \n Consistency: {}.".format(level_loss_ahp, dict(
        zip(range(8), np.mean(np.array(list(level_loss_ahp.values())), axis=1))), consistency_err_ahp))
    print("------------------")

    print("BU: ")
    print("Loss in each level: {}, \n average in each level: {}, \n Consistency: {}.".format(level_loss_bu, dict(
        zip(range(8), np.mean(np.array(list(level_loss_bu.values())), axis=1))), consistency_err_bu))
    print("------------------")
