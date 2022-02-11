import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pdb


def choose_node(dataset):
    node_list = []
    if dataset == 'labour':
        node_list = ['0', '1', '5', '34']
    elif dataset == 'sales':
        node_list = ['0', '2']
    elif dataset == 'wiki':
        node_list = ['0', '1', '10', '33', '100']
    elif dataset == 'sim_small':
        node_list = ['0', '1', '5']
    return node_list


def ts_plot(data, l, dataset):
    t = np.arange(data.shape[0])
    node_list = choose_node(dataset)
    series = [np.array(data[node]) for node in node_list]
    fig, axes = plt.subplots(l, 1, figsize=(5, 2))
    plt.subplots_adjust(wspace=.3, hspace=0)

    for i, ax in enumerate(axes.flat):
        ax.plot(t, series[i], color='k', linewidth=1)
        # ax.set_title('Level {}'.format(i + 1), fontsize=8)
        ax.grid(False)
        ax.xaxis.set_tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelsize=6)
        if i == l - 1:
            ax.set_xlabel('Time', fontsize=10)
        else:
            ax.set_xticks([])

    if not os.path.exists(os.getcwd() + '/plots'):
        os.mkdir(os.getcwd() + '/plots')
    plt.savefig(os.getcwd() + '/plots/' + dataset + '.pdf')


def forecast_plot(train, pred, dataset):
    t = np.arange(train.shape[0] + len(pred))
    train_display, test_display = 3 * len(pred), len(pred)
    t_train, t_test = t[-(train_display + test_display): -test_display], t[-(test_display + 1):]
    node_list = choose_node(dataset)
    train_plot = [np.array(train[node_list[0]][-train_display:]), np.array(train[node_list[-1]][-train_display:])]

    target_plot = [[train_plot[0][-1]] + [item[node_list[0]][3] for item in pred],
                   [train_plot[1][-1]] + [item[node_list[-1]][3] for item in pred]]

    upper_plot = [[train_plot[0][-1]] + [item[node_list[0]][0] for item in pred],
                  [train_plot[1][-1]] + [item[node_list[-1]][0] for item in pred]]
    lower_plot = [[train_plot[0][-1]] + [item[node_list[0]][2] for item in pred],
                  [train_plot[1][-1]] + [item[node_list[-1]][2] for item in pred]]
    med_plot = [[train_plot[0][-1]] + [item[node_list[0]][1] for item in pred],
                [train_plot[1][-1]] + [item[node_list[-1]][1] for item in pred]]

    fig, axes = plt.subplots(2, 1, figsize=(4, 3))
    plt.subplots_adjust(wspace=.2, hspace=0)
    sns.set_style("whitegrid")
    for i, ax in enumerate(axes.flat):
        ax.plot(t_train, train_plot[i], color='k', linewidth=1)
        ax.plot(t_test, target_plot[i], color='#a5e321', linewidth=1)
        ax.plot(t_test, med_plot[i], color='#21a5e3', linewidth=1)
        ax.fill_between(t_test, y1=upper_plot[i], y2=lower_plot[i], color='#cdebf9')
        if i == 0:
            ax.legend(('Training Samples', 'True Value', 'Predicted Mean', 'Predicted stddev'), loc='upper left',
                      fontsize='small', fancybox=True)
        ax.axvline(x=t_train[-1], color='k', linestyle='-', linewidth=3)
        ax.xaxis.set_tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.set_yticks([])
        if i == 1:
            ax.set_xlabel('Time', fontsize=8)
        else:
            ax.set_xticks([])
    plt.savefig(os.getcwd() + '/plots/' + dataset + '_forecast_temp.pdf')


def forecast_plot_single(train, pred, dataset):
    t = np.arange(train.shape[0] + len(pred))
    train_display, test_display = 3 * len(pred), len(pred)
    t_train, t_test = t[-(train_display + test_display): -test_display], t[-(test_display + 1):]
    train_plot = np.array(train[-train_display:])

    upper_plot = [[train_plot[-1]] + [item['0'][0] for item in pred]]
    lower_plot = [[train_plot[-1]] + [item['0'][2] for item in pred]]
    med_plot = [[train_plot[-1]] + [item['0'][1] for item in pred]]
    target_plot = [[train_plot[-1]] + [item['0'][3] for item in pred]]

    fig = plt.figure(figsize=(4, 3))
    plt.plot(t_train, train_plot, color='k', linewidth=1)
    plt.plot(t_test, target_plot[0], color='#a5e321', linewidth=1)
    plt.plot(t_test, med_plot[0], color='#21a5e3', linewidth=1)
    plt.fill_between(t_test, y1=upper_plot[0], y2=lower_plot[0], color='#cdebf9')
    plt.legend(('Training Samples', 'True Value', 'Predicted Mean', 'Predicted stddev'), loc='upper left',
                fontsize='x-small', fancybox=True)
    plt.axvline(x=t_train[-1], color='k', linestyle='-', linewidth=3)
    plt.tick_params(labelsize=6)
    plt.xlabel('Time', fontsize=8)
    plt.savefig(os.getcwd() + '/plots/' + dataset + '_forecast.pdf')
