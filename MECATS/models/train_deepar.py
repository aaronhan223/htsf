import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from preprocess import utils
from torch.utils.data import DataLoader
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Train')


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          test_loader: DataLoader,
          params: utils.Params,
          epoch: int) -> float:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    # Train_loader:
    # train_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
    # idx ([batch_size]): one integer denoting the time series id;
    # labels_batch ([batch_size, train_window]): z_{1:T}.
    for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]

        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(params.device)  # not scaled
        labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(params.device)  # not scaled
        idx = idx.unsqueeze(0).to(params.device)

        loss = torch.zeros(1, device=params.device)
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        for t in range(params.train_window):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (train_batch[t, :, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                train_batch[t, zero_index, 0] = mu[zero_index]
            mu, sigma, hidden, cell = model(train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell)
            loss += loss_fn(mu, sigma, labels_batch[t])

        loss.backward()
        optimizer.step()
        loss = loss.item() / params.train_window  # loss per timestep
        loss_epoch[i] = loss
        if i % 1000 == 0:
            test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=False)
            model.train()
            logger.info(f'train_loss: {loss}')
        if i == 0:
            logger.info(f'train_loss: {loss}')
    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer: optim, loss_fn,
                       params: utils.Params,
                       restore_file: str = None) -> None:
    '''Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        test_loader: load test data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    logger.info('begin training and evaluation')
    best_test_ND = float('inf')
    train_len = len(train_loader)
    ND_summary = np.zeros(params.num_epochs)
    loss_summary = np.zeros((train_len * params.num_epochs))
    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader,
                                                                        test_loader, params, epoch)
        test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=False)
        ND_summary[epoch] = test_metrics['ND']
        is_best = ND_summary[epoch] <= best_test_ND

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=is_best,
                              dataset=params.dataset,
                              method=params.method,
                              checkpoint=params.model_dir) # modify this to a different repo

        if is_best:
            logger.info('- Found new best ND')
            best_test_ND = ND_summary[epoch]
            best_json_path = os.path.join(params.model_dir, 'metrics_test_best_weights_{}.json'.format(params.dataset))
            utils.save_dict_to_json(test_metrics, best_json_path)

        logger.info('Current Best ND is: %.5f' % best_test_ND)

        utils.plot_all_epoch(ND_summary[:epoch + 1], params.dataset + '_ND', params.plot_dir)
        utils.plot_all_epoch(loss_summary[:(epoch + 1) * train_len], params.dataset + '_loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir, 'metrics_test_last_weights_{}.json'.format(params.dataset))
        utils.save_dict_to_json(test_metrics, last_json_path)


def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():
      plot_batch = np.random.randint(len(test_loader)-1)

      summary_metric = {}
      raw_metrics = utils.init_metrics(sample=sample)

      # Test_loader: 
      # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
      # id_batch ([batch_size]): one integer denoting the time series id;
      # v ([batch_size, 2]): scaling factor for each window;
      # labels ([batch_size, train_window]): z_{1:T}.
      for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
          test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
          id_batch = id_batch.unsqueeze(0).to(params.device)
          v_batch = v.to(torch.float32).to(params.device)
          labels = labels.to(torch.float32).to(params.device)
          batch_size = test_batch.shape[1]
          input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          hidden = model.init_hidden(batch_size)
          cell = model.init_cell(batch_size)

          for t in range(params.test_predict_start):
              # if z_t is missing, replace it by output mu from the last time step
              zero_index = (test_batch[t,:,0] == 0)
              if t > 0 and torch.sum(zero_index) > 0:
                  test_batch[t,zero_index,0] = mu[zero_index]
              
              mu, sigma, hidden, cell = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell)
              input_mu[:,t] = v_batch[:, 0] * mu + v_batch[:, 1]
              input_sigma[:,t] = v_batch[:, 0] * sigma
            # TODO: check sample_mu and input_mu, these are predictions
          if sample:
              samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling=True)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
          else:
              sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)

          if i == plot_batch:
              if sample:
                  sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
              else:
                  sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)                
              # select 10 from samples with highest error and 10 from the rest
              top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // 10]  # hard coded to be 10
              chosen = set(top_10_nd_sample.tolist())
              all_samples = set(range(batch_size))
              not_chosen = np.asarray(list(all_samples - chosen))
              if batch_size < 100: # make sure there are enough unique samples to choose top 10 from
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
              else:
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
              if batch_size < 12: # make sure there are enough unique samples to choose bottom 90 from
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
              else:
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
              combined_sample = np.concatenate((random_sample_10, random_sample_90))

              label_plot = labels[combined_sample].data.cpu().numpy()
              predict_mu = sample_mu[combined_sample].data.cpu().numpy()
              predict_sigma = sample_sigma[combined_sample].data.cpu().numpy()
              plot_mu = np.concatenate((input_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
              plot_sigma = np.concatenate((input_sigma[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
              plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}
              plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, params.valid_window, params.test_predict_start, plot_num, plot_metrics, sample)

      summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
      metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
      logger.info('- Full test metrics: ' + metrics_string)
    return summary_metric


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                         predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                         alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        #metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})


        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()
