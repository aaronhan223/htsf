'''
Author: Xing Han @ UT-Austin
Email: aaronhan223@utexas.edu
'''

import os
import sys
import logging
import torch
import numpy as np
from time import perf_counter
from hiertsforecaster.models.Models import get_moe_optimizers
from hiertsforecaster.recon.learning_moe import MoE_Learner
from hiertsforecaster.evaluation.metrics import compute_level_loss, compute_crps
from hiertsforecaster.preprocess import utils
from hiertsforecaster.preprocess.time_series_data import HierTimeSeriesData
from hiertsforecaster.preprocess.hierarchical import TreeNodes
from hiertsforecaster.recon import MinT
import pdb


logger = logging.getLogger('MECATS.mecats')


class mecats:
    '''
    Main class for running MECATS model.

    data (Pandas Dataframe): Input hierarchical time series data, column name is the vertex name ranging from 0 to N. N is the total number of vertex.
    nodes (List): 2-d array specifying the hierarchical structure of time series data
    plot (Bool): Whether plotting interim results (expert's weight, forecast etc.) during model training
    realm_id (int): id of one user.
    '''
    def __init__(self, data=None, nodes=None, plot=False, config=None, realm_id=None):
        
        self.data = data
        self.nodes = nodes
        self.plot = plot
        self.config = config
        self.realm_id = realm_id
        self.experts = config.MecatsParams.experts
        assert len(self.experts) > 1, 'The number of experts must be greater than one, please use MinT/SHARQ for single expert.'
        self.experts.sort()
        
        head_path = config.MecatsParams.path
        utils.check_directory(plot, config.MecatsParams.unit_test, realm_id)
        utils.set_logger(os.path.join(head_path, 'save', f'{realm_id}', 'history.log'))
        logger.info('\n\n<---------------NEW RUN--------------->')

        self.cuda = torch.cuda.is_available()
        utils.set_seed(self.config.ParallelParams, self.cuda, (self.config.MecatsParams.seed))
        self.hier_ts_data = HierTimeSeriesData(data=self.data, nodes=self.nodes)

    def fit(self):
        start_time = perf_counter()
        self.pretrain()
        pretrain_time = int(perf_counter() - start_time)
        start_time = perf_counter()
        self.learner.fit_moe()
        fit_moe_time = int(perf_counter() - start_time)
        if self.config.MecatsParams.online:
            self.online_training()
        start_time = perf_counter()
        loss, crps = self.predict()
        inf_time = int(perf_counter() - start_time)
        return {'mape': loss, 'crps': crps, 'pretrain_time': pretrain_time, 'fit_moe_time': fit_moe_time, 'inf_time': inf_time}
    
    def pretrain(self):
        logger.info('<------------Pre-train model bank start------------>')
        if self.config.MecatsParams.quantile:
            gns, optimizers, q_nets, q_optimizers = get_moe_optimizers(hier_ts_data=self.hier_ts_data, cuda=self.cuda, plot=self.plot, experts=self.experts, realm_id=self.realm_id)
            self.learner = MoE_Learner(optimizers=optimizers, gns=gns, hier_ts_data=self.hier_ts_data, cuda=self.cuda, plot=self.plot, experts=self.experts, realm_id=self.realm_id, q_nets=q_nets, q_optimizers=q_optimizers)
        else:
            gns, optimizers = get_moe_optimizers(hier_ts_data=self.hier_ts_data, cuda=self.cuda, plot=self.plot, experts=self.experts, realm_id=self.realm_id)
            self.learner = MoE_Learner(optimizers=optimizers, gns=gns, hier_ts_data=self.hier_ts_data, cuda=self.cuda, plot=self.plot, experts=self.experts, realm_id=self.realm_id)

    def online_training(self):
        logger.info('<------------Online update using test data------------>')
        logger.info('Change point status is {}'.format(self.config.OnlineParams.enable_cp))
        assert self.hier_ts_data.univariate, 'Online update can only be applied on univariate time series.'
        assert self.config.metric_mode, 'Online update only works in metric mode.'
        online_params = self.config.online_params
        step_loss_no_cp, l_range = self.learner.online_update(params=online_params)
        
        online_params.enable_cp = 1 - online_params.enable_cp
        logger.info('Change point status is {}'.format(online_params.enable_cp))
        step_loss_cp, _ = self.learner.online_update(params=online_params)
        utils.plot_loss_cp(step_loss_no_cp, step_loss_cp, l_range)
        logger.info('Program done!')
        sys.exit()

    def predict(self):
        logger.info('<------------Prediction on test data------------>')
        result, test, quantile = self.learner.inference()
        loss, crps = None, None

        if self.config.MecatsParams.metric_mode:
            logger.info('Computing multi-level loss and crps')
            loss = compute_level_loss(self.hier_ts_data, result, test)
            if self.config.MecatsParams.quantile:
                crps = compute_crps(test, quantile, self.hier_ts_data)
        if self.config.MecatsParams.unit_test:
            utils.write_data_to_dir(result, quantile, self.config.MecatsParams.metric_mode, loss, crps)
        return loss, crps
