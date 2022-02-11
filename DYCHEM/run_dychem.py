from mecats import mecats
from preprocess import utils
import os

params = utils.Params(os.path.join('parameters', 'dychem.json'))
mecats_model = mecats(DATASET=params.DATASET, 
                      FORECAST_GRANULARITY=params.FORECAST_GRANULARITY,
                      IS_HIERARCHICAL=params.IS_HIERARCHICAL,
                      FORECAST_HORIZON=params.FORECAST_HORIZON,
                      VERBOSE=params.VERBOSE, 
                      RECON=params.RECON, 
                      MODELS=params.MODELS,
                      DATA=params.DATA,
                      IF_TIME_SERIES=params.IF_TIME_SERIES,
                      HIERARCHY_GRAPH=params.HIERARCHY_GRAPH,
                      CATEGORICAL_FEATURES=params.CATEGORICAL_FEATURES,
                      QUANTILE=params.QUANTILE,
                      ONLINE=params.ONLINE,
                      GPU=params.GPU,
                      SEED=params.SEED)
mecats_model.fit_and_predict()
