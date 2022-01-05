# SHARQ

This repository is an implementation of the [paper](https://arxiv.org/abs/2102.12612) **Simultaneously Reconciled Quantile Forecasting of Hierarchically Related Time Series**, AISTATS 2021.

## Synopsis

### Requirements
The following packages are required to run the program:
- Pytorch 1.0+.
- Pandas
- rpy2
- matplotlib
- tqdm

### Running Instructions

#### Create running environment
To setup conda environment for running the program, use the following command:
```
conda env create -f htsf.yml
```
Activate the new environment:
```
conda activate htsf
```
Verify that the new environment was installed correctly:
```
conda env list
```

## Implementation Details
The program compares forecasting performance across benchmarked HTS algorithms on various real-world and simulated hierarchiclly related time series data.
The data set contains both temporal and cross-sectional hierarchies.
### Data set
- [Australian Labour Force](https://www.abs.gov.au/ausstats/abs@.nsf/mf/6202.0)
- [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)
- [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
- [M3 Competition Data](https://forecasters.org/resources/time-series-data/m3-competition/)
- [AEdemand Data](https://cran.r-project.org/web/packages/thief/thief.pdf)

<!-- ### Forecasting Algorithms
- Long Short Term Networks (LSTNet)
- Deep Autoregressive Models (DeepAR)
- Facebook Prophet
- PyDLM
- Auto-ARIMA
- Recurrent Neural Networks (RNN) -->

### Reconciliation Methods
- Bottom up (BU) method.
- [Trace Minimization (MinT)](https://robjhyndman.com/papers/MinT.pdf) including shrinkage, sampling and OLS estimators.
- [Empirical risk minimization (ERM)](http://souhaib-bentaieb.com/pdf/2019_KDD.pdf).
- [hts prophet](https://github.com/CollinRooney12/htsprophet).
- SHARQ (our method).

### Program Structure
- [`run_sharq.py`](run_sharq.py): script to run the sharq algorithm along with other models.
- [`sharq.py`](sharq.py): the wrapper of hts algorithms and data sets.
- [`algorithms`](algorithms): implementation of the list of forecasting models and reconciliation methods.
- [`data`](data): hierarchical time series data sets.
- [`preprocess`](preprocess): preprocess raw time series data from the web, define hierarchical graph structure, etc.
- [`evaluation`](evaluation): evaluation metrics and visualization for out of sample forecasting.

## Config
To change the reconciliation method, assign TRAINING_METHOD with different input string. The list
of available method is 
```
[‘sharq’, ‘base’, ‘mint_shr’, ‘mint_sam’, ‘mint_ols’, ‘erm’, ‘BU’].
```
To change the forecasting algorithm and its hyper-parameters, assign MODEL_HYPER_PARAMS with a
dictionary that contains model information, for example:
```
MODEL_HYPER_PARAMS = {‘alg’: ‘rnn’, ‘num_epoch’: 1000, ‘lr’: 0.1, ‘hidden_dim’: 5, ‘layer_dim’: 2,
‘nonlinearity’: ‘tanh’}.
```
You can either specify all the above hyper-parameters or only one of them.
## Troubleshooting
If you have further questions on implementation details, please contact aaronhan223@utexas.edu
