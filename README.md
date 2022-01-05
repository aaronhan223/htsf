# Hierarchical Time Series Forecasting

This repository contains implementations of hierarchical time series forecasting methods in the following papers:

- [Simultaneously Reconciled Quantile Forecasting of Hierarchically Related Time Series](https://arxiv.org/abs/2102.12612)
- [Mixture-of-Experts for Quantile Forecasts of Aggregated Time Series](https://arxiv.org/pdf/2112.11669.pdf)

## Running Instructions

To setup conda environment for running the program, use the following command:
```
conda env create -f htsf.yml
```
Activate the new environment:
```
conda activate htsf
```

## Implementation Details
This repository also compares forecasting performance across benchmarked HTS algorithms on various real-world and simulated hierarchiclly related time series data.

### Datasets
- [Australian Labour Force](https://www.abs.gov.au/ausstats/abs@.nsf/mf/6202.0)
- [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)
- [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
- [M3 Competition Data](https://forecasters.org/resources/time-series-data/m3-competition/)
- [AEdemand Data](https://cran.r-project.org/web/packages/thief/thief.pdf)

### Reconciliation Methods
- Bottom up (BU) method.
- [Trace Minimization (MinT)](https://robjhyndman.com/papers/MinT.pdf) including shrinkage, sampling and OLS estimators.
- [Empirical risk minimization (ERM)](https://souhaib-bentaieb.com/papers/2019_kdd_hts_reg.pdf).
- [hts prophet](https://github.com/CollinRooney12/htsprophet).