# MECATS

This repository is an implementation of the [paper](https://arxiv.org/pdf/2112.11669.pdf) **Mixture-of-Experts for Quantile Forecasts of
Aggregated Time Series**.

## Running Instructions
Run MECATS using the following command:
```
python run_mecats.py
```
The following are related files of the program:
 - All quantitative results, hyper-parameters of experiments, and historical records are saved in the log file under `save`
 - All figures shown in the paper are saved under `results`
 - All configurable parameters can be found under [`parameters`](./parameters)
 - Dataset used for experiment can be found under [`data`](./data)

## Config
To change the reconciliation method, modify RECON in [`mecats.json`](./parameters/mecats.json). Avaiable choices:
```
["sharq", "base", "mint_shr", "mint_sam", "mint_wls", "ols"].
```
Other stuffs in mecats.json:
 - QUANTILE: enable/disable (1/0) quantile uncertainty wrapper
 - ONLINE: enable/disable (1/0) online updating in general sequential data

## Troubleshooting
If you have further questions on implementation details, please contact aaronhan223@utexas.edu
