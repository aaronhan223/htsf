
class MecatsParams:

    hierarchical = 1
    horizon = 1
    verbose = 1
    time_series = 1
    quantile = 1
    online = 0
    metric_mode = True
    path = '../hiertsforecaster'
    unit_test = True
    experts = ['auto_arima', 'deepar', 'fbprophet', 'lstnet', 'pydlm']

    valid_ratio = 0.2
    pred_step = 12

    mid_quantile = 0.5
    other_quantiles = [0.05, 0.3, 0.7, 0.95]

    Lambda = 0.1
    seed = 78712

    n_gpu = 4
    ip_address = '127.0.0.1'
    port = '20000'
    

class GatingNetParams:

    recurrent_hidden_dim = 1
    layer_dim = 3
    batch_size = 16
    linear_hidden_dim = 60
    learning_rate = 1e-4
    num_epochs = 5
    Lambda = 0.1
    lr_decay = 0.1

    early_stop = False
    lr_schedule = False
    

class LstNetParams:

    learning_rate = 1e-3
    optimizer = 'adam'
    h = 1
    normalize = 0
    batch_size = 16
    num_epoch = 5
    model_dir = "../hiertsforecaster/save/lstnet"

    early_stop = False
    lr_schedule = False


class DeeparParams:

    learning_rate = 1e-4
    batch_size = 16
    lstm_layers = 3
    num_epochs = 2
    num_class = 1
    cov_dim = 1
    lstm_hidden_dim = 40
    embedding_dim = 20
    sample_times = 200
    lstm_dropout = 0.5
    predict_batch = 256
    stride = 1
    model_dir = "../hiertsforecaster/save/deepar"
    plot_dir = "../hiertsforecaster/results/deepar_fig"

    early_stop = False
    lr_schedule = False


class PydlmParams:

    trend_degree = 1
    discount = 0.95
    prior_cov = 1e7


class AutoArimaParams:

    seasonal = True
    m = 12
    max_dur = 8

    
class QuantileNetParams:

    d = 16
    num_layers = 6
    layer_dims = [120, 120, 60, 60, 10, 1]
    nonlinear = "relu"
    learning_rate = 1e-4
    num_epochs = 5
    quantiles = [0.95, 0.7, 0.3, 0.05]
    type = "median"

    early_stop = False
    lr_schedule = False