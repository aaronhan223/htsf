from pdb import set_trace
import numpy as np
import os


def get_data_length():
    return np.load('./hiertsforecaster/data/data_length.npz')['length']


class MecatsParams:

    hierarchical = 1
    horizon = 1
    verbose = 1
    time_series = 1
    quantile = 0
    online = 0
    metric_mode = False
    path = './hiertsforecaster'
    unit_test = False
    experts = ['auto_arima', 'average', 'fbprophet', 'pydlm']

    valid_ratio = 0.1
    pred_step = 546

    mid_quantile = 0.5
    other_quantiles = [0.05, 0.3, 0.7, 0.95]

    Lambda = 0.1
    seed = 78712
    

class ParallelParams:

    n_gpu = 4
    n_nodes = 1
    nr = 0
    world_size = n_nodes * n_gpu
    ip_address = '127.0.0.1'
    port = str(np.random.randint(low=15000, high=25000))


def get_window_length():
    return int(1.1 * MecatsParams.valid_ratio * get_data_length())


def get_batch_size():
    return int(len(range(get_window_length(), int((1 - MecatsParams.valid_ratio) * (get_data_length() - MecatsParams.pred_step)))) / 4)


class DeeparParams:

    def __init__(self) -> None:
        pass
    @property
    def learning_rate(self):
        return 1e-4
    @property
    def batch_size(self):
        return 16
    @property
    def lstm_layers(self):
        return 3
    @property
    def num_epochs(self):
        return 10
    @property
    def num_class(self):
        return 1
    @property
    def cov_dim(self):
        return 1
    @property
    def lstm_hidden_dim(self):
        return 40
    @property
    def embedding_dim(self):
        return 20
    @property
    def sample_times(self):
        return 20
    @property
    def lstm_dropout(self):
        if MecatsParams.unit_test:
            return 0
        else:
            return 0.1
    @property
    def predict_batch(self):
        return 256
    @property
    def stride(self):
        return 1     
    @property
    def val(self):
        return 0.1 
    @property
    def relative_metrics(self):
        return False 
    @property
    def train_window(self):
        return get_window_length()
    # @train_window.setter
    # def train_window(self):
    #     self.train_window = get_window_length()
    @property
    def valid_window(self):
        return get_window_length() 
    @property
    def predict_steps(self):
        return int(0.1 * get_data_length()) 
    @property
    def predict_start(self):
        return get_window_length() - int(0.1 * get_data_length()) 
    @property
    def test_predict_start(self):
        return get_window_length() - int(0.1 * get_data_length()) 
    @property
    def plot_dir(self):
        return "./hiertsforecaster/results/deepar_fig" 
    @property
    def early_stop(self):
        return True 
    @property
    def lr_schedule(self):
        return True 


class LstNetParams:

    def __init__(self) -> None:
        pass
    @property
    def learning_rate(self):
        return 1e-3 
    @property
    def optimizer(self):
        return 'adam'
    @property
    def h(self):
        return 1
    @property
    def normalize(self):
        return 0
    @property
    def batch_size(self):
        return 16
    @property
    def num_epoch(self):
        return 1000
    @property
    def window(self):
        return get_window_length()
    @property
    def early_stop(self):
        return True
    @property
    def lr_schedule(self):
        return True


class GatingNetParams:
    def __init__(self) -> None:
        pass
    @property
    def recurrent_hidden_dim(self):
        return 1
    @property
    def layer_dim(self):
        return 3
    @property
    def linear_hidden_dim(self):
        return 60
    @property
    def learning_rate(self):
        return 1e-3
    @property
    def num_epochs(self):
        return 100
    @property
    def Lambda(self):
        return 0
    @property
    def lr_decay(self):
        return 0.1
    @property
    def window(self):
        return get_window_length()
    @property
    def batch_size(self):
        return get_batch_size()
    @property
    def early_stop(self):
        return False # disable this in multi-gpu computing, as one process may terminate earlier to cause asynchronization
    @property
    def lr_schedule(self):
        return False


class OnlineParams:

    batch_size = 1
    epoch = 5
    enable_cp = 0
    hazard = 1e-3
    prior_mean = 0
    prior_var = 2
    observe_var = 1
    tau = 2


class PydlmParams:

    def __init__(self) -> None:
        pass
    @property
    def degree_linear(self):
        return 0
    @property
    def discount_linear(self):
        return 0.95
    @property
    def linear_trend_name(self):
        return "linear_trend"
    @property
    def w_linear(self):
        return 0.3
    @property
    def seasonality_period(self):
        return 52
    @property
    def discount_seasonal(self):
        return 0.99
    @property
    def seaonal_name(self):
        return 'weekly'
    @property
    def w_seasonal(self):
        return 0.1
    @property
    def window(self):
        return get_window_length()


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
    num_epochs = 1000
    quantiles = [0.95, 0.7, 0.3, 0.05]
    type = "median"
    window = get_window_length()

    early_stop = True
    lr_schedule = True


class RnnParams:

    learning_rate = 5e-4
    num_epoch = 1000
    batch_size = 128
    window = 168
    input_dim = 1
    hidden_dim = 5
    layer_dim = 3
    normalize = 0
    nonlinearity = 'tanh'
