import pytest
import pandas as pd
import numpy as np
from hiertsforecaster.mecats import mecats
from tests import test_config
import pdb


@pytest.fixture
def input_mecats() -> pd.DataFrame:
    data = pd.read_csv('./sample_data.csv', index_col=1).drop(columns=['Unnamed: 0'], axis=1)
    return data


@pytest.fixture
def expected_mecats(metrics_mode) -> pd.DataFrame:
    if not metrics_mode:
        point_pred = pd.read_csv("./point_prediction_metrics_false.csv", index_col=0)
        quant_pred = pd.read_csv("./quant_prediction_metrics_false.csv", index_col=0)
        return point_pred, quant_pred, {"MAPE": np.nan, "CRPS": np.nan}
    else:
        point_pred = pd.read_csv("./point_prediction_metrics_true.csv", index_col=0)
        quant_pred = pd.read_csv("./quant_prediction_metrics_true.csv", index_col=0)
        metric = np.load("./metrics.npz")
        return point_pred, quant_pred, {"MAPE": metric['mape'], "CRPS": metric['crps']}


@pytest.fixture(params=[True, False])
def metrics_mode(request) -> bool:
    return request.param


def test_mecats(
    input_mecats, expected_mecats, metrics_mode
) -> None:
    """
    Unit test for MECATS model.
    """

    test_config.MecatsParams.valid_ratio = 0.2
    test_config.MecatsParams.quantile = 1
    test_config.MecatsParams.online = 0
    test_config.MecatsParams.metric_mode = metrics_mode

    input_data = input_mecats
    expected_point, expected_quant, expected_metrics = expected_mecats
    mecats_model = mecats(data=input_data, nodes=[[2]], config=test_config)
    mecats_model.fit()

    if metrics_mode:
        metric = np.load("./pred_res/metrics.npz")
        mape, crps = metric['mape'], metric['crps']
        assert np.all(np.isclose(mape, expected_metrics['MAPE']))
        assert np.all(np.isclose(crps, expected_metrics['CRPS']))
        point_pred = pd.read_csv("./pred_res/point_prediction_metrics_true.csv", index_col=0)
        quant_pred = pd.read_csv("./pred_res/quant_prediction_metrics_true.csv", index_col=0)
    else:
        point_pred = pd.read_csv("./pred_res/point_prediction_metrics_false.csv", index_col=0)
        quant_pred = pd.read_csv("./pred_res/quant_prediction_metrics_false.csv", index_col=0)

    pd.testing.assert_frame_equal(point_pred, expected_point)
    pd.testing.assert_frame_equal(quant_pred, expected_quant)
