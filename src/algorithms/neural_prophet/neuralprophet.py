from datetime import time
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed

set_random_seed(1234)

from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_squared_error as mse,
    mean_squared_log_error as msle,
    r2_score as r2,
    median_absolute_error as mede,
    mean_absolute_percentage_error as mape,
)


def get_regression_metric(true, pred):
    metrics = {
        "mae": mae(true, pred),
        "mse": mse(true, pred),
        "msle": msle(true, pred),
        "r2": r2(true, pred),
        "mede": mede(true, pred),
        "mape": mape(true, pred),
    }
    return metrics


"""
NeuralProphet(
    growth="linear",  # Determine trend types: 'linear', 'discontinuous', 'off'
    changepoints=None, # list of dates that may include change points (None -> automatic )
    n_changepoints=5,
    changepoints_range=0.8,
    trend_reg=0,
    trend_reg_threshold=False,
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    seasonality_mode="additive",
    seasonality_reg=0,
    n_forecasts=1,
    n_lags=0,
    num_hidden_layers=0,
    d_hidden=None,     # Dimension of hidden layers of AR-Net
    ar_sparsity=None,  # Sparcity in the AR coefficients
    learning_rate=None,
    epochs=40,
    loss_func="Huber",
    normalize="auto",  # Type of normalization ('minmax', 'standardize', 'soft', 'off')
    impute_missing=True,
    log_level=None, # Determines the logging level of the logger object
)
"""


class NeuralProphet_MODEL(object):
    def __init__(self, neuralprophet_kwargs, **kwargs):
        set_random_seed(kwargs.get("seed", 1234))
        self.model = NeuralProphet(**neuralprophet_kwargs)
        self.model.add_country_holidays(country_name="KR")

    def fit(self, data, fit_kwargs):
        _ = self.model.fit(data, **fit_kwargs)

    def test(self, data):
        return self.model.test(data)

    def get_model(
        self,
    ):
        return self.model

    def plot_prediction(self, df, periods=0, n_historic_predictions=0):
        forecast = self.predict(df, periods, n_historic_predictions=n_historic_predictions)
        _ = self.model.plot(forecast)
        return plt.show()

    def plot_componets(self, **kwargs):
        _ = self.model.plot_components(**kwargs)
        return

    def plot_parameters(self, **kwargs):
        _ = self.model.plot_parameters(**kwargs)

    def plot_result(self, step, prediction):
        return self.model.highlight_nth_step_ahead_of_each_forecast(step).plot(prediction)

    def predict(self, base_df, periods=0, n_historic_predictions=0):
        future = self.model.make_future_dataframe(
            base_df, periods=periods, n_historic_predictions=n_historic_predictions
        )
        forecast = self.model.predict(future)
        return forecast
