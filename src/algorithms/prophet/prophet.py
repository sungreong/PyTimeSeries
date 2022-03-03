from datetime import time
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_squared_error as mse,
    mean_squared_log_error as msle,
    r2_score as r2,
    median_absolute_error as mede,
    mean_absolute_percentage_error as mape,
)
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.serialize import model_to_json, model_from_json
import json


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


class Prophet_MODEL(object):
    def __init__(self, prophet_kwargs={}, **kwargs):
        self.model = Prophet(**prophet_kwargs)
        self.model.add_country_holidays(country_name=kwargs.get("country", None))

    def fit(self, train):
        self.train = train
        self.model.fit(train)

    def get_time_column(
        self,
    ):
        future_df = self.model.make_future_dataframe(periods=0, freq="B")
        time_col = future_df.columns.tolist()
        return time_col

    def forecast(self, periods=0, freq="B"):
        future_df = self.model.make_future_dataframe(periods=periods, freq=freq)
        predictions_df = self.model.predict(future_df)
        return predictions_df

    def predict_train(
        self,
    ):
        time_column = self.get_time_column()
        train_dt = self.train[time_column]
        # train_df.set_index(time_column[0] , inplace=True)
        tr_predictions_df = self.model.predict(train_dt)
        return tr_predictions_df

    def predict(
        self,
        time_list,
    ):
        time_column = self.get_time_column()
        pred_df = pd.DataFrame(time_list, columns=time_column)
        pred_df = self.model.predict(pred_df)
        return pred_df

    def plot_time_interval(self, time_list, y_list):
        time_column = self.get_time_column()[0]
        pred_df = self.predict(time_list)
        # pred_df.set_index(time_column, inplace=True)
        plt.plot(time_list, y_list, color="blue", label="Actuals")
        plt.plot(pred_df[time_column], pred_df["yhat"], color="red", label="Predictions")
        plt.plot(
            pred_df[time_column],
            pred_df["yhat_lower"],
            color="green",
            linestyle="--",
            label="Confidence Intervals",
        )
        plt.plot(pred_df[time_column], pred_df["yhat_upper"], color="green", linestyle="--")
        plt.title("Predictions vs Actuals for full time interval", size=24)
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

    def plot_prediction(self, pred_result):
        self.model.plot(pred_result)
        plt.show()

    def plot_component(self, pred_result):
        self.model.plot_components(pred_result)
        plt.show()

    def plot_weekly(self, pred_result):
        time_column = self.get_time_column()[0]
        start_weekday = 0
        while True:
            if pred_result[time_column][start_weekday].weekday() == 0:
                break
            else:
                start_weekday = start_weekday + 1

        end_weekday = start_weekday + 5

        days = pred_result[time_column][start_weekday:end_weekday]
        weekly_seas = pred_result.weekly[start_weekday:end_weekday]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(np.arange(5), weekly_seas)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(days.dt.day_name())
        ax.set_xlabel("Day of week")

        plt.show()

    def metrics(self, target_name, test=None):
        tr_pred = self.predict_train()
        tr_pred = tr_pred["yhat"]
        train_perf = get_regression_metric(self.train[target_name].values, tr_pred.values)
        train_perf_pd = pd.DataFrame([train_perf], index=["train"])
        if test is not None:
            time_column = self.get_time_column()[0]
            te_pred = self.predict(test[time_column].values)
            te_pred = te_pred["yhat"]
            test_perf = get_regression_metric(test[target_name].values, te_pred.values)
            test_perf_pd = pd.DataFrame([test_perf], index=["test"])
            return pd.concat([train_perf_pd, test_perf_pd], axis=0)
        else:
            return train_perf_pd

    def plot_trend(self, df, target_name, title=""):
        time_column = self.get_time_column()[0]
        time_array = df[time_column].values
        y_array = df[target_name].values
        pred = self.predict(time_array)["yhat"]
        # perf = get_regression_metric(y_array, pred.values)
        plt.plot(time_array, y_array, label="Actual")
        plt.plot(time_array, pred, label="Prediction")
        plt.title(title)
        plt.legend()
        plt.xticks(rotation=45)
        return plt.show()

    def plot_changepoints(self, time_list):
        forecast = self.predict(time_list)
        fig = self.model.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), self.model, forecast)
        return a
