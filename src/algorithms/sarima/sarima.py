from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pandas as pd

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


class SARIMA_MODEL(object):
    def __init__(self, train, sarimax_kwargs, **kwargs):
        model = SARIMAX(train, **sarimax_kwargs)
        self.model = model.fit(disp=False)
        self.train = train
        self.train_start = train.index.tolist()[0]
        self.train_end = train.index.tolist()[-1]

    def summary(
        self,
    ):
        return self.model.summary()

    def plot_diagnstics(self, **kwargs):
        self.model.plot_diagnostics(**kwargs)
        return plt.show()

    def get_model(
        self,
    ):
        return self.model

    def forcaset(self, test):
        pred_uc = self.model.get_forecast(steps=len(test))
        return pred_uc

    def plot_forecast(self, test):
        pred_uc = self.forcaset(test)
        pred_mean = pred_uc.predicted_mean
        pred_ci = pred_uc.conf_int()
        pred_ci.index = test.index
        pred_mean.index = test.index
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(2, 1, 1)
        test.plot(ax=ax)
        pred_mean.plot(ax=ax)
        ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="#ff0066", alpha=0.2)
        test_perf = get_regression_metric(test.values, pred_mean.values)["mae"]
        ax.set_xlabel("Date")
        ax.set_ylabel("forecast test data")
        plt.title(f"test mae perf : {test_perf:.2f}")
        return plt.show()

    def predict(
        self,
    ):
        pred = self.model.get_prediction(start=self.train_start, end=self.train_end, dynamic=False)
        return pred

    def plot_train_result(
        self,
    ):
        pred = self.predict()
        pred_ci = pred.conf_int()
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(2, 1, 1)
        self.train.plot(ax=ax)
        pred_mean = pred.predicted_mean
        pred_mean.plot(ax=ax)
        train_perf = get_regression_metric(self.train.values, pred_mean.values)["mae"]
        ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="#ff0066", alpha=0.2)
        ax.set_xlabel("Date")
        ax.set_ylabel("predict train data")
        plt.title(f"train mae perf : {train_perf:.2f}")
        return plt.show()

    def metrics(self, test=None):
        tr_pred = self.predict()
        tr_pred_mean = tr_pred.predicted_mean
        train_perf = get_regression_metric(self.train.values, tr_pred_mean.values)
        train_perf_pd = pd.DataFrame([train_perf], index=["train"])
        if test is not None:
            pred_uc = self.forcaset(test)
            pred_mean = pred_uc.predicted_mean
            test_perf = get_regression_metric(test.values, pred_mean.values)
            test_perf_pd = pd.DataFrame([test_perf], index=["test"])
            return pd.concat([train_perf_pd, test_perf_pd], axis=0)
        else:
            return train_perf_pd

    def plot_train_test(self, test):
        pred_uc = self.forcaset(test)
        pred_mean = pred_uc.predicted_mean
        pred_ci = pred_uc.conf_int()
        pred_ci.index = test.index
        pred_mean.index = test.index
        total = pd.concat([self.train, test], axis=0)
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(2, 1, 1)
        pred_mean.plot(ax=ax, label="one-step ahead prediction", alpha=0.7, color="#ff0066")
        ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="#ff0066", alpha=0.2)
        total.plot(ax=ax)
        plt.legend(loc="upper left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        return plt.show()
