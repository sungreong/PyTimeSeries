import numpy as np
import pandas as pd, os
import math

PATH = os.path.realpath(__file__)

__all__ = ["process_ma_ratio", "process_ratio", "process_date", "process_log_returns"]


def ma_ratio(data, column, window):
    v = data[column].rolling(window).mean()
    data[f"{column}_ma{window:d}_ratio"] = (data[column] - v) / v
    return data


def process_ma_ratio(data: pd.DataFrame, windows=[5, 10, 20, 60, 120], columns=["close", "volume"], drop_na=True):
    new_cols = []
    for col in columns:
        assert col in list(data), f"{col} not in {list(data)}"
        for window in windows:
            data = ma_ratio(data, col, window)
            new_cols.append(f"{col}_ma{window:d}_ratio")
    else:
        print("make new cols")
        print(new_cols)
        if drop_na:
            data = data.dropna()
    return data


def process_ratio(data: pd.DataFrame):
    data["open_lastclose_ratio"] = np.zeros(len(data))
    data.loc[1:, "open_lastclose_ratio"] = (data["open"][1:].values - data["close"][:-1].values) / data["close"][
        :-1
    ].values
    data["high_close_ratio"] = (data["high"].values - data["close"].values) / data["close"].values
    data["low_close_ratio"] = (data["low"].values - data["close"].values) / data["close"].values
    data["close_lastclose_ratio"] = np.zeros(len(data))
    data.loc[1:, "close_lastclose_ratio"] = (data["close"][1:].values - data["close"][:-1].values) / data["close"][
        :-1
    ].values
    data["volume_lastvolume_ratio"] = np.zeros(len(data))
    data.loc[1:, "volume_lastvolume_ratio"] = (data["volume"][1:].values - data["volume"][:-1].values) / data[
        "volume"
    ][:-1].replace(to_replace=0, method="ffill").replace(to_replace=0, method="bfill").values
    data = data.dropna()
    return data


def process_date(data: pd.DataFrame, date_pd: pd.DataFrame):
    data["day_of_week"] = date_pd.dt.day_name().values
    data["month"] = date_pd.dt.month.values
    data["year"] = date_pd.dt.year.values
    data["day"] = date_pd.dt.day.values
    data["week_of_year"] = date_pd.dt.week.values
    data["year/month"] = data.apply(lambda x: f"{int(x['year']):04d}/{int(x['month']):02d}", axis=1)  #
    return data


def process_log_returns(data: pd.DataFrame, close_col, print_volatiltiy=True):
    data["log_returns"] = np.log(data[close_col] / data[close_col].shift())
    if print_volatiltiy:
        _volatility = data["log_returns"].std()
        daily_volatility = _volatility * 100
        monthly_volatility = math.sqrt(21) * daily_volatility
        annual_volatility = math.sqrt(252) * daily_volatility
        print("Daily volatility: ", f"{daily_volatility:.2f}%")
        print("Monthly volatility: ", f"{monthly_volatility:.2f}%")
        print("Annual volatility: ", f"{annual_volatility:.2f}%")
    return data
