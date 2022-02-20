import numpy as np
import pandas as pd, os

PATH = os.path.realpath(__file__)


def preprocess(data: pd.DataFrame, windows=[5, 10, 20, 60, 120]):
    check_cols = ["open", "close", "volume", "low", "high"]
    assert len(set(list(data)) & set(check_cols)) == len(check_cols), f"data 변수명 확인 필요 : {check_cols} != {list(data)}"
    for window in windows:
        data["close_ma{}".format(window)] = data["close"].rolling(window).mean()
        data["volume_ma{}".format(window)] = data["volume"].rolling(window).mean()
        data["close_ma%d_ratio" % window] = (data["close"] - data["close_ma%d" % window]) / data["close_ma%d" % window]
        data["volume_ma%d_ratio" % window] = (data["volume"] - data["volume_ma%d" % window]) / data[
            "volume_ma%d" % window
        ]

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
