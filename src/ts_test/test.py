from statsmodels.tsa.stattools import kpss, adfuller
import pandas as pd
import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt


def adf_test(df: pd.Series):
    """
    정상성(Stationary) 검증 방법 - Dicky-Fuller test

    $HO$ (귀무가설) : 비정상이 아니다라고 할만한 근거가 없다.

    $H1$ 대립가설 : 비정상이 아니다.

    * p-value가 5% 이내면, 귀무가설 기각

    * Adjusted Close를 통해 정상섬 검증

    * p-value가 0.05보다 작지 않으므로, 귀무가설을 기각할 수 없다. 그래서 비정상이 아니라고 할만한 근거가 없기 때문에 비정상이라고 말할 수도 있다.

    :param df: _description_
    :type df: pd.Series
    """
    result = adfuller(df)
    print("Test statistic: ", result[0])
    print("p-value: ", result[1])
    print("Critic values")
    for k, v in result[4].items():
        print("\t%s : %.3f" % (k, v))


def kpss_test(df: pd.Series):
    """
    단위근 검정 방법

    $HO$ (귀무가설) : 시계열 데이터가 정상성을 가진다.

    $H1$ 대립가설 : 시계열 데이터가 정상성을 가지지 않는다.

    * p-value가 5% 이내면, 귀무가설 기각

    :param df: _description_
    :type df: pd.Series
    """
    statistics, p_value, n_lags, critic_values = kpss(df)
    print(f"KPSS Statistics : {statistics}")
    print(f"p-value : {p_value}")
    print(f"num lags : {n_lags}")
    print("Critic Values : ")
    for k, v in critic_values.items():
        print(f"   {k} : {v}")


def plot_acf_pacf(series: pd.Series, lags_n=10, stock_name=""):
    """

    # ACF(Autocorrelation Function) , PACF(Partial Autocorrelation)

    ## ACF

    * $y_t$ 와 $y_{t+k}$ 사이에 correlation 측정 얼마나 관계가 있는지를 측정하는 것

    ## PACF

    * PACF(k) = Corr($e_t,e_{t-k}$)

    * TIME LAG가 130~150 사이에서 양에서 음수로 바뀐다.

    * PACF 에서는 0 이후에는 급격하게 감소함


    :param series: _description_
    :type series: pd.Series
    :param lags_n: _description_, defaults to 10
    :type lags_n: int, optional
    :param stock_name: _description_, defaults to ""
    :type stock_name: str, optional
    """
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    sgt.plot_acf(series, lags=lags_n, zero=False, ax=ax1)
    ax1.set_title(f"ACF {stock_name}")

    sgt.plot_pacf(series, lags=lags_n, zero=False, method=("ols"), ax=ax2)
    ax2.set_title(f"PACF {stock_name}")

    plt.show()
