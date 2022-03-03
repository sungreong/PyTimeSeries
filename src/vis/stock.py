import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


def candle_stick_example(df):
    import yfinance as yf

    df = yf.download("^GSPC", start="2018-12-31", end="2022-01-01")
    df.columns = [i.replace(" ", "_") for i in list(df)]
    fig = go.Figure(
        data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Adj_Close"])]
    )
    fig.update_layout(
        title="(2018-12-31 ~ 2021-12-31)",
        xaxis_title="Date",
        yaxis_title="US Dollars",
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="red",
        title_font_size=30,
        legend_title_font_color="green",
        xaxis_rangeslider_visible=False,
        shapes=[dict(x0="2020-12-31", x1="2020-12-31", y0=0, y1=1, xref="x", yref="paper", line_width=2)],
        annotations=[
            dict(
                x="2021-01-02",
                y=0.05,
                xref="x",
                yref="paper",
                showarrow=False,
                xanchor="left",
                text="Test",
                font={"color": "red"},
            ),
            dict(
                x="2020-11-18",
                y=0.05,
                xref="x",
                yref="paper",
                showarrow=False,
                xanchor="left",
                text="Train",
                font={"color": "blue"},
            ),
        ],
    )
    fig.show()


def stock_box_plot(df, by, figsize=(12, 12), x_rot=0):
    assert "Adj_Close" in list(df)
    assert "Change" in list(df)
    assert "Volume" in list(df)
    assert by in list(df)

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    df.boxplot(by=by, column=["Adj_Close"], ax=axes[0])
    axes[0].set_title("Adjusted Prices", size=24)

    df.boxplot(by=by, column=["Change"], ax=axes[1])
    axes[1].set_title("Returns", size=24)

    df.boxplot(by=by, column=["Volume"], ax=axes[2])
    axes[2].set_title("Volume", size=24)
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(x_rot)
    plt.tight_layout()
    plt.show()


def stock_rolling_plot(df, col, rollings=[5, 10], figsize=(12, 5)):
    plt.figure(figsize=figsize)
    df_ = df[col]
    df_.plot(label="Original", legend=True)
    for rolling in rollings:
        df_.rolling(rolling).mean().plot(legend=True, label=f"rolling_mean  {1+rolling}d")
    plt.show()


def stock_pairplot(df, select_cols, hue=None):
    for col in select_cols:
        assert col in list(df), f"{col} not in {list(df)}"
    if hue is None:
        g = sns.pairplot(df[select_cols], diag_kind="kde", corner=True)
    else:
        assert hue in list(df), f"{hue} not in {list(df)}"
        g = sns.pairplot(df[select_cols + [hue]], diag_kind="kde", corner=True, hue=hue)
    g.map_lower(sns.kdeplot, levels=4, color=".2")
    plt.show()
