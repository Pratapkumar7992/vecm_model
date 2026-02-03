# vecm_model.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')   # IMPORTANT for Flask
import matplotlib.pyplot as plt

from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.tsa.stattools import adfuller


def fetch_data():
    data = yf.download(["TCS.NS", "NIFTYBEES.NS", "^NSEI"], start="2018-01-01")
    df = data['Close']
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('B')  # give frequency to index (fix warning)
    return df


def check_stationarity(series):
    result = adfuller(series)
    return result[1]  # p-value


def build_vecm_and_forecast(steps=10):
    df = fetch_data()

    # 1️⃣ Differencing not required manually in VECM
    # 2️⃣ Check cointegration
    order = select_order(df, maxlags=5)
    lag = order.aic

    rank = select_coint_rank(df, det_order=0, k_ar_diff=lag)
    coint_rank = rank.rank

    model = VECM(df, k_ar_diff=lag, coint_rank=coint_rank)
    vecm_res = model.fit()

    forecast = vecm_res.predict(steps=steps)

    forecast_df = pd.DataFrame(
        forecast,
        columns=df.columns
    )

    forecast_df.index = pd.date_range(
        start=df.index[-1],
        periods=steps+1,
        freq='B'
    )[1:]

    # Plot
    plt.figure(figsize=(14, 7))
    for col in df.columns:
        plt.plot(df[col], label=f"Historical {col}")
        plt.plot(forecast_df[col], linestyle='--', label=f"Forecast {col}")

    plt.legend()
    plt.title("VECM Forecast")
    plt.savefig("static/forecast.png")
    plt.close()

    return df.tail(200), forecast_df
