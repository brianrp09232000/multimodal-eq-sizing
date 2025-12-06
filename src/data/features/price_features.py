import sys
import pathlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os

def calculate_leg_one_features(df):
    """
    Computes Simple Returns (Percentage Change) and other Leg A indicators.
    """
    df = df.copy()

    # 1. Standardize Dates (UTC) to match yfinance
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['ticker', 'Date'])

    g = df.groupby('ticker')

    # 1. Returns
    df['r1'] = g['Close'].transform(lambda x: x.pct_change(periods=1))
    df['r5'] = g['Close'].transform(lambda x: x.pct_change(periods=5))
    df['r10'] = g['Close'].transform(lambda x: x.pct_change(periods=10))

    # 2. Trend (EMA Difference)
    ema10 = g['Close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    ema30 = g['Close'].transform(lambda x: x.ewm(span=30, adjust=False).mean())
    df['trend_ema_diff'] = ema10 - ema30

    # 3. Volatility
    df['vol_realized_20d'] = g['r1'].transform(lambda x: x.rolling(window=20).std())

    hl_ratio_sq = (np.log(df['High'] / df['Low'])) ** 2
    rolling_sum = hl_ratio_sq.groupby(df['ticker']).rolling(20).sum().reset_index(level=0, drop=True)
    constant = 1.0 / (4.0 * 20 * np.log(2))
    df['vol_parkinson_20d'] = np.sqrt(constant * rolling_sum)

    # 4. Momentum (12-1) and Cross-Sectional Rank
    # We re-group by ticker to ensure shifts are specific to each stock
    # Logic: Close[t-21] / Close[t-252] - 1
    df['close_t_21'] = df.groupby('ticker')['Close'].shift(21)
    df['close_t_252'] = df.groupby('ticker')['Close'].shift(252)
    df['mom_12_1'] = df['close_t_21'] / df['close_t_252'] - 1

    # Cross-sectional Rank (Requires grouping by Date)
    # Rank magnitude of momentum compared to other stocks on the same day
    df['mom_position'] = df.groupby('Date')['mom_12_1'].rank(method='first')
    df['N'] = df.groupby('Date')['ticker'].transform('count')
    df['mom_rank'] = df['mom_position'] / df['N']

    # 5. SPY r1 (Market Feature) - Fetch SPY data dynamically based on the date range in the input df
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    # Add a buffer to start_date to ensure we can calculate the previous day's close
    buffer_start = start_date - timedelta(days=10)

    # Fetch SPY history
    spy = yf.Ticker("SPY").history(start=buffer_start, end=end_date + timedelta(days=1))
    spy = spy.reset_index()

    # Ensure SPY timezone matches the input df (UTC) to avoid merge errors
    if spy['Date'].dt.tz is None:
        spy['Date'] = spy['Date'].dt.tz_localize('UTC')
    else:
        spy['Date'] = spy['Date'].dt.tz_convert('UTC')

    # Calculate SPY r1
    spy['spy_r1'] = spy['Close'].pct_change(periods=1)

    # Merge SPY r1 back into the main dataframe
    spy_subset = spy[['Date', 'spy_r1']]
    df = pd.merge(df, spy_subset, on='Date', how='left')

    # Optional: Drop temporary calculation columns to keep df clean
    df = df.drop(columns=['close_t_21', 'close_t_252', 'mom_position', 'N'])

    return df
