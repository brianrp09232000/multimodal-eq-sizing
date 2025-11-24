# %% [code]
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta

def get_return_data(path: str) -> pd.DataFrame:
    df_ticker_return = pd.read_csv(path)
    df_ticker_return['Date'] = pd.to_datetime(df_ticker_return['Date'], utc=True)
    return df_ticker_return

def get_single_ticker_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    ticker_yf = yf.Ticker(ticker)
    df_return = ticker_yf.history(start = start, end = end+timedelta(days=1))
    df_return['ticker'] = ticker 
    df_return.reset_index(inplace=True)
    return df_return

def get_tickers_history(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    tickers_history_dfs = []
    for ticker in tickers:    
        df = get_single_ticker_history(ticker, start, end)
        tickers_history_dfs.append(df)
    if len(tickers_history_dfs) > 1:
        return pd.concat(tickers_history_dfs)
    return df

    

