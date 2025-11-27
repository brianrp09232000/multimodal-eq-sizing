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


def get_vix_data(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Gets VIX data and calculates the market regime z-score.
    Also has a year buffer to populate the 252 day rolling window
    V2 improves the date time errors
    """
    # Add buffer for rolling window
    buffer_start = start - timedelta(days=365)
    
    # Fetch data using yfinance
    vix_ticker = yf.Ticker("^VIX")
    vix = vix_ticker.history(start=buffer_start, end=end + timedelta(days=1))
    
    if vix.empty:
        raise ValueError("VIX data failure")

    # Process Columns with clarity for columns
    vix = vix[['Close']].rename(columns={'Close': 'VIX_Close'})
    
    rolling_window = 252
    vix['VIX_mean_252'] = vix['VIX_Close'].rolling(window=rolling_window).mean()
    vix['VIX_std_252'] = vix['VIX_Close'].rolling(window=rolling_window).std()
    
    # Compute z-score
    vix['VIX_z'] = (vix['VIX_Close'] - vix['VIX_mean_252']) / vix['VIX_std_252']
    
    # Clip to [-3, 3] per the proposal
    vix['VIX_z'] = vix['VIX_z'].clip(lower=-3, upper=3)
    
    # Standardized the time
    # This works if start already has a timezone
    # We should not be passing in unformatted dates, like datetime(2023, 1, 1)
    # Otherwise it will crash again
    start_utc = pd.Timestamp(start).tz_convert('UTC')
    vix.index = pd.to_datetime(vix.index, utc=True)
    
    vix = vix[vix.index >= start_utc].copy()
    
    # Reset index to make date a column
    vix.reset_index(inplace=True)
    print("Yay!🥳")
    
    return vix[['Date', 'VIX_Close', 'VIX_z']]

