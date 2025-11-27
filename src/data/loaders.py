# %% [code]
import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime, timedelta

def get_return_data(path: str) -> pd.DataFrame:
    df_ticker_return = pd.read_csv(path)
    df_ticker_return['Date'] = pd.to_datetime(df_ticker_return['Date'], utc=True)
    df_ticker_return.loc[df_ticker_return['ticker'] == 'FB', 'ticker'] = 'META'
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

def get_excess_return(df_ticker_return: pd.DataFrame, start: datetime, end:datetime) -> pd.DataFrame:
    """
    Compute same-day open-to-close (O2C) excess return versus SPY.

    Returns a DataFrame with columns:
      - ticker
      - Date
      - excess_return = (Close/Open - 1) - (SPY_Close/SPY_Open - 1)
    """
    df_base_ticker_return = get_single_ticker_history("SPY", start, end)
    return_formula = "(Close / Open) - 1"
    df_ticker_return["o2c_return"] = df_ticker_return.eval(return_formula)
    df_base_ticker_return["spy_o2c_return"] = df_base_ticker_return.eval(return_formula)

    df_ticker_return = df_ticker_return[["ticker", "Date", "o2c_return"]]
    df_base_ticker_return = df_base_ticker_return[["Date", "spy_o2c_return"]]

    excess_return_formula = "o2c_return - spy_o2c_return"
    df_with_target = pd.merge(df_ticker_return, df_base_ticker_return, on="Date", how="left")
    df_with_target["excess_return"] = df_with_target.eval(excess_return_formula)
    df_with_target = df_with_target.drop(['o2c_return', 'spy_o2c_return'], axis=1)
    
    return df_with_target

def get_spread_z(df,
                 date_col="Date", ticker_col="ticker",
                 high_col="High", low_col="Low",
                 rolling_window=252, min_expanding=20, clip_z=3.0):
    """
    Adds:
      - s_cs: Corwin–Schultz raw relative spread (decimal)
      - spread_z: per-ticker z-score of s_cs; uses expanding stats until 'rolling_window' is available,
                  then switches to rolling(window).
    """
    out = df.copy()
    out = out.sort_values([ticker_col, date_col])

    g = out.groupby(ticker_col, sort=False)
    H  = out[high_col].astype(float)
    L  = out[low_col].astype(float)
    Hm1 = g[high_col].shift(1).astype(float)
    Lm1 = g[low_col].shift(1).astype(float)

    # --- Corwin–Schultz s_cs ---
    eps = 1e-12
    lnHL_t   = np.log((H / L).clip(lower=1+eps))
    lnHL_tm1 = np.log((Hm1 / Lm1).clip(lower=1+eps))
    beta  = lnHL_t.pow(2) + lnHL_tm1.pow(2)

    Hmax  = pd.concat([H, Hm1], axis=1).max(axis=1)
    Lmin  = pd.concat([L, Lm1], axis=1).min(axis=1)
    gamma = np.log((Hmax / Lmin).clip(lower=1+eps)).pow(2)

    k = 3 - 2*np.sqrt(2)
    alpha = (np.sqrt(2*beta.clip(lower=0)) - np.sqrt(beta.clip(lower=0))) / (2*k) \
            - np.sqrt((gamma / k).clip(lower=0))
    out["s_cs"] = (2*(np.exp(alpha) - 1)) / (np.exp(alpha) + 1)

    # Optional static winsor per ticker to calm tails (still causal enough)
    qlo = g["s_cs"].transform(lambda s: s.quantile(0.01))
    qhi = g["s_cs"].transform(lambda s: s.quantile(0.99))
    s_w = out["s_cs"].clip(lower=qlo, upper=qhi)

    # --- Expanding stats (young phase) ---
    mu_exp = s_w.groupby(out[ticker_col]).expanding(min_periods=min_expanding).mean().reset_index(level=0, drop=True)
    sd_exp = s_w.groupby(out[ticker_col]).expanding(min_periods=min_expanding).std(ddof=0).reset_index(level=0, drop=True)

    # --- Rolling stats (mature phase) ---
    mu_roll = s_w.groupby(out[ticker_col]).rolling(rolling_window, min_periods=rolling_window).mean().reset_index(level=0, drop=True)
    sd_roll = s_w.groupby(out[ticker_col]).rolling(rolling_window, min_periods=rolling_window).std(ddof=0).reset_index(level=0, drop=True)

    age = g.cumcount() + 1
    use_roll = (age >= rolling_window)

    # choose per row (causal)
    mu = np.where(use_roll, mu_roll, mu_exp)
    sd = np.where(use_roll, sd_roll, sd_exp)

    spread_z = (s_w.values - mu) / sd
    out["spread_z"] = pd.Series(spread_z, index=out.index).clip(-clip_z, clip_z)
    out.drop('s_cs', axis=1)

    return out

def get_sector_map(tickers):
    sectors = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
            sectors[t] = info.get("sector")
        except Exception as e:
            print(f"Failed for {t}: {e}")
            sectors[t] = None
    return pd.Series(sectors, name="sector")
