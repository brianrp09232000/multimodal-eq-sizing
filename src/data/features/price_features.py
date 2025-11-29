import pandas as pd
import numpy as np
from datetime import timedelta
import yfinance as yf
from src.data.loaders import get_return_data, get_single_ticker_history, get_tickers_history

def compute_momentum_rank(input_df):
    """
    Compute 12-1 momentum and cross-sectional momentum rank for each ticker.
    12-1 momentum = Close[t-21] / Close[t-252] - 1.
    cross-sectional rank each day = position / N.
    ----------
    Input dataset must contain columns ['ticker', 'Date']; 
    ----------
    Output dataset adds additional columns:['mom_12_1','mom_rank']
    """

    # Fetch data from yfinance starting at a buffer_start date to support 12–1 momentum calculations
    start = input_df['Date'].min() 
    end = input_df['Date'].max() 
    buffer_start = start - timedelta(days=400)
    df = get_tickers_history(list(input_df['ticker'].unique()), buffer_start, end)
    df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    # Compute 12-1 momentum
    df['Close_t_21']  = df.groupby('ticker')['Close'].shift(21)
    df['Close_t_252'] = df.groupby('ticker')['Close'].shift(252)
    df['mom_12_1'] = df['Close_t_21'] / df['Close_t_252'] - 1

    # Compute cross-sectional 12-1 momentum ranking
    df['mom_position'] = (
        df.groupby('Date')['mom_12_1']
        .rank(method='first')
    )
    df['N'] = df.groupby('Date')['ticker'].transform('count')
    df['mom_rank'] = df['mom_position'] / df['N']

    # Drop buffer dates data; Keep mom_12_1 and mom_rank columns
    df2 = df[df['Date']>=start]
    df2 = df2[['Date','ticker','mom_12_1','mom_rank']]
    
    # Merge 12-1 momentum and rank into the input dataset
    output_df = pd.merge(input_df, df2, on =['ticker','Date'], how='left')

    return output_df

def get_log_mktcap(input_df):
    """
    Compute log(market capitalization) for each ticker.
    log(market cap_t) = log(Close_t) * SharesOutstanding. 
    * Note that Yahoo only provides the latest shares outstanding, so we don't have the historical 
    shares outstanding at time t. The calculation method for market capitalization may not be accurate.
    ----------
    Input dataset must contain columns ['ticker', 'Date']; 
    ----------
    Output dataset adds additional columns:['log_mktcap']
    """
    start = input_df['Date'].min() 
    end = input_df['Date'].max()
    tickers = list(input_df['ticker'].unique())
    
    rows = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)

        # Get CURRENT shares outstanding
        shares = stock.info.get("sharesOutstanding", None)

        # Fetch historical price
        df_price = get_single_ticker_history(ticker, start, end)

        # Add ticker column
        df_price["ticker"] = ticker
        df_price = df_price.reset_index()

        # Compute market cap and log market cap
        if shares is not None:
            df_price["mktcap"] = df_price["Close"] * shares
            df_price["log_mktcap"] = (df_price["mktcap"]).apply(
                lambda x: None if pd.isna(x) else np.log(x)
            )
        else:
            df_price["market_cap"] = None
            df_price["log_mktcap"] = None

        rows.append(df_price)
    # concat all tickers and merge with the input dataset    
    mktcap = pd.concat(rows, ignore_index=True)[['Date','ticker','log_mktcap']]
    output_df = pd.merge(input_df, mktcap, on=['Date','ticker'], how='left')
    
    return output_df
    
def compute_SPY_r1 (input_df):
    """
    Compute SPY r1 = (Close_t/Clost_t_1) -1
    ----------
    Input dataset must contain columns ['Date']. 
    ----------
    Output dataset adds additional columns:['spy_r1']
    """
    start = input_df['Date'].min() 
    end = input_df['Date'].max()
    buffer_start = start - timedelta(days=7)
    
    df_spy = get_single_ticker_history("SPY", buffer_start, end)
    df_spy["spy_r1"] = df_spy["Close"] / df_spy["Close"].shift(1) - 1
    df_spy = df_spy[df_spy['Date']>=start]
    df_spy = df_spy[['Date','ticker','spy_r1']]
    
    output_df = pd.merge(input_df, df_spy, on=['Date'], how='left')
    return output_df
