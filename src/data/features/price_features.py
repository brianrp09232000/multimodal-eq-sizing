import pandas as pd
from datetime import timedelta
from src.data.loaders import get_return_data, get_single_ticker_history, get_tickers_history

def compute_momentum_rank(input_df):
    """
    Compute 12-1 momentum and cross-sectional momentum rank for each ticker.
    12-1 momentum = Close[t-21] / Close[t-252] - 1.
    cross-sectional rank each day = position / N.
    ----------
    Input dataset must contain columns ['ticker', 'Date']; 
    ----------
    Output dataset adds additional columns:
        ['mom_12_1','mom_rank']
    """

    # Fetch data from yfinance starting at a buffer_start date to support 12–1 momentum calculations
    start = input_df['Date'].min() 
    end = input_df['Date'].max() 
    buffer_start = start - timedelta(days=400)
    df = get_tickers_history(list(raw_data['ticker'].unique()), buffer_start, end)
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

    return output_df# price_features.py – TODO: implement
