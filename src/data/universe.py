import pandas as pd
import sys
import pathlib

repo_root = pathlib.Path("/kaggle/working/multimodal-eq-sizing")
sys.path.append(str(repo_root))

from src.data.features.news_features import count_headlines_per_ticker


def tickers_with_most_headlines(news_df, n=200):
    """Finds the tickers with the most headlines 
    Input: news_df pandas dataframe with ticker column for ticker symbols
            optional: n interger, number of top tickers to return
    Output: pandas dataframe containing the number of headlines per ticker
                for the tickers with the most headlines"""

    #count headlines for each ticker
    df = count_headlines_per_ticker(news_df)

    #limit dataframe to n tickers
    df = df.sort_values(['count'], ascending=False)
    df = df[:n]
    df.reset_index(drop=True, inplace=True)
    
    return df
