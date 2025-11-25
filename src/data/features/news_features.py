import pandas as pd
from datetime import datetime, timedelta


def counting_headlines_all_days(news_df):
    """Counts the number of headlines for each ticker symbol each day
    Input: news_df pandas dataframe with ticker column for ticker symbols and date for the headline date
    Output: pandas dataframe containing the number of headlines per ticker per day
                indexes are dates as strings and tickers are the column names"""
    
    #check if necessary columns exist, return empty df if they don't
    if ('date' not in list(news_df.columns)) or ('ticker' not in list(news_df.columns)):
        print('input dataframe does not have both ticker and date columns')
        return pd.DataFrame()
        
    #create list of dates
    date_min = datetime(2010,1,4)
    date_max = datetime(2018,12,28)
    date_lst = [str((date_min+timedelta(i)).date()) for i in range(int((date_max-date_min).days)+1)]
    
    #create empty nested dictionary with tickers as keys and date_lst as values    
    tickers = set(news_df.ticker)
    headline_count = dict([(ticker, dict.fromkeys(date_lst, 0)) for ticker in tickers])

    #count number of articles per ticker per date
    for row in news_df.index:
        ticker = news_df.ticker[row]
        hl_date = news_df.date[row]
        headline_count[ticker][hl_date] += 1

    #create dataframe
    df = pd.DataFrame.from_dict(headline_count, orient="columns")
    df_sorted = df.sort_index()

    return df_sorted


def counting_headlines_weekdays(news_df):
    """Counts the number of headlines for each ticker symbol each day
            only contains counts for weekdays, no information for Saturdays or Sundays
    Input: news_df pandas dataframe with ticker column for ticker symbols and date for the headline date
    Output: pandas dataframe containing the number of headlines per ticker per day
                indexes are dates as strings and tickers are the column names"""

    #check if necessary columns exist, return empty df if they don't
    if ('date' not in list(news_df.columns)) or ('ticker' not in list(news_df.columns)):
        print('input dataframe does not have both ticker and date columns')
        return pd.DataFrame()
    
    #create list of dates
    date_min = datetime(2010,1,4)
    date_max = datetime(2018,12,28)
    date_lst = [((date_min+timedelta(i)).date()) for i in range(int((date_max-date_min).days)+1)]
    date_lst = [str(d) for d in date_lst if d.weekday() < 5]
    
    #create empty nested dictionary with tickers as keys and date_lst as values    
    tickers = set(news_df.ticker)
    headline_count = dict([(ticker, dict.fromkeys(date_lst, 0)) for ticker in tickers])

    #count number of articles per ticker per date
    for row in news_df.index:
        ticker = news_df.ticker[row]
        hl_date = news_df.date[row]
        headline_count[ticker][hl_date] += 1

    #create dataframe
    df = pd.DataFrame.from_dict(headline_count, orient="columns")
    df_sorted = df.sort_index()

    return df_sorted
