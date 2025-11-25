import pandas as pd
from datetime import datetime, timedelta


def count_headlines_all_days(news_df):
    """Counts the number of headlines for each ticker symbol each day
    Input: news_df pandas dataframe with ticker column for ticker symbols and date for the headline date
    Output: pandas dataframe containing the number of headlines per ticker per day
                indexes are dates in string and tickers as the column names"""

    #check if necessary columns exist, return empty dataframe if they don't
    columns = list(news_df.columns)
    if (('date' not in columns) and ('Date' not in columns)) or (('ticker' not in columns) and ('Stock_symbol' not in columns)):
        print('input dataframe does not have both ticker and date columns')
        return pd.DataFrame()

    #find column names
    date_col = 'date' if 'date' in columns else 'Date'
    ticker_col = 'ticker' if 'ticker' in columns else 'Stock_symbol'
    
    #create list of dates Jan 4, 2010 through Dec 28, 2018 but can start or end earlier if data is available
    format_code = "%Y-%m-%d"  # Corresponds to 'YYYY-MM-DD'
    date_min = datetime.strptime(min('2010-01-04',min(news_df[date_col])[:10]), format_code).date() 
    date_max = datetime.strptime(max('2018-12-28',max(news_df[date_col])[:10]), format_code).date()
    date_lst = [str(date_min+timedelta(i)) for i in range(int((date_max-date_min).days)+1)]
    
    #create empty nested dictionary with tickers as keys and date_lst as values    
    tickers = set(news_df[ticker_col])
    headline_count = dict([(ticker, dict.fromkeys(date_lst, 0)) for ticker in tickers])
    
    #count number of articles per ticker per date
    for ticker, hl_date in zip(news_df[ticker_col],news_df[date_col]):
        headline_count[ticker][hl_date[:10]] += 1
    
    #create dataframe
    df = pd.DataFrame.from_dict(headline_count, orient="columns")
    df = df.sort_index()

    return df


def count_headlines_per_day(news_df):
    """Counts the number of headlines for each day 
    Input: news_df pandas dataframe with column counting headlines per day
    Output: pandas dataframe containing the number of headlines per day"""

    #check if necessary columns exist, return empty dataframe if they don't
    columns = list(news_df.columns)
    if (('date' not in columns) and ('Date' not in columns)) :
        print('input dataframe does not have both ticker and date columns')
        return pd.DataFrame()

    #find column names
    date_col = 'date' if 'date' in columns else 'Date'
    
    #create list of dates Jan 4, 2010 through Dec 28, 2018 but can start or end earlier if data is available
    format_code = "%Y-%m-%d"  # Corresponds to 'YYYY-MM-DD'
    date_min = datetime.strptime(min('2010-01-04',min(news_df[date_col])[:10]), format_code).date() 
    date_max = datetime.strptime(max('2018-12-28',max(news_df[date_col])[:10]), format_code).date()
    date_lst = [str(date_min+timedelta(i)) for i in range(int((date_max-date_min).days)+1)]
    
    #create empty nested dictionary with tickers as keys and date_lst as values    
    headline_count = dict([(day, 0) for day in date_lst])
    
    #count number of articles per ticker per date
    for day in news_df[date_col]:
        headline_count[day[:10]] += 1
    
    #create dataframe
    df = pd.DataFrame.from_dict(headline_count, orient="index", columns=['count'])
    df = df.sort_index()

    return df


def count_headlines_per_ticker(news_df):
    """Counts the number of headlines for each ticker symbol 
    Input: news_df pandas dataframe with ticker column for ticker symbols
    Output: pandas dataframe containing the number of headlines per ticker"""

    #check if necessary columns exist, return empty dataframe if they don't
    columns = list(news_df.columns)
    if not (('ticker' in columns) or ('Stock_symbol' in columns)):
        print('input dataframe does not have both ticker column')
        return pd.DataFrame()

    #find correct column name, column names vary by dataset
    ticker_col = 'ticker' if 'ticker' in columns else 'Stock_symbol'
    
    #create empty nested dictionary with tickers as keys and date_lst as values   
    tickers = set(news_df[ticker_col])
    headline_count = dict([(ticker, 0) for ticker in tickers])
    
    #count number of articles per ticker per date
    for ticker in news_df[ticker_col]:
        headline_count[ticker] += 1

    #create dataframe
    df = pd.DataFrame.from_dict(headline_count, orient="index",columns=['count'])
    df['ticker'] = df.index
    df.index = list(range(len(df)))
    df = df[['ticker','count']]
    
    return df
