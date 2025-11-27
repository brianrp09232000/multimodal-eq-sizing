import pandas as pd
from datetime import datetime, timedelta


def count_headlines_all_days(news_df):
    """Counts the number of headlines for each ticker symbol each day
    Input: news_df pandas dataframe with ticker column for ticker symbols and date for the headline date
    Output: pandas dataframe containing the number of headlines per ticker per day
                indexes are dates in string and tickers as the column names"""
    
    #check columns in dataframe
    columns = list(news_df.columns)
    if (('date' not in columns) and ('Date' not in columns)) or (('ticker' not in columns) and ('Stock_symbol' not in columns)):
        print('input dataframe does not have both ticker and date columns')
        return pd.DataFrame()
    
    #find column names
    date_col = 'date' if 'date' in columns else 'Date'
    ticker_col = 'ticker' if 'ticker' in columns else 'Stock_symbol'
    
    # Count occurrences in the date column
    # assumes date format is 'YYYY-MM-DD' and time may or may not appear after the date
    # removes time from column
    headline_dates = news_df[date_col].str[:10]#.value_counts()
    df = pd.DataFrame({ticker_col: news_df[ticker_col],
                       date_col: headline_dates})
    
    # count headlines per day per ticker
    df = df.groupby([date_col, ticker_col]).size().unstack(fill_value=0)
    
    #create list of dates needed
    format_code = "%Y-%m-%d"# Corresponds to 'YYYY-MM-DD'
    set_of_dates = set(df.index)
    date_min = datetime.strptime(min('2010-01-04',min(set_of_dates)), format_code).date() #datetime(2000,1,1).date()#
    date_max = datetime.strptime(max('2018-12-28',max(set_of_dates)), format_code).date()
    date_lst = [str(date_min+timedelta(i)) for i in range(int((date_max-date_min).days)+1)]
    
    #find dates not in dataframe
    missing_dates = dict([(day,int(0)) for day in set(date_lst).difference(set(df.index))])
    
    #add missing dates to dataframe
    tickers = list(set(df.columns))
    tickers.sort()
    empty_dict = dict([(ticker, missing_dates) for ticker in tickers])
    add_dates = pd.DataFrame(empty_dict)
    df = pd.concat([df, add_dates], ignore_index=False)
    
    #sort rows and columns
    df = df.sort_index()
    df = df.T
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
    Output: pandas dataframe containing two columns: ticker names and the 
                number of headlines for the ticker"""
    
    #check if necessary columns exist, return empty dataframe if they don't
    columns = list(news_df.columns)
    if not (('ticker' in columns) or ('Stock_symbol' in columns)):
        print('input dataframe does not have both ticker column')
        return pd.DataFrame()
    
    #find correct column name, column names vary by dataset
    ticker_col = 'ticker' if 'ticker' in columns else 'Stock_symbol'
    
    # Count occurrences in a specific column
    headline_counts = news_df[ticker_col].value_counts()
    df = headline_counts.to_frame(name='count')
    df['ticker'] = list(df.index)
    df = df.reset_index()
    
    return df[['ticker','count']]
