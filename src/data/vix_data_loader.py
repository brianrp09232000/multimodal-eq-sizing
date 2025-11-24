import yfinance as yf
import pandas as pd
import numpy as np 

def extract_and_process_vix(target_start_date="2011-05-02", target_end_date="2018-12-26"):
    """
    The goal is to fetch the VIX data for the z-score calculation all within the
    same time frame of our data
    But we also fetch 1 year before the target start date to ensure the 252 rolling
    window
    """

    # Adding a 365 day buffer for the rolling window
    buffer_start = pd.to_datetime(target_start_date) - pd.Timedelta(days=365)
    fetch_start = buffer_start.strftime('%Y-%m-%d')
    fetch_end = (pd.to_datetime(target_end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    # Get data
    vix = yf.download("^VIX", start=fetch_start, end=fetch_end, progress=False)

    # Quick check to see if empty
    if vix.empty:
        raise ValueError("Download failed, nothing returned form yf")

    # Keep only Adjusted Close aka the price at market close
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    # Picking the right column regardless of versions we have
    if 'Adj Close' in vix.columns:
        vix = vix[['Adj Close']].rename(columns={'Adj Close': 'VIX_Close'})
    elif 'Close' in vix.columns:
        vix = vix[['Close']].rename(columns={'Close': 'VIX_Close'})
    else:
        raise KeyError(f"Could not find 'Adj Close' or 'Close' in columns: {vix.columns}")

    # Rolling 252 day mean and std Dev
    rolling_window = 252
    vix['VIX_mean_252'] = vix['VIX_Close'].rolling(window=rolling_window).mean()
    vix['VIX_std_252'] = vix['VIX_Close'].rolling(window=rolling_window).std()

    # Compute z score 
    vix['VIX_z'] = (vix['VIX_Close'] - vix['VIX_mean_252']) / vix['VIX_std_252']

    # Proposal says to clip to [-3, 3]
    vix['VIX_z'] = vix['VIX_z'].clip(lower=-3, upper=3)

    # Drop the buffer year
    # We filter to keep only dates >= the actual project start date
    final_df = vix[vix.index >= target_start_date].copy()

    # Select only the columns needed for the feature store
    output_df = final_df[['VIX_Close', 'VIX_z']]

    print(f"Processed Rows: {len(output_df)}")
    print(output_df.head())

    return output_df

if __name__ == "__main__":
    try:
        vix_data = extract_and_process_vix()
        vix_data.to_csv("vix_features.csv")
        print("LETS GO!😩: Saved to vix_features.csv")
    except Exception as e:
        print(f"ERROR NOOO!👹: {e}")