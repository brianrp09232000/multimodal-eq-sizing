# %% [code]
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import re

import torch
from torch.utils.data import Dataset
# Switched to AutoTokenizer to support distilroberta
from transformers import AutoTokenizer

# Constants for Event Flags
EARNINGS_KEYWORDS = [
    "earnings",
    "eps",
    "quarterly",
    "q1",
    "q2",
    "q3",
    "q4",
    "full-year",
    "revenue",
]

GUIDANCE_KEYWORDS = [
    "guidance",
    "forecast",
    "outlook",
    "raises",
    "cuts",
]

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

def get_adv_dollar(
    df: pd.DataFrame,
    window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """
    Compute dollar-volume ADV per (ticker, date) and return ONLY join keys
    and new columns so it can be merged back to the main df.

    Output columns:
        ["Date", "ticker", "dollar_volume", "adv_dollar"]
    """
    tmp = df[["Date", "ticker", "Close", "Volume"]].copy()

    # sort by ticker & date to get a proper rolling window
    tmp = tmp.sort_values(["ticker", "Date"])

    # daily dollar volume
    tmp["dollar_volume"] = tmp["Close"] * tmp["Volume"]

    # rolling ADV in dollars, by ticker
    tmp["adv_dollar"] = (
        tmp
        .groupby("ticker")["dollar_volume"]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # return only keys + new cols for joining
    return tmp[["Date", "ticker", "dollar_volume", "adv_dollar"]]


# Loaders for Leg 2

_GLOBAL_TOKENIZER = None

def _get_tokenizer():
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        try:
            # Updated to DistilRoBERTa for speed
            _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')
        except OSError:
            _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained('distilroberta-base')
    return _GLOBAL_TOKENIZER

class NewsDataset(Dataset):
    """
    Leg 2 Dataset
    Handles missing news, computes flags, and prepares aux features
    """
    def __init__(self, dataframe):
        self.data = dataframe
        # Precompile regex for efficiency
        self.earnings_regex = re.compile("|".join(EARNINGS_KEYWORDS))
        self.guidance_regex = re.compile("|".join(GUIDANCE_KEYWORDS))

    def __len__(self):
        return len(self.data)
        
    def _compute_flags(self, sentences):
        # Simple keyword matching to signal important corporate events
        earnings_flag = 0.0
        guidance_flag = 0.0
        
        combined_text = " ".join(sentences).lower()
        
        # Use precompiled regex searches
        if self.earnings_regex.search(combined_text):
            earnings_flag = 1.0
        if self.guidance_regex.search(combined_text):
            guidance_flag = 1.0
            
        return earnings_flag, guidance_flag

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.
        
        Args:
            idx (int): The index of the item to retrieve.
            
        Returns:
            A dictionary containing:
                - sentences: List of sentence strings
                - time_gaps: List of time gaps corresponding to sentences
                - target: Target value
                - aux: List of auxiliary features [novelty, velocity, earnings_flag, guidance_flag]
                - news_mask: Mask indicating presence of news (1.0 for present, 0.0 for absent)
        """
        row = self.data.iloc[idx]
        
        sentences = row['Article_title']
        # Safety checks for sentences handling parsing errors
        if not isinstance(sentences, list):
            try:
                sentences = ast.literal_eval(sentences)
            except:
                sentences = []

        # 1. Handle Missing News via masking
        # If no headlines... news mask = 0
        news_mask = 1.0
        if not sentences:
            sentences = ["No news reported."] # Placeholder for BERT tokenization
            news_mask = 0.0
        
        # 2. Velocity
        # Velocity = count in 24h
        velocity = float(len(sentences)) if news_mask == 1.0 else 0.0
        
        # 3. Event Flags
        earn_flag, guide_flag = self._compute_flags(sentences)
        
        # 4. Time Gaps how recent
        # Needed for recency weighted pooling in the model
        time_gaps = row.get('gaps', [0.0] * len(sentences))
        if not isinstance(time_gaps, list) or len(time_gaps) != len(sentences):
                time_gaps = [0.0] * len(sentences)

        # 5. Novelty
        # Novelty = % new entities
        novelty = float(row.get('novelty', 0.0))

        return {
            'sentences': sentences,
            'time_gaps': time_gaps,
            'target': row.get('target', 0.0),
            'aux': [novelty, velocity, earn_flag, guide_flag], # [4 dims]
            'news_mask': news_mask
        }

def han_collate_fn(batch):
    """
    Batches variable-length documents for the HAN model
    Crucial for processing hierarchical data sentences -> doc
    """
    tokenizer = _get_tokenizer()

    flat_sentences = []
    doc_lengths = []
    targets = []
    aux_features = []
    news_masks = []
    
    # Collect time gaps to pad them later
    batch_time_gaps = []

    for item in batch:
        sentences = item['sentences']
        gaps = item['time_gaps']
        
        # Memory safety: Cap sentences at 50 to avoid OOM on GPU
        if len(sentences) > 50:
            sentences = sentences[:50]
            gaps = gaps[:50]
            
        doc_lengths.append(len(sentences))
        flat_sentences.extend(sentences)
        targets.append(item['target'])
        aux_features.append(item['aux'])
        news_masks.append(item['news_mask'])
        batch_time_gaps.append(gaps)

    # Tokenize flattened sentences in one go for efficiency purposes
    tokenized = tokenizer(
        flat_sentences,
        padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    
    # Pad time gaps to match max document length in this batch
    max_len = max(doc_lengths)
    padded_gaps = torch.zeros(len(batch), max_len)
    for i, gaps in enumerate(batch_time_gaps):
        length = len(gaps)
        padded_gaps[i, :length] = torch.tensor(gaps, dtype=torch.float)

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'doc_lengths': doc_lengths,
        'time_gaps': padded_gaps,
        'targets': torch.tensor(targets, dtype=torch.float).unsqueeze(1),
        'aux_features': torch.tensor(aux_features, dtype=torch.float),
        'news_mask': torch.tensor(news_masks, dtype=torch.float).unsqueeze(1)
    }