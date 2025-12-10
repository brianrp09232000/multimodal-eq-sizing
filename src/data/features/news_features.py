import pandas as pd
from datetime import datetime, timedelta
import typing as _t
import os
import glob
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModel,
    pipeline,
)

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


def count_headlines_per_ticker(news_df, start=None, end=None):
    """Counts the number of headlines for each ticker symbol 
    Input: news_df pandas dataframe with ticker column for ticker symbols
    Output: pandas dataframe containing two columns: ticker names and the 
                number of headlines for the ticker"""
    
    #check columns in dataframe
    columns = list(news_df.columns)
    if (('date' not in columns) and ('Date' not in columns)) or (('ticker' not in columns) and ('Stock_symbol' not in columns)):
        print('input dataframe does not have both ticker and date columns')
        return pd.DataFrame()
    
    #find column names
    date_col = 'date' if 'date' in columns else 'Date'
    ticker_col = 'ticker' if 'ticker' in columns else 'Stock_symbol'

    #filter dates
    if start is not None: 
        start_filter = news_df[date_col] >= str(start)
        news_df = news_df[start_filter]
    if end is not None: 
        end_filter = news_df[date_col] <= str(end)
        news_df = news_df[end_filter]
    
    # Count occurrences in a specific column
    headline_counts = news_df[ticker_col].value_counts()
    df = headline_counts.to_frame(name='count')
    df['ticker'] = list(df.index)
    df = df.reset_index(drop=True)
    
    return df[['ticker','count']]


# -----------------------------
# Configuration
# -----------------------------

# Name of the NER model - use DistilBERT NER (faster)
# (you can swap for another HF NER model if you like)
NER_MODEL_NAME = "elastic/distilbert-base-uncased-finetuned-conll03-english"

# Name of the encoder for z_news (can be same or different)
# We now use FinBERT as a "precomputed, frozen" encoder for finance text.
# ENCODER_MODEL_NAME = "ProsusAI/finbert"
ENCODER_MODEL_NAME = "yiyanghkust/finbert-tone"

# -----------------------------
# Globals for recency-weighted pooling
# -----------------------------

# Tau (time decay coefficient) from the paper:
# alpha_k ∝ exp(w^T h_k - tau * Δt_k)
# You asked for tau = 1.0 (Option B).
NEWS_ATTENTION_TAU: float = 1.0

# w is the "attention direction" vector used in w^T h_k.
# We initialize it once (random normal) the first time we see an embedding
# and then keep it fixed (i.e., NOT learned here).
_NEWS_ATTENTION_W: np.ndarray | None = None


# -----------------------------
# Helpers
# -----------------------------

def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> _t.Iterator[pd.DataFrame]:
    """
    Yield successive row chunks of a pandas DataFrame.

    Parameters
    ----------
    df
        Input DataFrame to be split into chunks.
    chunk_size
        Maximum number of rows in each yielded chunk.

    Yields
    ------
    A view of the original DataFrame with up to `chunk_size` rows.
    """
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size]


def _clean_news_df(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure standard dtypes for the news dataset chunk.

    This normalizes the key columns used by the Leg B preprocessing:
      - Date: datetime64[ns]
      - Article_title: str
      - Article: str (may be empty string if column missing)
      - Stock_symbol: str

    Parameters
    ----------
    chunk
        Input DataFrame chunk.
    
    Returns
    -------
    DataFrame chunk with normalized dtypes.
    """
    chunk = chunk.copy()
    chunk["Date"] = pd.to_datetime(chunk["Date"])
    chunk["Article_title"] = chunk["Article_title"].astype(str)
    chunk["Article"] = chunk.get("Article", "").astype(str)
    chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str)
    return chunk


def iter_dataset_chunks(
    file_path: _t.Optional[str] = None,
    directory: _t.Optional[str] = None,
    pattern: _t.Optional[str] = None,
    news_df: _t.Optional[pd.DataFrame] = None,
    max_rows: _t.Optional[int] = None,
    chunk_size: int = 20_000,
) -> _t.Iterator[pd.DataFrame]:
    """
    Yield chunks of a dataset either from:
      - a provided in memory DataFrame (`news_df`), or
      - one or more CSV files on disk.

    If `news_df` is provided, CSV loading is skipped.

    Parameters
    ----------
    file_path 
        Path to a single CSV file.
    directory
        Directory containing CSV shards.
    pattern
        Glob pattern for CSV shards inside `directory`.
    news_df
        If provided, iterate directly over this DataFrame in chunks.
    max_rows 
        Maximum number of rows to yield across all chunks (None for no limit).
    chunk_size
        Number of rows per yielded chunk.

    Yields
    ------
    A DataFrame chunk of size up to `chunk_size` with normalized dtypes.
    """

    # Work with passed DataFrame in chunks if provided
    if news_df is not None:
        print("Using in memory DataFrame (news_df) for chunking...")
        rows_yielded = 0

        for raw_chunk in chunk_dataframe(news_df, chunk_size):
            chunk = _clean_news_df(raw_chunk)
            yield chunk
            rows_yielded += len(chunk)

            if max_rows is not None and rows_yielded >= max_rows:
                print(f"Reached MAX_ROWS={max_rows}, stopping chunk loading.")
                return

        return

    # Load from CSV files on disk and iterate in chunks
    files: list[str] = []
    search_path = ""

    if file_path and os.path.isfile(file_path):
        files = [file_path]
    elif directory and pattern:
        search_path = os.path.join(directory, pattern)
        files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(
            "No CSV files found and no in memory DataFrame provided."
        )

    print(f"Found {len(files)} CSV files:")
    for f in files:
        print(f"  - {f}")

    rows_yielded = 0

    for f in files:
        for raw_chunk in pd.read_csv(f, chunksize=chunk_size):
            chunk = _clean_news_df(raw_chunk)
            yield chunk
            rows_yielded += len(chunk)

            if max_rows is not None and rows_yielded >= max_rows:
                print(f"Reached MAX_ROWS={max_rows}, stopping chunk loading.")
                return



# -----------------------------
# Step 1: NER setup
# -----------------------------

def create_ner_pipeline(model_name: str = NER_MODEL_NAME, use_onxx: bool = False) -> pipeline:
    """
    Create a PyTorch (or ONNX) based NER pipeline using a pretrained model.

    Parameters
    ----------
    model_name
        Name of the pretrained NER model. (pip install "optimum[onnxruntime-gpu])

    Returns
    -------
    transformers.Pipeline
        NER pipeline object.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_onxx:
        # Optional ONNX acceleration via optimum
        from optimum.onnxruntime import ORTModelForTokenClassification
        print("Using ONNX Runtime for NER...")
        ner_model = ORTModelForTokenClassification.from_pretrained(
            model_name,
            export=True,
        )
    else:
        ner_model = AutoModelForTokenClassification.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1

    ner_pipe = pipeline(
        task="ner",
        model=ner_model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        framework="pt",
        device=device,
    )
    return ner_pipe


def extract_entities_from_text(ner_pipe: pipeline, text: str) -> list[str]:
    """
    Run NER on a piece of text and return a list of entity strings.

    Parameters
    ----------
    ner_pipe
        NER pipeline object.
    text
        Input text to extract entities from.

    Returns
    -------
    list of str
        List of extracted entity strings.
    """
    if not text or isinstance(text, float):
        return []

    try:
        entities = ner_pipe(text)
    except Exception as e:
        print(f"NER failed for text: {str(text)[:80]}... Error: {e}")
        return []

    result = []
    for ent in entities:
        word = ent.get("word")
        if word:
            result.append(word)
    return result


def clean_entity_word(word: str) -> str:
    """
    Normalize entity tokens coming from a WordPiece tokenizer.

    Parameters
    ----------
    word
        Token string from NER output.

    Returns
    -------
    str
        Cleaned token string.
    """
    if not isinstance(word, str):
        return ""
    # remove all leading '#' characters (##, ###, etc.)
    w = word.lstrip("#").strip()
    return w.lower()


# -----------------------------
# Step 2: Event flags
# -----------------------------

EARNINGS_KEYWORDS = [
    "earnings",
    "eps",
    "quarterly results",
    "q1",
    "q2",
    "q3",
    "q4",
    "full-year results",
    "revenue",
]

GUIDANCE_KEYWORDS = [
    "guidance",
    "forecast",
    "outlook",
    "raises guidance",
    "cuts guidance",
]

MERGER_KEYWORDS = [
    "merger",
    "acquisition",
    "acquire",
    "to buy",
    "to acquire",
    "takeover",
    "m&a",
]

RATING_KEYWORDS = [
    "upgrade",
    "downgrade",
    "raised to",
    "cut to",
    "buy rating",
    "sell rating",
    "hold rating",
]


def make_text_for_flags(row: pd.Series) -> str:
    """
    Combine title and article for flag detection.

    Parameters
    ----------
    row
        DataFrame row containing 'Article_title' and 'Article'.

    Returns
    -------
    str
        Combined text for flag detection.
    """
    title = row.get("Article_title", "")
    article = row.get("Article", "")
    text = f"{title} {article}"
    return str(text).lower()


def contains_any(text: str, keywords: list[str]) -> int:
    """
    Return 1 if any keyword appears in text, else 0.

    Parameters
    ----------
    text
        Text to search for keywords.
    keywords
        List of keywords to check.

    Returns
    -------
    int
        1 if any keyword is found, else 0.
    """
    for kw in keywords:
        if kw in text:
            return 1
    return 0


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-row flags: earnings_flag, guidance_flag, merger_flag, rating_flag.

    Parameters
    ----------
    df
        DataFrame containing news rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with added event flag columns.
    """
    texts = df.apply(make_text_for_flags, axis=1)

    df["earnings_flag_row"] = texts.apply(lambda t: contains_any(t, EARNINGS_KEYWORDS))
    df["guidance_flag_row"] = texts.apply(lambda t: contains_any(t, GUIDANCE_KEYWORDS))
    df["merger_flag_row"] = texts.apply(lambda t: contains_any(t, MERGER_KEYWORDS))
    df["rating_flag_row"] = texts.apply(lambda t: contains_any(t, RATING_KEYWORDS))

    return df


# -----------------------------
# Step 3: z_news encoder (FinBERT)
# -----------------------------

def create_encoder(model_name: str = ENCODER_MODEL_NAME, use_onnx: bool = False) -> tuple[AutoTokenizer, AutoModel]:
    """
    Create a transformer encoder (PyTorch or ONNX) to embed headlines.

    We now use FinBERT ("ProsusAI/finbert") to produce finance-specific
    text embeddings, which correspond to h_k in the paper.

    Parameters
    ----------
    model_name
        Name of the pretrained encoder model.
    use_onnx
        Whether to use ONNX for GPU acceleration.

    Returns
    -------
    tuple
        (tokenizer, encoder_model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_onnx:
        from optimum.onnxruntime import ORTModel
        print("Using ONNX Runtime for encoder...")
        encoder_model = ORTModel.from_pretrained(
            model_name,
            export=True,
        )
    else:
        encoder_model = AutoModel.from_pretrained(model_name)

    return tokenizer, encoder_model


def encode_text(
    tokenizer: AutoTokenizer,
    encoder_model: AutoModel,
    text_list: list[str],
    batch_size: int = 32
) -> list[np.ndarray]:
    """
    Encode a list of text into vector embeddings using the encoder (FinBERT).
    Uses the [CLS] token representation (index 0), which we treat as h_k.

    Parameters
    ----------
    tokenizer
        Tokenizer for the encoder model.
    encoder_model
        Encoder model for generating embeddings.
    text_list
        List of strings.
    batch_size
        Batch size for encoding.

    Returns
    -------
    list of np.ndarray
        List of headline embeddings (h_k vectors).
    """
    all_embeddings: list[np.ndarray] = []

    encoder_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(encoder_model, "to"):
        encoder_model.to(device)

    for start in range(0, len(text_list), batch_size):
        batch_texts = text_list[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = encoder_model(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )

        # outputs.last_hidden_state shape: (batch, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        cls_embeddings = last_hidden[:, 0, :]  # shape: (batch, hidden_size)

        for i in range(cls_embeddings.shape[0]):
            vec = cls_embeddings[i].detach().cpu().numpy()
            all_embeddings.append(vec)

    return all_embeddings


# -----------------------------
# Step 4: Build per-row NER and embeddings
# -----------------------------

def process_rows_with_ner_and_embeddings(
    df: pd.DataFrame,
    ner_pipe: pipeline,
    tokenizer: AutoTokenizer,
    encoder_model: AutoModel,
    text_column: str,
) -> pd.DataFrame:
    """
    For each row:
        - get entities from a chosen column (e.g. Article_title or Article) using NER
        - get embedding of that text using FinBERT (for z_news later)

    Parameters
    ----------
    df
        DataFrame containing news rows.
    ner_pipe
        NER pipeline object.
    tokenizer
        Tokenizer for encoder model.
    encoder_model
        Encoder model for generating embeddings.
    text_column
        Column to use for NER and for text embeddings.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'entities' and 'title_embedding' columns.
    """
    if text_column not in df.columns:
        raise ValueError(f"NER text column '{text_column}' not found in DataFrame columns.")

    # Text for both embeddings and NER
    texts = df[text_column].fillna("").astype(str).tolist()

    # 1) Embeddings for all text rows (h_k vectors)
    print(f"Encoding headline embeddings (FinBERT) on {len(texts)} rows...")
    embeddings = encode_text(
        tokenizer=tokenizer,
        encoder_model=encoder_model,
        text_list=texts,
        batch_size=32,
    )
    df["title_embedding"] = embeddings

    # 2) Batched NER on the chosen column
    print(f"Running NER on column '{text_column}' (batched) for {len(texts)} rows...")
    ner_outputs = ner_pipe(texts, batch_size=32)

    entities_list = []
    for ents in ner_outputs:
        if isinstance(ents, list):
            words_raw = [ent.get("word") for ent in ents if ent.get("word")]
        else:
            words_raw = [ents.get("word")] if ents.get("word") else []

        # clean wordpieces like "##corp" → "corp"
        words = [clean_entity_word(w) for w in words_raw if clean_entity_word(w)]
        entities_list.append(words)

    df["entities"] = entities_list

    return df


# -----------------------------
# Step 5: Aggregate to stock-date level
# -----------------------------

def aggregate_per_stock_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate row-level info into per (Stock_symbol, Date) level.

    For each stock-date bucket we compute:
        - velocity: number of headlines in the last 24h (here, that date)
        - entities_today: union of entities mentioned
        - embeddings_today: list of FinBERT embeddings h_k
        - max of event flags across headlines

    Parameters
    ----------
    df
        DataFrame containing news rows.
    Returns
    -------
        Aggregated DataFrame.
    """
    def combine_entities(series):
        combined = set()
        for entity_list in series:
            for e in entity_list:
                combined.add(e)
        return sorted(combined)

    def combine_embeddings(series):
        return list(series)
    def combine_titles(series):
        return list(series)

    grouped = df.groupby(["Stock_symbol", "Date"], as_index=False).agg(
        velocity=("Article_title", "size"),
        Article_title=("Article_title", combine_titles),
        entities_today=("entities", combine_entities),
        embeddings_today=("title_embedding", combine_embeddings),
        earnings_flag=("earnings_flag_row", "max"),
        guidance_flag=("guidance_flag_row", "max"),
        merger_flag=("merger_flag_row", "max"),
        rating_flag=("rating_flag_row", "max"),
    )

    return grouped


# -----------------------------
# Step 6: Compute novelty
# -----------------------------

def compute_novelty_per_stock(group: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """
    For a single stock, compute novelty as the fraction of entities that are new
    relative to the union of entities from the previous lookback_days.

    Parameters
    ----------
    group
        DataFrame for a single stock.
    lookback_days
        Number of days to look back for novelty calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'novelty' column.
    """
    group = group.sort_values("Date").copy()
    novelty_values = []

    for idx, row in group.iterrows():
        current_date = row["Date"]
        current_entities = set(row["entities_today"])

        # % of news entities (entities_today) vs previous lookback_days
        mask = (group["Date"] < current_date) & (
            group["Date"] >= current_date - pd.Timedelta(days=lookback_days)
        )
        past_rows = group.loc[mask]

        past_entities = set()
        for _, prow in past_rows.iterrows():
            for e in prow["entities_today"]:
                past_entities.add(e)

        if not current_entities:
            novelty = 0.0
        else:
            new_entities = current_entities - past_entities
            novelty = len(new_entities) / len(current_entities)

        novelty_values.append(novelty)

    group["novelty"] = novelty_values
    return group


def add_novelty(agg_df: pd.DataFrame, lookback_days: int = 3, num_workers: int = 4) -> pd.DataFrame:
    """
    Apply novelty computation per stock and concatenate.
    Uses multiprocessing to parallelize across stocks.

    Parameters
    ----------
    agg_df
        Aggregated DataFrame at stock-date level.
    lookback_days
        Number of days to look back for novelty calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'novelty' column.
    """
    groups = [subdf for _, subdf in agg_df.groupby("Stock_symbol")]

    print(f"Computing novelty in parallel for {len(groups)} stocks "
          f"with {num_workers} workers...")

    args = [(g, lookback_days) for g in groups]

    if num_workers <= 1:
        results = [compute_novelty_per_stock(g, lookback_days) for g in groups]
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.starmap(compute_novelty_per_stock, args)

    result = pd.concat(results, ignore_index=True)
    return result


# -----------------------------
# Step 7: Aggregate embeddings into z_news (recency-weighted pooling)
# -----------------------------

def aggregate_z_news(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Stock_symbol, Date), aggregate embeddings_today into a single
    z_news vector using the formula from the paper:

        alpha_k ∝ exp( w^T h_k - tau * Δt_k )
        z_news = Σ_k alpha_k * h_k

    which is a content-based softmax weighting over the headlines in the
    24-hour window instead of a plain average.

    Parameters
    ----------
    agg_df
        Aggregated DataFrame at stock-date level.

    Returns
    -------
    DataFrame with added 'z_news' column.
    """
    global _NEWS_ATTENTION_W

    z_vectors = []

    for _, row in agg_df.iterrows():
        emb_list = row["embeddings_today"]
        if not emb_list:
            z_vec = None
        else:
            # Stack embeddings -> matrix H of shape (num_headlines, hidden_dim)
            mat = np.stack(emb_list, axis=0).astype("float32")

            num_headlines, hidden_dim = mat.shape

            # Initialize w once, based on embedding dimension.
            if _NEWS_ATTENTION_W is None:
                # Random normal; scale 1/sqrt(dim) is a common choice.
                _NEWS_ATTENTION_W = (
                    np.random.normal(loc=0.0, scale=1.0 / np.sqrt(hidden_dim), size=(hidden_dim,))
                    .astype("float32")
                )
                print(f"[aggregate_z_news] Initialized attention vector w with dim={hidden_dim}")

            # Compute scores s_k = w^T h_k - tau * Δt_k
            # Here Δt_k = 0 for all k (single-day 24h window), so:
            # s_k = w^T h_k
            scores = mat @ _NEWS_ATTENTION_W
            
            # Softmax over scores to obtain alpha_k
            scores = scores - np.max(scores)
            weights = np.exp(scores)
            if weights.sum() == 0.0:
                weights = np.ones_like(weights) / float(len(weights))
            else:
                weights = weights / weights.sum()

            # Compute z_news = Σ_k alpha_k * h_k
            z_vec = (weights[:, None] * mat).sum(axis=0)

        z_vectors.append(z_vec)

    agg_df["z_news"] = z_vectors
    return agg_df


# -----------------------------
# Main pipeline (with chunked processing)
# -----------------------------

def built_news_features(
    ner_text_column: str,
    output_path: str,
    news_df: _t.Optional[pd.DataFrame] = None,
    file_path: _t.Optional[str] = None,
    directory: _t.Optional[str] = None,
    file_pattern: _t.Optional[str] = None,
    max_rows: _t.Optional[int] = None,
    chunk_size: _t.Optional[int] = 20_000,
) -> pd.DataFrame:
    """
    Main pipeline for extracting and aggregating news features for Leg B.

    Process:
        1. Compute headline embeddings h_k (FinBERT, frozen).
        2. Extract entities and event flags.
        3. Aggregate to (Stock_symbol, Date):
        4. Compute novelty (% new entities vs previous 3 days).
        5. Compute z_news using content-based softmax pooling
        6. Save final dataframe for use as Leg B input to the MLP.

    Parameters
    ----------
    ner_text_column
        Column to use for NER extraction and FinBERT embeddings (e.g. "Article_title").
    output_path
        Path where the pickle with final features will be saved.
    file_path
        Path to a single CSV file. (You are using this currently.)
    directory, file_pattern, max_rows, chunk_size
        Optional knobs if you want to iterate over many files.

    Returns
    -------
    pd.DataFrame
        Final DataFrame with aggregated news features.
    """

    # Create models ONCE
    print("Loading NER model...")
    ner_pipe = create_ner_pipeline()

    print("Loading encoder model for z_news (FinBERT)...")
    encoder_tokenizer, encoder_model = create_encoder()

    all_rows = []
    total_rows = 0

    # Process dataset in chunks
    print("Loading dataset in chunks...")
    df_iterator = iter_dataset_chunks(
        file_path=file_path,
        directory=directory,
        news_df=news_df,
        pattern=file_pattern,
        max_rows=max_rows,
        chunk_size=chunk_size,
    )

    for df_chunk in df_iterator:
        # Ensure the text column for NER exists in the dataframe
        assert ner_text_column in df_chunk.columns, (
            f"NER text column '{ner_text_column}' not found in DataFrame columns."
        )

        print(f"Processing chunk with {len(df_chunk)} rows...")
        total_rows += len(df_chunk)

        # 2. Add event flags at row level
        df_chunk = add_event_flags(df_chunk)

        # 3–4. Per-row entities and embeddings
        df_chunk = process_rows_with_ner_and_embeddings(
            df=df_chunk,
            ner_pipe=ner_pipe,
            tokenizer=encoder_tokenizer,
            encoder_model=encoder_model,
            text_column=ner_text_column,
        )

        all_rows.append(df_chunk)

    print(f"Total rows processed across chunks: {total_rows}")
    df = pd.concat(all_rows, ignore_index=True)

    # 6. Aggregate per stock-date
    print("Aggregating per (Stock_symbol, Date)...")
    df_stock_date = aggregate_per_stock_date(df=df)

    # 7. Add novelty (parallelized, default 4 workers -> num_workers = 4)
    print("Computing novelty...")
    df_stock_date = add_novelty(agg_df=df_stock_date)

    # 8. Aggregate embeddings into z_news using the paper's pooling formula
    print("Aggregating embeddings into z_news (recency-weighted pooling)...")
    df_stock_date = aggregate_z_news(agg_df=df_stock_date)

    # 9. Final columns for Leg B prototype
    output_cols = [
        "Date",
        "Stock_symbol",
        "Article_title",
        "velocity",
        "novelty",
        "earnings_flag",
        "guidance_flag",
        "merger_flag",
        "rating_flag",
        "z_news",
        "entities_today",
    ]
    df_output = df_stock_date[output_cols].sort_values(["Stock_symbol", "Date"])

    # 10. Save to a pickle or parquet since z_news is a vector
    print(f"Saving features to {output_path} ...")
    df_output.to_pickle(output_path)

    print("Done.")
    print("Columns in final output:")
    print(df_output.columns)

    return df_output