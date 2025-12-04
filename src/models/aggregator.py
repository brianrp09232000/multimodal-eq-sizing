from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

@dataclass
class AggregatorParams:
    """
    Container for all aggregator hyperparameters.
    """
    # sets the scale of what we consider “big disagreement”.
    theta: float = 0.0006  # 6 bps
    
    # controls how sharp the penalty is.
    p:int = 2

    # controls how much weight we assign to each tower.
    w: float = 0.5

def apply_aggregator(
    df: pd.DataFrame,
    px_col: str = "r_px_cal",
    news_col: str = "r_news_cal",
    flag_col: str = "news_flag",
    params: Optional[AggregatorParams] = None,
    out_col: str = "r_tilde",
) -> pd.DataFrame:
    """
    Equal-weight + disagreement shrink aggregator.

    - If news_flag == 1: use both legs.
    - If news_flag == 0: fall back to price only.
    """
    if params is None:
        params = AggregatorParams()
    
    df = df.copy()

    px   = df[px_col]
    news = df[news_col]
    flag = df[flag_col]

    # Equal-weight average (or price-only if no news)
    r_bar = np.where(
        (flag == 1) & news.notna(),
        params.w * px + (1-params.w) * news,   # price + news
        px                                     # price only
    )

    # Disagreement (0 if only one leg)
    d = np.where(
        (flag == 1) & news.notna(),
        np.abs(px - news),
        0.0
    )

    # Shrink factor
    shrink = 1.0 / (1.0 + (d / params.theta) ** params.p)

    # Final pre-calibrated blend
    df[out_col] = r_bar * shrink

    # Keep for diagnostics
    df["agg_r_bar"] = r_bar
    df["agg_disagreement"] = d
    df["agg_shrink"] = shrink
  
    return df
 
