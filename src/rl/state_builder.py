from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.backtest.simulator import SimulationConfig

def ensure_z_column(
    df: pd.DataFrame,
    z_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Ensure there is a column z_col in df.

    If it doesn't exist, create a dummy N(0,1) z. This lets you
    develop the RL pipeline before the real alpha is wired in.
    """
    if z_col in df.columns:
        return df

    rng = np.random.default_rng(seed)
    out = df.copy()
    out[z_col] = rng.normal(loc=0.0, scale=1.0, size=len(out))
    print("Using randomly generated dummy z column")
    return out


def build_rl_dataset(
    df: pd.DataFrame,
    z_col: str,
    reward_return_col: str = "excess_return",
    action_col: str = "action_weight_raw",
    weight_col: str = "weight_after_guards",
    extra_state_cols: Optional[Iterable[str]] = None,
    lambda_risk: float = 0.1,
    sim_config: Optional[SimulationConfig] = None
) -> pd.DataFrame:
    """
    Build offline RL tuples from a panel DataFrame.

    Assumes df has:
      - 'Date'
      - 'ticker'
      - reward_return_col   (realized next-day excess O2C return)
      - z_col               (signal used by behavior policy)
      - action_col          (raw bucket 0 / wmax/2 / wmax)
      - weight_col          (executed weight after guards)

    Returns a DataFrame with:
      - Date, ticker
      - reward, action, done
      - state_* columns
      - next_state_* columns
    """
    if extra_state_cols is None:
        extra_state_cols = []

    if sim_config is None:
        sim_config = SimulationConfig()

    data = df.copy()
    data = data.sort_values(["ticker", "Date"])

    # ------- 1) state definition -------
    base_state_cols = [z_col, "VIX_z", "spread_z", weight_col]
    state_cols = list(dict.fromkeys(base_state_cols + list(extra_state_cols)))

    # ------- 2) previous weight per ticker -------
    data["prev_weight"] = (
        data.groupby("ticker")[weight_col]
            .shift(1)
            .fillna(0.0)
    )

    # ------- 3) next-day return per ticker -------
    data["next_return"] = (
        data.groupby("ticker")[reward_return_col]
            .shift(-1)
    )

    # ------- 4) reward_t = w_t * r_{t+1} - c|Δw| - λ w_t^2 -------
    c = sim_config.cost_bps / 10_000.0  # bps -> decimal
    w_t = data[weight_col]
    w_prev = data["prev_weight"]
    dw = w_t - w_prev
    r_next = data["next_return"]

    data["reward"] = (
        w_t * r_next
        - c * dw.abs()
        - lambda_risk * (w_t ** 2)
    )

    # ------- 5) state and next_state columns -------
    for col in state_cols:
        data[f"state_{col}"] = data[col]
        data[f"next_state_{col}"] = (
            data.groupby("ticker")[col]
                .shift(-1)
        )

    # ------- 6) action & done flag -------
    data["action"] = data[action_col]
    data["done"] = data["next_return"].isna().astype(int)

    # For CQL / Q-learning, usually drop terminal transitions that have no next_state
    mask_valid = data["next_return"].notna()
    rl_df = data.loc[mask_valid].copy()

    keep_cols = (
        ["Date", "ticker", "reward", "action", "done"]
        + [f"state_{cname}" for cname in state_cols]
        + [f"next_state_{cname}" for cname in state_cols]
    )

    rl_df = rl_df[keep_cols]

    return rl_df
