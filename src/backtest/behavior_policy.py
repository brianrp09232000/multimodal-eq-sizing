# src/backtest/behavior_policy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.guards import apply_portfolio_guards, PortfolioGuardParams

@dataclass
class BehaviorPolicyParams:
    """
    Container for all Behavior Policy hyperparameters.
    These defaults follow the project proposal but with a simplified interface.
    """
    # max action possible
    wmax: float = 0.02

    # epsilon-greedy nudge
    eps: float = 0.05

    # quatiles for acion mapping
    tau1_quantile: float = 0.50
    tau2_quantile: float = 0.80


# -------- 2) Thresholded z -> bucket policy --------

def _bucket_from_z(
    z_val: float,
    tau1: float,
    tau2: float,
    wmax: float,
) -> float:
    """
    Map z to one of {0, wmax/2, wmax} using tau1, tau2 thresholds.
    """
    if z_val <= 0:
        return 0.0
    if z_val < tau1:
        return 0.0
    elif z_val < tau2:
        return 0.5 * wmax
    else:
        return wmax


def _maybe_epsilon_nudge(
    bucket: float,
    wmax: float,
    rng: np.random.Generator,
    eps: float,
) -> float:
    """
    With probability eps, nudge bucket up or down by one step, if possible.

    Buckets are {0, wmax/2, wmax}.
    """
    if rng.random() > eps:
        return bucket

    buckets = [0.0, 0.5 * wmax, wmax]
    try:
        idx = buckets.index(bucket)
    except ValueError:
        idx = 0  # fallback, shouldn't really happen

    direction = rng.choice([-1, 1])
    new_idx = max(0, min(len(buckets) - 1, idx + direction))
    return buckets[new_idx]


def compute_behavior_actions_for_day(
    df_day: pd.DataFrame,
    z_col: str,
    params: Optional[BehaviorPolicyParams] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """
    Implement the dumb behavior policy for a single Date:

      - Compute P_x1, P_x2 of positive z.
      - Map z -> {0, wmax/2, wmax} via those thresholds.
      - Apply a small epsilon-greedy nudge.

    Returns a Series indexed by ticker with raw action weights.
    """
    if params is None:
          params = BehaviorPolicyParams()
      
    if rng is None:
        rng = np.random.default_rng(123)
      

    if z_col not in df_day.columns:
        raise ValueError(f"{z_col} not found in df_day columns")

    z = df_day[z_col]

    # only use positive z's to define thresholds
    pos_z = z[z > 0]
    if len(pos_z) == 0:
        # no positive signals: hold cash
        return pd.Series(0.0, index=df_day["ticker"].values, name="action_weight")

    tau1 = pos_z.quantile(params.tau1_quantile)
    tau2 = pos_z.quantile(params.tau2_quantile)
    base = df_day[z_col].apply(
        lambda z: _bucket_from_z(z, tau1=tau1, tau2=tau2, wmax=params.wmax)
    )

    actions = base.apply(
        lambda b: _maybe_epsilon_nudge(b, wmax=params.wmax, rng=rng, eps=params.eps)
    )
    actions.name = "action_weight_raw"

    return actions


# -------- 3) Run policy + guards across all dates --------

def run_behavior_policy_with_guards(
    df: pd.DataFrame,
    z_col,
    nav: float = 1_000_000.0,
    allow_short: bool = False,
    trading_enabled: bool = True,
    guard_params: Optional[PortfolioGuardParams] = None,
    policy_params: Optional[BehaviorPolicyParams] = None,
) -> pd.DataFrame:
    """
    Run the dumb behavior policy + portfolio guards over the full panel.

    Assumes df has at least:
      - 'Date'
      - 'ticker'
      - z_col         (signal used by policy, dummy or real)
      - 'sector'
      - 'spread_z'
      - 'Close'
      - 'adv_dollar'
      - 'vix_z'

    Returns a copy of df with:
      - 'action_weight_raw'   (before guards)
      - 'weight_after_guards' (after guard stack)
    """
    out = df.copy()
    out = out.sort_values(["Date", "ticker"])

    # keep track of previous (guarded) weights per ticker
    prev_w = pd.Series(dtype=float)
    prev_w.index.name = "ticker"

    all_action = []
    all_guarded = []

    rng = np.random.default_rng(2025)

    for date, day_df in out.groupby("Date", sort=True):
        # 1) behavior policy actions
        action_w = compute_behavior_actions_for_day(
            day_df,
            z_col=z_col
        )

        # 2) align prev weights
        tickers = day_df["ticker"].values
        idx = pd.Index(tickers, name="ticker")
        prev_w_day = prev_w.reindex(idx).fillna(0.0)

        # 3) build inputs for guards
        sectors = day_df["sector"]
        z_series = day_df[z_col]
        spread_z = day_df["spread_z"]
        prices = day_df["Close"]
        adv_dollar = day_df["adv_dollar"]
        vix_z_value = float(day_df["vix_z"].iloc[0])

        guarded_w, _shares = apply_portfolio_guards(
            action_weights=action_w,
            prev_weights=prev_w_day,
            sectors=sectors,
            z=z_series,
            spread_z=spread_z,
            prices=prices,
            adv_dollar=adv_dollar,
            nav=nav,
            vix_z=vix_z_value,
            allow_short=allow_short,
            trading_enabled=trading_enabled,
        )

        action_raw_day_df = action_w.rename("action_weight_raw").to_frame()
        action_raw_day_df["Date"] = date
        action_raw_day_df['ticker']=tickers
        all_action.append(action_raw_day_df)
    
        
        action_guards_day_df = guarded_w.rename("weight_after_guards").to_frame()
        action_guards_day_df["Date"] = date
        action_guards_day_df['ticker']=tickers
        all_guarded.append(action_guards_day_df)

        prev_w = guarded_w.copy()

    actions_concat = pd.concat(all_action)
    guarded_concat = pd.concat(all_guarded)
    out = (
        out
        .merge(actions_concat, on=["ticker", "Date"], how="inner")
        .merge(guarded_concat, on=["ticker", "Date"], how="inner")
    )
    return out
