from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.guards import apply_portfolio_guards, PortfolioGuardParams
from src.backtest.cost_models import turnover_and_cost  

@dataclass
class SimulationConfig:
    initial_nav: float = 1_000_000.0
    allow_short: bool = False
    trading_enabled: bool = True
    cost_bps: int = 10

def simulate_policy(
    df: pd.DataFrame,
    policy_fn: Callable[[pd.DataFrame, pd.Series], pd.Series],
    z_col: str,
    guard_params: Optional[PortfolioGuardParams] = None,
    sim_config: Optional[SimulationConfig] = None,
) -> pd.DataFrame:
    """
    Run a backtest over df using the provided policy + guards.

    df must have at least:
      - 'Date'
      - 'ticker'
      - 'Close'
      - 'excess_return' (or 'o2c_return' if not excess)
      - 'sector'
      - 'spread_z'
      - 'adv_dollar'
      - 'vix_z'

    policy_fn:
      (df_day, prev_weights) -> desired weights (raw buckets) per ticker

    Returns a DataFrame with one row per Date:
      - 'Date'
      - 'nav'
      - 'portfolio_return'
      - 'turnover'
      - maybe 'gross_exposure', etc.
    """
    if guard_params is None:
        guard_params = PortfolioGuardParams()
    if sim_config is None:
        sim_config = SimulationConfig()

    df = df.copy()
    df = df.sort_values(["Date", "ticker"])

    # Previous day portfolio weights per ticker
    prev_weights = pd.Series(dtype=float)
    prev_weights.index.name = "ticker"

    nav = sim_config.initial_nav
    daily_records = []

    for date, day_df in df.groupby("Date", sort=True):
        tickers = day_df["ticker"].values
        idx = pd.Index(tickers, name="ticker")

        # Align prev weights to today's universe
        prev_w_day = prev_weights.reindex(idx).fillna(0.0)

        # 1) Policy proposes raw action weights (buckets)
        action_weights = policy_fn(day_df, prev_w_day)

        # 2) Apply guards to get feasible weights
        sectors = day_df["sector"]
        z_series = day_df[z_col]
        spread_z = day_df["spread_z"]
        prices = day_df["Close"]
        adv_dollar = day_df["adv_dollar"]
        vix_z_value = float(day_df["VIX_z"].iloc[0])

        guarded_w, _shares = apply_portfolio_guards(
            action_weights=action_weights,
            prev_weights=prev_w_day,
            sectors=sectors,
            z=z_series,
            spread_z=spread_z,
            prices=prices,
            adv_dollar=adv_dollar,
            nav=nav,
            vix_z=vix_z_value,
            allow_short=sim_config.allow_short,
            trading_enabled=sim_config.trading_enabled,
            params=guard_params,
        )

        # 3) Compute turnover and transaction costs
        turnover, trading_cost = turnover_and_cost(prev_w_day, guarded_w, cost_bps=sim_config.cost_bps)

        # 4) Compute portfolio return for this day
        r_vec = day_df["excess_return"].values
        w_vec = guarded_w.values
        port_ret = np.dot(w_vec, r_vec) - trading_cost

        # 5) Update NAV
        nav = nav * (1.0 + port_ret)

        # 6) Save record for this day
        daily_records.append(
            {
                "Date": date,
                "nav": nav,
                "portfolio_return": port_ret,
                "turnover": turnover,
                "gross_exposure": np.sum(np.abs(w_vec)),
                "long_exposure": np.sum(np.clip(w_vec, 0, None)),
                "short_exposure": np.sum(np.clip(w_vec, None, 0)),
            }
        )

        # 7) Update prev_weights for next day
        prev_weights = guarded_w

    res = pd.DataFrame(daily_records).sort_values("Date").reset_index(drop=True)
    return res
