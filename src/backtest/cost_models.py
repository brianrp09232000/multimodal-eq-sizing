from typing import Tuple

import pandas as pd


def compute_turnover(
    prev_weights: pd.Series,
    new_weights: pd.Series,
) -> float:
    """
    Compute one-day portfolio turnover as a fraction of NAV.

    Both prev_weights and new_weights should be aligned on the same ticker index.
    """
    prev_aligned = prev_weights.reindex(new_weights.index).fillna(0.0)
    dw = (new_weights - prev_aligned).abs()
    turnover = float(dw.sum())
    return turnover


def turnover_cost(
    turnover: float,
    cost_bps: float = 10.0,
) -> float:
    """
    Linear transaction cost model:

        cost = turnover * (cost_bps / 10_000)

    where:
      - turnover is fraction of NAV traded (e.g. 0.20 for 20%)
      - cost_bps is per-dollar trading cost in basis points
    """
    cost_per_dollar = cost_bps / 10_000.0
    return turnover * cost_per_dollar


def turnover_and_cost(
    prev_weights: pd.Series,
    new_weights: pd.Series,
    cost_bps: float = 10.0,
) -> Tuple[float, float]:
    """
    Convenience function: return (turnover, cost) for given weight vectors.
    """
    turnover = compute_turnover(prev_weights, new_weights)
    cost = turnover_cost(turnover, cost_bps=cost_bps)
    return turnover, cost
