from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PortfolioGuardParams:
    """
    Container for all portfolio guard hyperparameters.
    These defaults follow the project proposal but with a simplified interface.
    """
    # G1: no-trade band
    no_trade_band: float = 0.0015  # 0.15% in weight space

    # G2: per-name cap
    per_name_cap: float = 0.02     # 2% of NAV

    # G3: sector cap
    sector_cap: float = 0.15       # 15% of NAV

    # G4: gross exposure cap
    gross_exposure_cap: float = 0.60  # 60% of NAV

    # G5: turnover budget
    turnover_cap: float = 0.20     # 20% of NAV per rebalance

    # G6: liquidity / participation
    max_notional_participation: float = 0.08  # 8% of ADV
    max_spread_z: float = 1.5
    min_price: float = 2.0

    # G7: regime gating
    vix_abs_z_gate: float = 2.0
    vix_shrink: float = 0.5

    # G11: smoothing
    smoothing_alpha: float = 0.3  # EMA weight on previous holdings


# --------------------------
# Individual guard functions
# --------------------------

def guard_g1_no_trade_band(
    target_w: pd.Series,
    prev_w: pd.Series,
    band: float,
) -> pd.Series:
    """
    G1. No-Trade Band: if |target - prev| < band, hold previous weight.
    """
    prev_w = prev_w.reindex(target_w.index).fillna(0.0)
    delta = target_w - prev_w
    hold_mask = delta.abs() < band
    adjusted = target_w.copy()
    adjusted[hold_mask] = prev_w[hold_mask]
    return adjusted


def guard_g2_per_name_cap(
    w: pd.Series,
    cap: float,
    long_only: bool = True,
) -> pd.Series:
    """
    G2. Per-Name Cap: clip absolute exposure; long-only optionally floors at 0.
    """
    capped = w.clip(lower=-cap, upper=cap)
    if long_only:
        capped = capped.clip(lower=0.0)
    return capped


def guard_g3_sector_caps(
    w: pd.Series,
    sectors: pd.Series,
    sector_cap: float,
) -> pd.Series:
    """
    G3. Sector Caps: for each sector s, enforce sum_{i in s} w_i <= sector_cap
    via proportional scaling within the sector.
    """
    adjusted = w.copy()
    sectors = sectors.reindex(w.index)
    sector_sums = adjusted.groupby(sectors).sum()
    for sector, total in sector_sums.items():
        if pd.isna(sector) or total <= sector_cap:
            continue
        scale = sector_cap / total if total != 0 else 1.0
        idx = sectors[sectors == sector].index
        adjusted.loc[idx] = adjusted.loc[idx] * scale
    return adjusted


def guard_g4_gross_exposure_cap(
    w: pd.Series,
    gross_cap: float,
) -> pd.Series:
    """
    G4. Gross Exposure Cap: enforce sum |w_i| <= gross_cap
    via proportional scaling across all names.
    """
    gross = w.abs().sum()
    if gross <= gross_cap or gross == 0:
        return w
    scale = gross_cap / gross
    return w * scale


def guard_g5_turnover_budget(
    desired_w: pd.Series,
    prev_w: pd.Series,
    z: pd.Series,
    spread_z: pd.Series,
    turnover_cap: float,
) -> pd.Series:
    """
    G5. Turnover Budget: sum_i |w_i - w_{i,-1}| <= turnover_cap.
    Greedy allocation of turnover starting from highest |z| / spread_z.

    Parameters
    ----------
    desired_w : pd.Series
        Weights after G1–G4.
    prev_w : pd.Series
        Executed weights at t-1.
    z : pd.Series
        Risk-normalized edge per name (z-score).
    spread_z : pd.Series
        Trading-cost proxy per name (z-score); higher = more expensive.
    """
    desired_w = desired_w.copy()
    prev_w = prev_w.reindex(desired_w.index).fillna(0.0)
    z = z.reindex(desired_w.index).fillna(0.0)
    spread_z = spread_z.reindex(desired_w.index).fillna(0.0)

    full_delta = desired_w - prev_w
    full_turnover = full_delta.abs().sum()
    if full_turnover <= turnover_cap:
        return desired_w

    remaining = turnover_cap
    new_w = prev_w.copy()

    # Benefit proxy: larger |z| and lower spread_z preferred
    cost_adjusted_spread = spread_z.abs().replace(0, np.nan)
    score = z.abs() / cost_adjusted_spread
    score = score.fillna(z.abs())  # fallback to |z|

    order = score.sort_values(ascending=False).index

    for name in order:
        if remaining <= 0:
            break
        desired = desired_w.loc[name]
        current = new_w.loc[name]
        delta = desired - current
        needed = abs(delta)
        if needed == 0:
            continue
        allowed = min(needed, remaining)
        step = np.sign(delta) * allowed
        new_w.loc[name] = current + step
        remaining -= abs(step)

    return new_w


def guard_g6_liquidity_participation(
    w: pd.Series,
    prev_w: pd.Series,
    prices: pd.Series,
    adv_dollar: pd.Series,
    nav: float,
    spread_z: pd.Series,
    max_participation: float,
    max_spread_z: float,
    min_price: float,
) -> pd.Series:
    """
    G6. Liquidity & Participation:
    - Skip trades where spread_z > max_spread_z or price < min_price.
    - Enforce notional vs ADV participation limit; if breached, revert to prev_w.

    Implementation detail: if a trade breaches any constraint, we revert that
    name to its previous weight (simple and conservative).
    """
    w = w.copy()
    prev_w = prev_w.reindex(w.index).fillna(0.0)
    prices = prices.reindex(w.index)
    spread_z = spread_z.reindex(w.index)
    adv_dollar = adv_dollar.reindex(w.index)

    # Skip names that are too illiquid or too cheap
    skip_mask = (spread_z > max_spread_z) | (prices < min_price)
    w[skip_mask] = prev_w[skip_mask]

    # Participation vs ADV
    delta_w = (w - prev_w).abs()
    notional = delta_w * nav
    max_notional = adv_dollar * max_participation

    # Only enforce where we actually have ADV
    valid_adv = adv_dollar > 0
    breach = valid_adv & (notional > max_notional)
    w[breach] = prev_w[breach]

    return w


def guard_g7_regime_gating(
    w: pd.Series,
    vix_z: float,
    gate_abs_z: float,
    shrink: float,
) -> pd.Series:
    """
    G7. Regime Gating: when |VIX_z| > gate_abs_z, shrink targets by `shrink`.
    """
    if abs(vix_z) > gate_abs_z:
        return w * shrink
    return w


def guard_g9_shorting(
    w: pd.Series,
    allow_short: bool,
) -> pd.Series:
    """
    G9. Shorting:
    - If allow_short is False: clip shorts to 0.
    - If allow_short is True: leave as is (no borrow constraints here).
    """
    if not allow_short:
        return w.clip(lower=0.0)
    return w


def guard_g10_rounding(
    w: pd.Series,
    prices: pd.Series,
    nav: float,
    lot_size: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    G10. Rounding only (no price-band logic):

    - Convert weights to share counts at current prices & NAV.
    - Round to nearest `lot_size`.
    - Convert back to weights.

    Returns
    -------
    rounded_w : pd.Series
        Weights corresponding to rounded share counts.
    shares : pd.Series
        Rounded share counts (useful for orders).
    """
    prices = prices.reindex(w.index).replace(0, np.nan)
    print("\nPrices on first date:")
    print(prices)
    
    print("\nWeights before rounding (w):")
    print(w)
    
    dollar_exposure = w * nav
    raw_shares = dollar_exposure / prices
    print("\nDollar exposure:", dollar_exposure)
    print("\nRaw shares:", raw_shares)

    rounded_shares = (raw_shares / lot_size).round() * lot_size
    rounded_w = (rounded_shares * prices) / nav
    rounded_w = rounded_w.fillna(0.0)
    return rounded_w, rounded_shares.fillna(0)


def guard_g11_smoothing(
    w: pd.Series,
    prev_w: pd.Series,
    alpha: float,
) -> pd.Series:
    """
    G11. Smoothing: EMA toward guarded target.
    w_t <- alpha * w_{t-1} + (1 - alpha) * w_t
    """
    prev_w = prev_w.reindex(w.index).fillna(0.0)
    return alpha * prev_w + (1.0 - alpha) * w


def guard_g12_feasible_projection(
    w: pd.Series,
    sectors: pd.Series,
    params: PortfolioGuardParams,
    long_only: bool = True,
) -> pd.Series:
    """
    G12. Feasible Projection (Quadratic Heuristic):

    Implemented via sequential proportional scalings:
    - per-name caps
    - sector caps
    - gross exposure caps
    """
    w_proj = guard_g2_per_name_cap(w, params.per_name_cap, long_only=long_only)
    w_proj = guard_g3_sector_caps(w_proj, sectors, params.sector_cap)
    w_proj = guard_g4_gross_exposure_cap(w_proj, params.gross_exposure_cap)
    return w_proj


def guard_g13_kill_switch(
    trading_enabled: bool,
    w: pd.Series,
    prev_w: pd.Series,
) -> pd.Series:
    """
    G13. Kill-Switch:

    If trading_enabled is False, we simply hold previous weights.
    """
    if trading_enabled:
        return w
    prev_w = prev_w.reindex(w.index).fillna(0.0)
    return prev_w


# ---------------------------------------
# Main convenience wrapper: G1–G13 stack
# ---------------------------------------

def apply_portfolio_guards(
    action_weights: pd.Series,
    prev_weights: pd.Series,
    sectors: pd.Series,
    z: pd.Series,
    spread_z: pd.Series,
    prices: pd.Series,
    adv_dollar: pd.Series,
    nav: float,
    vix_z: float,
    allow_short: bool = False,
    trading_enabled: bool = True,
    params: Optional[PortfolioGuardParams] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply the main portfolio guard stack (G1–G13) and return:

    - final_weights: guarded, projected, and rounded target weights for t
    - final_shares: share counts implied by final_weights at `nav`

    Required inputs:
    - action_weights: model / RL target weights
    - prev_weights: last executed weights
    - sectors: sector label per ticker
    - z: alpha z-score per ticker
    - spread_z: trading-cost proxy per ticker
    - prices: current prices (Close)
    - adv_dollar: ADV in dollars per ticker
    - nav: portfolio NAV
    - vix_z: VIX z-score for regime gating
    - allow_short: whether shorts allowed
    - trading_enabled: kill-switch flag
    """
    if params is None:
        params = PortfolioGuardParams()

    # Align everything to the same index
    prev_weights = prev_weights.fillna(0.0)
    sectors = sectors
    z = z.fillna(0.0)
    spread_z = spread_z.fillna(0.0)
    prices = prices
    adv_dollar = adv_dollar

    # ---- G1: no-trade band
    w = guard_g1_no_trade_band(action_weights, prev_weights, params.no_trade_band)

    # ---- G2: per-name cap
    w = guard_g2_per_name_cap(w, params.per_name_cap, long_only=not allow_short)

    # ---- G3: sector caps
    w = guard_g3_sector_caps(w, sectors, params.sector_cap)

    # ---- G4: gross exposure cap
    w = guard_g4_gross_exposure_cap(w, params.gross_exposure_cap)

    # ---- G5: turnover budget
    w = guard_g5_turnover_budget(w, prev_weights, z, spread_z, params.turnover_cap)

    # ---- G6: liquidity & participation
    w = guard_g6_liquidity_participation(
        w=w,
        prev_w=prev_weights,
        prices=prices,
        adv_dollar=adv_dollar,
        nav=nav,
        spread_z=spread_z,
        max_participation=params.max_notional_participation,
        max_spread_z=params.max_spread_z,
        min_price=params.min_price,
    )

    # ---- G7: regime gating
    w = guard_g7_regime_gating(
        w=w,
        vix_z=vix_z,
        gate_abs_z=params.vix_abs_z_gate,
        shrink=params.vix_shrink,
    )

    # ---- G9: shorting
    w = guard_g9_shorting(w, allow_short=allow_short)

    # ---- G10: rounding (no price bands)
    rounded_w, shares = guard_g10_rounding(
        w=w,
        prices=prices,
        nav=nav,
        lot_size=1,  # change if you want odd-lot handling
    )

    # ---- G11: smoothing
    smoothed_w = guard_g11_smoothing(
        w=rounded_w,
        prev_w=prev_weights,
        alpha=params.smoothing_alpha,
    )

    # ---- G12: feasible projection
    projected_w = guard_g12_feasible_projection(
        w=smoothed_w,
        sectors=sectors,
        params=params,
        long_only=not allow_short,
    )

    # ---- G13: kill-switch
    final_w = guard_g13_kill_switch(
        trading_enabled=trading_enabled,
        w=projected_w,
        prev_w=prev_weights,
    )

    return final_w, shares
