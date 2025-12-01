from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _drawdown_curve(nav: pd.Series) -> pd.Series:
    """
    Compute drawdown series from a NAV curve.
    """
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    return dd


@dataclass
class BacktestSummary:
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    avg_daily_return: float
    avg_daily_turnover: float
    ann_turnover: float
    avg_gross_exposure: float


def summarize_backtest(
    bt: pd.DataFrame,
    return_col: str = "portfolio_return",
    nav_col: str = "nav",
    turnover_col: str = "turnover",
    gross_col: str = "gross_exposure",
) -> BacktestSummary:
    """
    Compute core performance stats from simulator output.
    """
    bt = bt.sort_values("Date").copy()

    r = bt[return_col].dropna()
    nav = bt[nav_col].dropna()

    if len(r) == 0:
        raise ValueError("No returns in backtest dataframe")

    # 1) CAGR (compound annual growth rate)
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    n_days = len(bt)
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    # 2) Annualized volatility
    daily_vol = r.std(ddof=1)
    ann_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    # 3) Sharpe (assuming rf ~ 0)
    sharpe = cagr / ann_vol if ann_vol > 0 else np.nan

    # 4) Max drawdown & Calmar
    dd = _drawdown_curve(nav)
    max_dd = dd.min()  # negative number
    calmar = -cagr / max_dd if max_dd < 0 else np.nan

    # 5) Hit rate / avg daily return
    hit_rate = (r > 0).mean()
    avg_daily_return = r.mean()

    # 6) Turnover: avg daily and annualized
    if turnover_col in bt.columns:
        turnover = bt[turnover_col].fillna(0.0)
        avg_daily_turnover = turnover.mean()
        # annualized turnover ~ sum of daily turnover over a year
        ann_turnover = avg_daily_turnover * TRADING_DAYS_PER_YEAR
    else:
        avg_daily_turnover = np.nan
        ann_turnover = np.nan

    # 7) Average gross exposure
    if gross_col in bt.columns:
        avg_gross_exposure = bt[gross_col].mean()
    else:
        avg_gross_exposure = np.nan

    return BacktestSummary(
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        hit_rate=hit_rate,
        avg_daily_return=avg_daily_return,
        avg_daily_turnover=avg_daily_turnover,
        ann_turnover=ann_turnover,
        avg_gross_exposure=avg_gross_exposure,
    )


def summary_to_series(summary: BacktestSummary, name: str) -> pd.Series:
    """
    Convert BacktestSummary dataclass to a Pandas Series,
    to make it easy to build comparison tables.
    """
    s = pd.Series({
        "CAGR": summary.cagr,
        "AnnVol": summary.ann_vol,
        "Sharpe": summary.sharpe,
        "MaxDD": summary.max_drawdown,
        "Calmar": summary.calmar,
        "HitRate": summary.hit_rate,
        "AvgDailyRet": summary.avg_daily_return,
        "AvgDailyTurnover": summary.avg_daily_turnover,
        "AnnTurnover": summary.ann_turnover,
        "AvgGrossExposure": summary.avg_gross_exposure,
    })
    s.name = name
    return s
