import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dataclasses import dataclass
from typing import Tuple, List, Callable, Any, Dict


# ---------------------------------
# Shared fold definition / splitter
# ---------------------------------

@dataclass
class YearlyFold:
    train_years: List[int]
    val_year: int
    train_mask: pd.Series
    val_mask: pd.Series


def _prepare_yearly_folds(
    dates: pd.Series,
    min_train_years: int = 2,
) -> List[YearlyFold]:
    """
    Internal helper to prepare yearly walk-forward folds.
    """
    # ✅ Always coerce to datetime with UTC
    dates = pd.to_datetime(dates, utc=True)

    years = dates.dt.year
    unique_years = sorted(years.unique())

    if len(unique_years) <= min_train_years:
        raise ValueError(
            f"Not enough years of data. Need > {min_train_years}, "
            f"found {len(unique_years)}."
        )

    folds: List[YearlyFold] = []

    for i in range(min_train_years, len(unique_years)):
        val_year = unique_years[i]
        train_years_list = unique_years[:i]

        train_mask = years.isin(train_years_list)
        val_mask = years == val_year

        folds.append(
            YearlyFold(
                train_years=train_years_list,
                val_year=val_year,
                train_mask=train_mask,
                val_mask=val_mask,
            )
        )

    return folds


# --------------------------
# Public RL-friendly splitter
# --------------------------

def make_yearly_walkforward_splits(
    dates: pd.Series,
    min_train_years: int = 2,
) -> List[YearlyFoldMasks]:
    """
    Build yearly expanding-window train/val masks for RL or any custom use.

    Parameters
    ----------
    dates : pd.Series
        Datetime-like series aligned with your DataFrame rows.
    min_train_years : int
        Minimum number of years in the first training set.

    Returns
    -------
    folds : List[YearlyFoldMasks]
        Folds containing train_years, val_year, train_mask, val_mask.
    """
    folds = _prepare_yearly_folds(dates, min_train_years=min_train_years)

    print(
        f"Built {len(folds)} yearly walk-forward folds "
        f"(first val year={folds[0].val_year}, last val year={folds[-1].val_year})"
    )

    return folds


# ------------------------------
# OOF generator using same logic
# ------------------------------

def generate_yearly_oof(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    min_train_years: int = 2,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generates out-of-fold predictions using a yearly expanding window.

    Parameters
    ----------
    model_factory : Callable[[], Any]
        Function that returns a fresh, unfitted model instance.
    X : pd.DataFrame
        Feature dataframe.
    y : pd.Series
        Target series.
    dates : pd.Series
        Date series aligned with X and y; must be datetime-like.
    min_train_years : int
        Minimum number of years in the first training set.
    n_jobs : int
        Number of parallel workers (-1 for all cores).

    Returns
    -------
    oof_preds : np.ndarray
        Concatenated predictions for all validation years.
    oof_targets : np.ndarray
        Concatenated targets corresponding to oof_preds.
    fold_stats : List[Dict]
        Metadata about each fold (train_years, val_year, sizes).
    """
    folds = _prepare_yearly_folds(dates, min_train_years=min_train_years)

    if len(folds) == 0:
        raise ValueError("No folds created — check your date range and min_train_years.")

    print(f"Starting yearly walk-forward CV: {len(folds)} folds scheduled")
    print(f"First fold val year: {folds[0].val_year}")
    print(f"Last fold val year: {folds[-1].val_year}")

    fold_args = []
    for fold in folds:
        train_mask = fold.train_mask
        val_mask = fold.val_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        fold_args.append(
            (
                model_factory,
                X_train, y_train,
                X_val, y_val,
                fold.train_years, fold.val_year,
            )
        )

    # Execute folds in parallel (same as before)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_train_predict_year)(*args) for args in fold_args
    )

    # Unpack results
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    fold_stats: List[Dict] = []

    for preds, targets, stats in results:
        all_preds.append(preds)
        all_targets.append(targets)
        fold_stats.append(stats)

    if len(all_preds) > 0:
        oof_preds = np.concatenate(all_preds)
        oof_targets = np.concatenate(all_targets)
    else:
        oof_preds = np.array([])
        oof_targets = np.array([])

    return oof_preds, oof_targets, fold_stats


def _train_predict_year(
    model_factory: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    train_years: List[int],
    val_year: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Helper function for a single yearly fold execution.
    """
    model = model_factory()
    model.fit(X_train, y_train)

    # For regression/excess return
    preds = model.predict(X_val)

    # Ensure 1D arrays
    preds = np.ravel(preds)
    y_val = np.ravel(y_val)

    stats = {
        "val_year": val_year,
        "train_years": f"{min(train_years)}-{max(train_years)}",
        "n_train": len(y_train),
        "n_val": len(y_val),
    }

    return preds, y_val, stats
