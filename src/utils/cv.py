import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Tuple, List, Callable, Any, Dict

def generate_yearly_oof(model_factory: Callable[[], Any],X: pd.DataFrame,y: pd.Series, dates: pd.Series, min_train_years: int = 2,n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generates out-of-fold predictions using a yearly expanding window
    
    model_factory: Function that returns a fresh, unfitted model instance
    X: Feature df
    y: Target series
    dates: Date series must be datetime objects
    min_train_years: Minimum number of years in the first training set
    n_jobs: Number of parallel workers -1 for all cores
    
    Returns:
        oof_preds: Concatenated predictions for all validation years
        oof_targets: Concatenated targets corresponding to oof_preds
        fold_stats: Metadata about each fold (train_years, val_year)
    """
    # This is the check for the dates to be datetime and extract years
    if not np.issubdtype(dates.dtype, np.datetime64):
        dates = pd.to_datetime(dates, utc=True)
    
    years = dates.dt.year
    unique_years = sorted(years.unique())
    
    # We need at least (min_train_years + 1) years of data to have a validation set
    if len(unique_years) <= min_train_years:
        raise ValueError(f"Not enough years of data. Need > {min_train_years} years, found {len(unique_years)}.")

    fold_args = []
    
    # Loop starts at the index of the first VALIDATION year
    # If min_train_years=2 (2011, 2012), index 2 is 2013 the first validation year
    for i in range(min_train_years, len(unique_years)):
        val_year = unique_years[i]
        train_years_list = unique_years[:i]
        
        # Create boolean masks for this fold
        # Train: All years strictly before the current validation year
        train_mask = years.isin(train_years_list)
        # Val: The specific validation year
        val_mask = years == val_year
        
        # Slice data
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        fold_args.append((
            model_factory,
            X_train, y_train,
            X_val, y_val,
            train_years_list, val_year
        ))

    print(f"Starting yearly walk-forward CV: {len(fold_args)} folds scheduled")
    print(f"First fold val year: {fold_args[0][-1]}")
    print(f"Last fold val year: {fold_args[-1][-1]}")

    # Execute folds in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_train_predict_year)(*args) for args in fold_args
    )
    
    # Unpack results
    all_preds = []
    all_targets = []
    fold_stats = []
    
    for preds, targets, stats in results:
        all_preds.append(preds)
        all_targets.append(targets)
        fold_stats.append(stats)

    # Concatenate to create the full OOF dataset for the calibrator
    if len(all_preds) > 0:
        oof_preds = np.concatenate(all_preds)
        oof_targets = np.concatenate(all_targets)
    else:
        # Fallback if no folds ran
        oof_preds = np.array([])
        oof_targets = np.array([])
    
    return oof_preds, oof_targets, fold_stats

def _train_predict_year(model_factory, X_train, y_train, X_val, y_val, train_years, val_year):
    """
    Helper function for a single yearly fold execution.
    """
    model = model_factory()
    model.fit(X_train, y_train)
    
    # For regression/excess return
    preds = model.predict(X_val)
    
    # Ensure 1D array
    preds = np.ravel(preds)
    y_val = np.ravel(y_val)
    
    stats = {
        "val_year": val_year,
        "train_years": f"{min(train_years)}-{max(train_years)}",
        "n_train": len(y_train),
        "n_val": len(y_val)
    }
    
    return preds, y_val, stats