import pickle
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from typing import Union

class IsotonicCalibrator:
    """
    A wrapper for Isotonic Regression to calibrate raw model scores to 
    expected excess returns.
    
    Workflow:
    1. Generate OOF predictions using cv_utils.generate_yearly_oof
    2. fit(oof_preds, oof_targets)
    3. predict(test_raw_scores)
    """
    
    def __init__(self, out_of_bounds: str = "clip"):
        """
        out_of_bounds: 'clip' restricts outputs to min/max seen in training
        """
        # increasing=True enforces strict monotonicity aka higher score = higher return
        self.iso_reg = IsotonicRegression(increasing=True, out_of_bounds=out_of_bounds)
        self.is_fitted = False

    def fit(self, raw_scores: np.ndarray, targets: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fits the isotonic regression on (raw_score, target) pairs
        """
        # Flatten inputs to ensure 1D arrays
        X = np.ravel(raw_scores)
        y = np.ravel(targets)
        
        self.iso_reg.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Applies the learned calibration mapping to new data
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator NEEDS to be fitted before calling predict()")
            
        X = np.ravel(raw_scores)
        return self.iso_reg.predict(X)

    def save(self, filepath: Union[str, Path]):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.iso_reg, f)
        print(f"Calibrator saved to {path}")

    def load(self, filepath: Union[str, Path]):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator not found at {path}")
        with open(path, 'rb') as f:
            self.iso_reg = pickle.load(f)
        self.is_fitted = True
        print(f"Calibrator loaded from {path}")