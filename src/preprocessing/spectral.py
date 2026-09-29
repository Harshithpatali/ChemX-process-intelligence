"""Spectral preprocessing transforms for ChemX.

Each transform is documented with mathematical definition and
when it helps / harms chemometric models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: row-wise mean-center and scale by std.

    x'_i = (x_i - mean(x)) / std(x)

    Useful for scatter correction in NIR. Can harm if absolute intensity
    carries chemical information.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return (X - mean) / std


class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction against a reference spectrum.

    Fit: compute mean spectrum as reference.
    Transform: for each sample fit a + b * ref to spectrum, correct (x - a) / b.
    """

    def __init__(self):
        self.reference_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        ref = self.reference_
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            # linear regression: X[i] ≈ a + b * ref
            A = np.vstack([np.ones_like(ref), ref]).T
            coef, _, _, _ = np.linalg.lstsq(A, X[i], rcond=None)
            a, b = coef
            if abs(b) < 1e-12:
                b = 1.0
            out[i] = (X[i] - a) / b
        return out


class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """Savitzky-Golay smoothing / derivative.

    Parameters
    ----------
    window : odd int
    polyorder : int < window
    deriv : 0 (smooth), 1 (1st derivative), 2 (2nd derivative)
    """

    def __init__(self, window: int = 15, polyorder: int = 2, deriv: int = 0):
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return savgol_filter(
            X, window_length=self.window, polyorder=self.polyorder, deriv=self.deriv, axis=1
        )


class MeanCenter(BaseEstimator, TransformerMixin):
    """Column-wise mean centering (standard for PCA/PLS)."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        self.mean_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) - self.mean_


class Autoscale(BaseEstimator, TransformerMixin):
    """Column-wise mean-center and scale to unit variance."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_


def apply_preprocessing(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str = "snv",
) -> Tuple[np.ndarray, np.ndarray, object]:
    """Convenience: fit on train, transform both.

    method: 'raw' | 'snv' | 'msc' | 'sg1' | 'sg2' | 'snv_sg1'
    """
    if method == "raw":
        return X_train.copy(), X_test.copy(), None

    if method == "snv":
        tf = SNV()
        return tf.fit_transform(X_train), tf.transform(X_test), tf

    if method == "msc":
        tf = MSC()
        return tf.fit_transform(X_train), tf.transform(X_test), tf

    if method == "sg1":
        tf = SavitzkyGolay(window=15, polyorder=2, deriv=1)
        return tf.fit_transform(X_train), tf.transform(X_test), tf

    if method == "sg2":
        tf = SavitzkyGolay(window=15, polyorder=2, deriv=2)
        return tf.fit_transform(X_train), tf.transform(X_test), tf

    if method == "snv_sg1":
        snv = SNV()
        sg = SavitzkyGolay(window=15, polyorder=2, deriv=1)
        Xt = sg.fit_transform(snv.fit_transform(X_train))
        Xs = sg.transform(snv.transform(X_test))
        return Xt, Xs, (snv, sg)

    raise ValueError(f"Unknown preprocessing method: {method}")
