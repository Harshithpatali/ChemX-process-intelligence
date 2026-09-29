"""Bayesian soft sensor: probabilistic prediction with credible intervals.

Starts with Bayesian linear regression (closed-form conjugate normal-inverse-gamma
style via sklearn BayesianRidge) for transparency and calibration.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler


class BayesianSoftSensor:
    """Probabilistic soft sensor: p(y | X) via Bayesian Ridge regression.

    Outputs point prediction + approximate 95% predictive interval.
    """

    def __init__(self, n_features_pca: Optional[int] = 20):
        self.n_features_pca = n_features_pca
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()
        self.model_ = BayesianRidge(compute_score=True)
        self.pca_components_: Optional[np.ndarray] = None
        self.pca_mean_: Optional[np.ndarray] = None
        self.fitted_ = False

    def _reduce(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Optional PCA reduction for high-dimensional spectra."""
        if self.n_features_pca is None or X.shape[1] <= self.n_features_pca:
            return X
        from sklearn.decomposition import PCA
        if fit:
            pca = PCA(n_components=min(self.n_features_pca, X.shape[0] - 1, X.shape[1]))
            Z = pca.fit_transform(X)
            self.pca_components_ = pca.components_
            self.pca_mean_ = pca.mean_
            return Z
        Xc = X - self.pca_mean_
        return Xc @ self.pca_components_.T

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianSoftSensor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        Xs = self.scaler_X_.fit_transform(X)
        Z = self._reduce(Xs, fit=True)
        ys = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()
        self.model_.fit(Z, ys)
        self.fitted_ = True
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_X_.transform(X)
        Z = self._reduce(Xs, fit=False)
        ys_mean, ys_std = self.model_.predict(Z, return_std=True)
        y_mean = self.scaler_y_.inverse_transform(ys_mean.reshape(-1, 1)).ravel()
        # Approximate std in original scale
        y_std = ys_std * self.scaler_y_.scale_[0]
        if return_std:
            return y_mean, y_std
        return y_mean, None

    def predict_interval(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """Return mean and an approximate (1-alpha) Gaussian predictive interval."""
        from scipy import stats
        mean, std = self.predict(X, return_std=True)
        z = stats.norm.ppf(1 - alpha / 2)
        lower = mean - z * std
        upper = mean + z * std
        return {
            "prediction": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
            "alpha": alpha,
        }

    def coverage(
        self, X: np.ndarray, y: np.ndarray, alpha: float = 0.05
    ) -> float:
        """Empirical coverage of the predictive intervals."""
        interval = self.predict_interval(X, alpha=alpha)
        y = np.asarray(y).ravel()
        inside = (y >= interval["lower"]) & (y <= interval["upper"])
        return float(inside.mean())
