"""Principal Component Analysis for ChemX with Hotelling T² and Q residuals."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA as SkPCA


class ChemometricPCA:
    """PCA with process-monitoring statistics.

    Model: X ≈ T Pᵀ + E
    - T: scores
    - P: loadings
    - Hotelling's T²: distance within model space
    - Q residual: squared prediction error (SPE)
    """

    def __init__(self, n_components: int = 10, center: bool = True):
        self.n_components = n_components
        self.center = center
        self.pca_: Optional[SkPCA] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.t2_limit_: Optional[float] = None
        self.q_limit_: Optional[float] = None

    def fit(self, X: np.ndarray) -> "ChemometricPCA":
        X = np.asarray(X, dtype=float)
        if self.center:
            self.mean_ = X.mean(axis=0)
            Xc = X - self.mean_
        else:
            self.mean_ = np.zeros(X.shape[1])
            Xc = X
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pca_ = SkPCA(n_components=n_comp)
        self.pca_.fit(Xc)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        # Control limits from training data (95% empirical)
        T, Q = self._tq(Xc)
        self.t2_limit_ = float(np.percentile(T, 95))
        self.q_limit_ = float(np.percentile(Q, 95))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=float) - self.mean_
        return self.pca_.transform(Xc)

    def _tq(self, Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T_scores = self.pca_.transform(Xc)
        X_hat = self.pca_.inverse_transform(T_scores)
        E = Xc - X_hat
        # Hotelling T² with diagonal approximation using explained variance
        var = self.pca_.explained_variance_
        var = np.where(var < 1e-12, 1e-12, var)
        T2 = np.sum((T_scores ** 2) / var, axis=1)
        Q = np.sum(E ** 2, axis=1)
        return T2, Q

    def hotelling_t2(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=float) - self.mean_
        T2, _ = self._tq(Xc)
        return T2

    def q_residuals(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=float) - self.mean_
        _, Q = self._tq(Xc)
        return Q

    def monitor(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return T², Q, and anomaly flags (exceeds 95% training limits)."""
        T2 = self.hotelling_t2(X)
        Q = self.q_residuals(X)
        anomaly = (T2 > self.t2_limit_) | (Q > self.q_limit_)
        return {
            "T2": T2,
            "Q": Q,
            "T2_limit": np.full_like(T2, self.t2_limit_),
            "Q_limit": np.full_like(Q, self.q_limit_),
            "anomaly": anomaly.astype(int),
        }

    def loadings(self) -> np.ndarray:
        return self.pca_.components_

    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio_)

    def contribution_q(self, X: np.ndarray, sample_idx: int = 0) -> np.ndarray:
        """Variable contributions to Q residual for one sample."""
        Xc = np.asarray(X, dtype=float) - self.mean_
        T_scores = self.pca_.transform(Xc)
        X_hat = self.pca_.inverse_transform(T_scores)
        E = Xc - X_hat
        return E[sample_idx] ** 2
