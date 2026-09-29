"""Multivariate process monitoring: T², Q, contributions, regime hints."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from src.chemometrics.pca import ChemometricPCA


class ProcessMonitor:
    """PCA-based multivariate statistical process control (MSPC)."""

    def __init__(self, n_components: int = 5):
        self.pca = ChemometricPCA(n_components=n_components)
        self.gmm_: Optional[GaussianMixture] = None
        self.fitted_ = False

    def fit(self, X_reference: np.ndarray) -> "ProcessMonitor":
        """Fit on normal operating condition (NOC) data."""
        self.pca.fit(X_reference)
        scores = self.pca.transform(X_reference)
        # Soft regime clustering on scores
        n_regimes = min(3, max(1, X_reference.shape[0] // 30))
        self.gmm_ = GaussianMixture(n_components=n_regimes, random_state=42)
        self.gmm_.fit(scores)
        self.fitted_ = True
        return self

    def monitor(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        out = self.pca.monitor(X)
        scores = self.pca.transform(X)
        if self.gmm_ is not None:
            out["regime"] = self.gmm_.predict(scores)
            out["regime_proba"] = self.gmm_.predict_proba(scores).max(axis=1)
        return out

    def root_cause(
        self, X: np.ndarray, sample_idx: int = 0, top_k: int = 10
    ) -> Dict[str, object]:
        """Statistical contribution analysis for an anomalous sample.

        Returns largest contributors to Q residual (not causal claims).
        """
        contrib = self.pca.contribution_q(X, sample_idx)
        order = np.argsort(contrib)[::-1]
        top = order[:top_k]
        status = self.monitor(X[sample_idx : sample_idx + 1])
        return {
            "sample_idx": sample_idx,
            "T2": float(status["T2"][0]),
            "Q": float(status["Q"][0]),
            "anomaly": bool(status["anomaly"][0]),
            "top_contributor_indices": top.tolist(),
            "top_contributor_values": contrib[top].tolist(),
            "message": (
                "Largest statistical contributors to the Q residual "
                "(not proven causal effects)."
            ),
        }

    def summary_status(self, X: np.ndarray) -> Dict[str, object]:
        m = self.monitor(X)
        n_anom = int(m["anomaly"].sum())
        return {
            "n_samples": len(m["anomaly"]),
            "n_anomalies": n_anom,
            "anomaly_rate": n_anom / max(1, len(m["anomaly"])),
            "mean_T2": float(m["T2"].mean()),
            "mean_Q": float(m["Q"].mean()),
            "T2_limit": float(m["T2_limit"][0]),
            "Q_limit": float(m["Q_limit"][0]),
        }
