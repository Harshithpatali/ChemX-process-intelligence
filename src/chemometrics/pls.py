"""Partial Least Squares regression and related chemometric models."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "bias": bias}


class PLSModel:
    """PLS regression with VIP scores and coefficient interpretation."""

    def __init__(self, n_components: int = 10, scale: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.model_: Optional[PLSRegression] = None
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.model_ = PLSRegression(n_components=n_comp, scale=self.scale)
        self.model_.fit(X, y)
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_ = np.where(self.x_std_ < 1e-12, 1.0, self.x_std_)
        self.y_mean_ = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X).ravel()

    def coefficients(self) -> np.ndarray:
        return self.model_.coef_.ravel()

    def vip_scores(self) -> np.ndarray:
        """Variable Importance in Projection (Wold)."""
        t = self.model_.x_scores_          # (n, h)
        w = self.model_.x_weights_         # (p, h)
        q = np.asarray(self.model_.y_loadings_).ravel()
        p, h = w.shape
        s = np.array([float((t[:, j] ** 2).sum() * (q[j] ** 2)) for j in range(h)])
        total_s = s.sum() + 1e-12
        vips = np.zeros(p)
        for i in range(p):
            weight = np.array([
                (w[i, j] / (np.linalg.norm(w[:, j]) + 1e-12)) ** 2 for j in range(h)
            ])
            vips[i] = np.sqrt(p * np.dot(s, weight) / total_s)
        return vips

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return regression_metrics(y, self.predict(X))


def compare_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compare PCR, PLS, Ridge, ElasticNet on the same split."""
    results = {}

    # PLS
    pls = PLSModel(n_components=n_components)
    pls.fit(X_train, y_train)
    results["PLS"] = pls.evaluate(X_test, y_test)

    # PCR = PCA + linear regression via Ridge with high alpha on scores? 
    # Simpler: use sklearn Pipeline PCA + Ridge(alpha=0) ~ OLS on scores
    from sklearn.linear_model import LinearRegression
    n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    pcr = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_comp)),
        ("lr", LinearRegression()),
    ])
    pcr.fit(X_train, y_train)
    results["PCR"] = regression_metrics(y_test, pcr.predict(X_test))

    # Ridge
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    ridge.fit(X_train, y_train)
    results["Ridge"] = regression_metrics(y_test, ridge.predict(X_test))

    # Elastic Net
    enet = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ])
    enet.fit(X_train, y_train)
    results["ElasticNet"] = regression_metrics(y_test, enet.predict(X_test))

    return results


def select_n_components_cv(
    X: np.ndarray, y: np.ndarray, max_comp: int = 20, cv: int = 5
) -> Tuple[int, List[float]]:
    """Choose n_components by minimizing RMSE of cross-validated predictions."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    max_comp = min(max_comp, X.shape[0] - 1, X.shape[1])
    rmses = []
    for n in range(1, max_comp + 1):
        pls = PLSRegression(n_components=n, scale=True)
        yhat = cross_val_predict(pls, X, y, cv=cv)
        rmses.append(float(np.sqrt(mean_squared_error(y, yhat))))
    best = int(np.argmin(rmses)) + 1
    return best, rmses
