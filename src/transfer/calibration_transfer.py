"""Calibration transfer experiments across NIR instruments (Corn dataset)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from src.preprocessing.spectral import SNV, MSC


def direct_standardization(
    X_primary: np.ndarray, X_secondary: np.ndarray
) -> np.ndarray:
    """Simple Direct Standardization: transfer matrix F such that X_p ≈ X_s @ F.

    Uses least-squares on paired samples (same physical samples).
    """
    # X_primary ≈ X_secondary @ F  →  F = pinv(X_s) @ X_p
    F, _, _, _ = np.linalg.lstsq(X_secondary, X_primary, rcond=None)
    return F


def apply_transfer(X_secondary: np.ndarray, F: np.ndarray) -> np.ndarray:
    return X_secondary @ F


def transfer_experiment(
    X_a: np.ndarray,
    X_b: np.ndarray,
    y: np.ndarray,
    n_components: int = 8,
    property_idx: int = 0,
) -> Dict[str, float]:
    """Train PLS on instrument A, evaluate on B under several transfer strategies.

    y can be multi-column; property_idx selects the target.
    """
    y = np.asarray(y)
    if y.ndim > 1:
        y = y[:, property_idx]
    y = y.ravel()

    if X_a.shape[0] != X_b.shape[0] or X_a.shape[0] != y.shape[0]:
        raise ValueError("X_a, X_b and y must contain the same number of paired samples.")
    if X_a.shape[0] < 4:
        raise ValueError("At least four paired samples are required.")

    # Keep the historical 60-sample split when the public dataset is large,
    # but remain safe for smaller unit-test or future datasets.
    n_train = min(60, max(2, X_a.shape[0] // 2))
    Xa_tr, Xa_te = X_a[:n_train], X_a[n_train:]
    Xb_tr, Xb_te = X_b[:n_train], X_b[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    results = {}

    def eval_pls(Xtr, ytr, Xte, yte):
        pls = PLSRegression(n_components=min(n_components, Xtr.shape[0] - 1))
        pls.fit(Xtr, ytr)
        pred = pls.predict(Xte).ravel()
        return float(np.sqrt(mean_squared_error(yte, pred)))

    # 1. No transfer: train A, test B raw
    results["no_transfer_RMSE"] = eval_pls(Xa_tr, y_tr, Xb_te, y_te)

    # 2. Same instrument baseline (train A, test A)
    results["same_instrument_RMSE"] = eval_pls(Xa_tr, y_tr, Xa_te, y_te)

    # 3. SNV on both
    snv = SNV()
    results["snv_transfer_RMSE"] = eval_pls(
        snv.fit_transform(Xa_tr), y_tr, snv.transform(Xb_te), y_te
    )

    # 4. MSC on both
    msc = MSC()
    results["msc_transfer_RMSE"] = eval_pls(
        msc.fit_transform(Xa_tr), y_tr, msc.transform(Xb_te), y_te
    )

    # 5. Direct standardization
    F = direct_standardization(Xa_tr, Xb_tr)
    Xb_te_ds = apply_transfer(Xb_te, F)
    results["direct_std_RMSE"] = eval_pls(Xa_tr, y_tr, Xb_te_ds, y_te)

    return results
