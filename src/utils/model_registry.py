"""Persist and load soft-sensor / monitoring artifacts for industry deployment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from src.config import get_settings
from src.utils.io import load_mlnir, train_test_split_indices
from src.chemometrics.pls import PLSModel
from src.chemometrics.pca import ChemometricPCA
from src.bayesian.soft_sensor import BayesianSoftSensor
from src.monitoring.process_monitor import ProcessMonitor

logger = logging.getLogger("chemx.models")


def train_and_save(force: bool = False) -> Dict[str, Any]:
    """Train MLNIR soft-sensor stack and write artifacts under models/."""
    settings = get_settings()
    meta_path = settings.model_dir / "mlnir_meta.json"
    if meta_path.exists() and not force:
        logger.info("Models already present; use force=True to retrain")
        with open(meta_path) as f:
            return json.load(f)

    data = load_mlnir()
    X, y = data["X"], data["y"]
    tr, te = train_test_split_indices(len(y), settings.test_size, settings.random_seed)
    mu = X[tr].mean(0)
    Xtr = X[tr] - mu
    Xte = X[te] - mu

    pls = PLSModel(n_components=settings.pls_components).fit(Xtr, y[tr])
    bayes = BayesianSoftSensor(n_features_pca=30).fit(Xtr, y[tr])
    pca = ChemometricPCA(n_components=settings.pca_components).fit(Xtr)
    mon = ProcessMonitor(n_components=settings.pca_components).fit(Xtr)

    artifacts = {
        "mu": mu,
        "pls": pls,
        "bayes": bayes,
        "pca": pca,
        "monitor": mon,
        "n_features": int(X.shape[1]),
        "axis": data["axis"],
    }
    joblib.dump(artifacts, settings.model_dir / "mlnir_bundle.joblib")

    metrics = {
        "pls_test_rmse": float(
            np.sqrt(np.mean((pls.predict(Xte) - y[te]) ** 2))
        ),
        "bayesian_coverage_95": float(bayes.coverage(Xte, y[te])),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features": int(X.shape[1]),
        "source": data["source"],
        "target": "normalized_density",
    }
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved model bundle and metrics: %s", metrics)
    return metrics


def load_bundle() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model bundle + metrics; train if missing."""
    settings = get_settings()
    bundle_path = settings.model_dir / "mlnir_bundle.joblib"
    meta_path = settings.model_dir / "mlnir_meta.json"
    if not bundle_path.exists():
        metrics = train_and_save(force=True)
    else:
        with open(meta_path) as f:
            metrics = json.load(f)
    bundle = joblib.load(bundle_path)
    return bundle, metrics
