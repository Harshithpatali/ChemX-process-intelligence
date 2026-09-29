"""Versioned model-bundle loading for deployment."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Tuple

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
    settings = get_settings()
    meta_path = settings.model_dir / "mlnir_meta.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text())

    data = load_mlnir()
    X, y = data["X"], data["y"]
    tr, te = train_test_split_indices(len(y), settings.test_size, settings.random_seed)
    mu = X[tr].mean(0)
    Xtr, Xte = X[tr] - mu, X[te] - mu

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
        "model_version": "mlnir-density-v1",
        "pls_test_rmse": float(np.sqrt(np.mean((pls.predict(Xte) - y[te]) ** 2))),
        "bayesian_coverage_95": float(bayes.coverage(Xte, y[te])),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features": int(X.shape[1]),
        "source": data["source"],
        "target": "normalized_density",
    }
    meta_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_bundle() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    settings = get_settings()
    bundle_path = settings.model_dir / "mlnir_bundle.joblib"
    meta_path = settings.model_dir / "mlnir_meta.json"

    if not bundle_path.exists() or not meta_path.exists():
        if settings.is_production or not settings.allow_runtime_training:
            raise RuntimeError(
                "ChemX model bundle is missing. Build/train the model before deployment "
                "or enable CHEMX_ALLOW_RUNTIME_TRAINING outside production."
            )
        train_and_save(force=True)

    bundle = joblib.load(bundle_path)
    metrics = json.loads(meta_path.read_text())
    return bundle, metrics
