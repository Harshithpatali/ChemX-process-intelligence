"""Data loading utilities for ChemX real datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def load_mlnir(transpose: bool = True) -> Dict[str, np.ndarray]:
    """Load MLNIRdata: hydrocarbon NIR spectra + normalized density.

    Returns dict with keys: X (n x p), y (n,), axis (p,), n_samples, n_features.
    """
    base = RAW / "mlnirdata"
    X = pd.read_csv(base / "MLNIR_matrixX_NirSpectrumData.csv", header=None).values
    if transpose:
        X = X.T  # n_samples x n_wavelengths
    axis = pd.read_csv(base / "MLNIR_matrixX_NirSpectrumDataAxis.csv", header=None).values.ravel()
    y = pd.read_csv(base / "MLNIR_matrixY_NirPropertyDensityNormalized.csv", header=None).values.ravel()
    return {
        "X": X.astype(float),
        "y": y.astype(float),
        "axis": axis.astype(float),
        "axis_unit": "cm-1",
        "y_name": "normalized_density",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "source": "MLNIRdata (Zenodo 10.5281/zenodo.16781223)",
    }


def load_corn(instrument: str = "m5") -> Dict[str, np.ndarray]:
    """Load Eigenvector corn multi-instrument NIR data.

    instrument: 'm5' | 'mp5' | 'mp6'
    """
    key = f"{instrument}spec"
    d = sio.loadmat(RAW / "corn" / "corn.mat")
    X = d[key]["data"][0, 0].astype(float)
    axis = d[key]["axisscale"][0, 0][1, 0].ravel().astype(float)
    props = d["propvals"]["data"][0, 0].astype(float)
    names = ["Moisture", "Oil", "Protein", "Starch"]
    return {
        "X": X,
        "y": props,
        "y_names": names,
        "axis": axis,
        "axis_unit": "nm",
        "instrument": instrument,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "source": "Eigenvector Corn NIR (Cargill / Eigenvector)",
    }


def load_corn_all() -> Dict[str, Dict]:
    """Load all three corn instruments."""
    return {inst: load_corn(inst) for inst in ["m5", "mp5", "mp6"]}


def load_sugar() -> Dict[str, np.ndarray]:
    """Load UCPH sugar process fluorescence + quality data."""
    d = sio.loadmat(RAW / "sugar" / "process" / "data.mat")
    X = d["X"].astype(float)  # 268 x 3997 unfolded
    y = d["y"].astype(float)
    EmAx = d["EmAx"].ravel().astype(float)
    ExAx = d["ExAx"].ravel().astype(float)
    time = d["time"].ravel().astype(float)
    Proc = d["Proc"].astype(float)
    Lab = d["Lab"].astype(float)
    return {
        "X": X,
        "y": y,
        "y_names": [str(s).strip() for s in d["Yidx"]],
        "EmAx": EmAx,
        "ExAx": ExAx,
        "time": time,
        "Proc": Proc,
        "Lab": Lab,
        "DimX": d["DimX"].ravel(),
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "source": "UCPH Sugar Process (Bro 1999)",
    }


def train_test_split_indices(
    n: int, test_size: float = 0.25, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple reproducible index split (no stratification)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    return idx[n_test:], idx[:n_test]
