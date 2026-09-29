"""Load public ChemX research datasets.

Raw datasets are optional at runtime. The deployed API uses the versioned model
bundle under models/ and does not require data/raw/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def load_mlnir(transpose: bool = True) -> Dict[str, np.ndarray]:
    base = RAW / "mlnirdata"
    required = [
        base / "MLNIR_matrixX_NirSpectrumData.csv",
        base / "MLNIR_matrixX_NirSpectrumDataAxis.csv",
        base / "MLNIR_matrixY_NirPropertyDensityNormalized.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "MLNIRdata is not installed locally. Download the public dataset first; "
            "the deployed API does not require raw data. Missing: " + ", ".join(missing)
        )
    X = pd.read_csv(required[0], header=None).values
    if transpose:
        X = X.T
    axis = pd.read_csv(required[1], header=None).values.ravel()
    y = pd.read_csv(required[2], header=None).values.ravel()
    return {
        "X": X.astype(float),
        "y": y.astype(float),
        "axis": axis.astype(float),
        "axis_unit": "cm-1",
        "y_name": "normalized_density",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "source": "MLNIRdata (Zenodo 10.5281/zenodo.16783068; dataset DOI 10.5281/zenodo.16781222)",
    }


def load_corn(instrument: str = "m5") -> Dict[str, np.ndarray]:
    d = sio.loadmat(RAW / "corn" / "corn.mat")
    key = f"{instrument}spec"
    X = d[key]["data"][0, 0].astype(float)
    axis = d[key]["axisscale"][0, 0][1, 0].ravel().astype(float)
    props = d["propvals"]["data"][0, 0].astype(float)
    return {
        "X": X,
        "y": props,
        "y_names": ["Moisture", "Oil", "Protein", "Starch"],
        "axis": axis,
        "axis_unit": "nm",
        "instrument": instrument,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "source": "Eigenvector Corn NIR",
    }


def load_corn_all() -> Dict[str, Dict]:
    return {inst: load_corn(inst) for inst in ["m5", "mp5", "mp6"]}


def load_sugar() -> Dict[str, np.ndarray]:
    d = sio.loadmat(RAW / "sugar" / "process" / "data.mat")
    return {
        "X": d["X"].astype(float),
        "y": d["y"].astype(float),
        "y_names": [str(s).strip() for s in d["Yidx"]],
        "EmAx": d["EmAx"].ravel().astype(float),
        "ExAx": d["ExAx"].ravel().astype(float),
        "time": d["time"].ravel().astype(float),
        "Proc": d["Proc"].astype(float),
        "Lab": d["Lab"].astype(float),
        "DimX": d["DimX"].ravel(),
        "n_samples": d["X"].shape[0],
        "n_features": d["X"].shape[1],
        "source": "UCPH Sugar Process (Bro 1999)",
    }


def train_test_split_indices(n: int, test_size: float = 0.25, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    return idx[n_test:], idx[:n_test]
