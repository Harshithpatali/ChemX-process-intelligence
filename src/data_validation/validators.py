"""Core validation functions for ChemX datasets.

All functions return structured dicts suitable for logging and unit tests.
No modelling assumptions are made here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame]


def check_missing(X: ArrayLike, name: str = "X") -> Dict[str, Any]:
    """Count missing values (NaN / None)."""
    arr = np.asarray(X, dtype=float)
    n_missing = int(np.isnan(arr).sum())
    pct = 100.0 * n_missing / arr.size if arr.size else 0.0
    return {
        "name": name,
        "n_missing": n_missing,
        "pct_missing": pct,
        "has_missing": n_missing > 0,
        "shape": arr.shape,
    }


def check_duplicates(X: ArrayLike, name: str = "X", axis: int = 0) -> Dict[str, Any]:
    """Count exact duplicate rows (axis=0) or columns (axis=1)."""
    arr = np.asarray(X)
    if axis == 0:
        unique = np.unique(arr, axis=0)
        n_dup = arr.shape[0] - unique.shape[0]
    else:
        unique = np.unique(arr, axis=1)
        n_dup = arr.shape[1] - unique.shape[1]
    return {
        "name": name,
        "n_duplicates": int(n_dup),
        "has_duplicates": n_dup > 0,
        "axis": axis,
    }


def check_constant_features(X: ArrayLike, name: str = "X", tol: float = 1e-12) -> Dict[str, Any]:
    """Identify features (columns) with near-zero variance."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected 2-D array")
    stds = np.nanstd(arr, axis=0)
    constant_mask = stds <= tol
    n_const = int(constant_mask.sum())
    return {
        "name": name,
        "n_constant": n_const,
        "constant_indices": np.where(constant_mask)[0].tolist(),
        "has_constant": n_const > 0,
    }


def check_near_zero_variance(
    X: ArrayLike, name: str = "X", freq_cut: float = 95 / 5, unique_cut: float = 10
) -> Dict[str, Any]:
    """Near-zero variance heuristic (caret-style).

    A feature is near-zero variance if:
    - number of unique values / n_samples < unique_cut / 100, OR
    - most common value frequency ratio > freq_cut.
    Simplified for continuous spectral data: mainly unique-count based.
    """
    arr = np.asarray(X, dtype=float)
    n_samples, n_features = arr.shape
    nzv_idx = []
    for j in range(n_features):
        col = arr[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            nzv_idx.append(j)
            continue
        n_unique = len(np.unique(col))
        if n_unique <= 1:
            nzv_idx.append(j)
            continue
        # frequency ratio of most common
        vals, counts = np.unique(col, return_counts=True)
        ratio = counts.max() / counts.min() if counts.min() > 0 else np.inf
        if (n_unique / n_samples * 100 < unique_cut) or (ratio > freq_cut):
            nzv_idx.append(j)
    return {
        "name": name,
        "n_near_zero_var": len(nzv_idx),
        "near_zero_var_indices": nzv_idx[:50],  # cap for readability
        "has_near_zero_var": len(nzv_idx) > 0,
    }


def check_value_range(
    X: ArrayLike,
    name: str = "X",
    min_allowed: Optional[float] = None,
    max_allowed: Optional[float] = None,
) -> Dict[str, Any]:
    """Report min/max and flag values outside optional bounds."""
    arr = np.asarray(X, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"name": name, "min": None, "max": None, "n_out_of_range": 0}
    mn, mx = float(finite.min()), float(finite.max())
    n_oor = 0
    if min_allowed is not None:
        n_oor += int((finite < min_allowed).sum())
    if max_allowed is not None:
        n_oor += int((finite > max_allowed).sum())
    return {
        "name": name,
        "min": mn,
        "max": mx,
        "n_out_of_range": n_oor,
        "has_out_of_range": n_oor > 0,
    }


def check_wavelength_order(
    axis: ArrayLike, name: str = "wavelength_axis", expect_increasing: bool = True
) -> Dict[str, Any]:
    """Verify wavelength / wavenumber axis is monotonic."""
    ax = np.asarray(axis).ravel()
    diffs = np.diff(ax)
    if expect_increasing:
        ok = np.all(diffs > 0)
        direction = "increasing"
    else:
        ok = np.all(diffs < 0)
        direction = "decreasing"
    return {
        "name": name,
        "is_monotonic": bool(ok),
        "expected_direction": direction,
        "n_points": len(ax),
        "range": (float(ax.min()), float(ax.max())) if len(ax) else None,
    }


def check_shape_consistency(
    X: ArrayLike, expected_shape: Tuple[int, ...], name: str = "X"
) -> Dict[str, Any]:
    """Assert array shape matches expected dimensions."""
    arr = np.asarray(X)
    match = arr.shape == expected_shape
    return {
        "name": name,
        "actual_shape": arr.shape,
        "expected_shape": expected_shape,
        "matches": match,
    }


def summarize_dataset(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    name: str = "dataset",
) -> Dict[str, Any]:
    """High-level summary combining several checks."""
    arr = np.asarray(X, dtype=float)
    summary: Dict[str, Any] = {
        "name": name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "missing": check_missing(arr, name),
        "duplicates": check_duplicates(arr, name),
        "range": check_value_range(arr, name),
    }
    if arr.ndim == 2:
        summary["constant_features"] = check_constant_features(arr, name)
    if y is not None:
        yarr = np.asarray(y, dtype=float)
        summary["y_shape"] = yarr.shape
        summary["y_missing"] = check_missing(yarr, "y")
        summary["y_range"] = check_value_range(yarr, "y")
    return summary


# ---------------------------------------------------------------------------
# Dataset-specific audit entry points
# ---------------------------------------------------------------------------

def audit_mlnir(raw_dir: Union[str, Path]) -> Dict[str, Any]:
    """Full audit of MLNIRdata CSVs."""
    raw_dir = Path(raw_dir)
    X = pd.read_csv(raw_dir / "MLNIR_matrixX_NirSpectrumData.csv", header=None).values
    # CSV is wavelengths × samples → transpose to n × p
    X = X.T
    axis = pd.read_csv(raw_dir / "MLNIR_matrixX_NirSpectrumDataAxis.csv", header=None).values.ravel()
    y = pd.read_csv(raw_dir / "MLNIR_matrixY_NirPropertyDensityNormalized.csv", header=None).values.ravel()

    report = {
        "dataset": "MLNIRdata",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "X_summary": summarize_dataset(X, y, "MLNIR_spectra"),
        "wavelength_order": check_wavelength_order(axis, expect_increasing=True),  # values increase; plots often reverse axis
        "axis_range_cm-1": (float(axis.min()), float(axis.max())),
        "y_name": "normalized_density",
        "y_range": (float(y.min()), float(y.max())),
        "notes": "Density is released normalized to [0,1]. Spectra in arbitrary absorbance units.",
    }
    return report


def audit_corn(mat_path: Union[str, Path]) -> Dict[str, Any]:
    """Full audit of Eigenvector corn.mat."""
    import scipy.io as sio

    mat_path = Path(mat_path)
    d = sio.loadmat(mat_path)
    reports = {"dataset": "Corn_multi_instrument"}
    for inst in ["m5spec", "mp5spec", "mp6spec"]:
        data = d[inst]["data"][0, 0]
        reports[inst] = summarize_dataset(data, name=inst)
    pv = d["propvals"]["data"][0, 0]
    reports["properties"] = {
        "names": ["Moisture", "Oil", "Protein", "Starch"],
        "shape": pv.shape,
        "missing": check_missing(pv, "propvals"),
        "means": pv.mean(0).tolist(),
        "ranges": list(zip(pv.min(0).tolist(), pv.max(0).tolist())),
    }
    # wavelength axis from m5
    ax = d["m5spec"]["axisscale"][0, 0][1, 0].ravel()
    reports["wavelength_nm"] = {
        "n": len(ax),
        "range": (int(ax.min()), int(ax.max())),
        "order": check_wavelength_order(ax, expect_increasing=True),
    }
    return reports


def audit_sugar(mat_path: Union[str, Path]) -> Dict[str, Any]:
    """Full audit of UCPH sugar process data.mat."""
    import scipy.io as sio

    mat_path = Path(mat_path)
    d = sio.loadmat(mat_path)
    X = d["X"]  # 268 x 3997 (unfolded EEM)
    y = d["y"]
    Proc = d["Proc"]
    Lab = d["Lab"]
    EmAx = d["EmAx"].ravel()
    ExAx = d["ExAx"].ravel()
    DimX = d["DimX"].ravel()

    report = {
        "dataset": "Sugar_process",
        "DimX": DimX.tolist(),
        "n_samples": int(X.shape[0]),
        "X_unfolded_shape": X.shape,
        "X_summary": summarize_dataset(X, name="fluorescence_unfolded"),
        "y": {
            "columns": [str(s).strip() for s in d["Yidx"]],
            "shape": y.shape,
            "missing": check_missing(y, "y"),
            "ranges": list(zip(np.nanmin(y, 0).tolist(), np.nanmax(y, 0).tolist())),
        },
        "Proc": {
            "shape": Proc.shape,
            "missing": check_missing(Proc, "Proc"),
            "expected_3way": d["DimProc"].ravel().tolist(),
        },
        "Lab": {
            "shape": Lab.shape,
            "missing": check_missing(Lab, "Lab"),
            "expected_3way": d["DimLab"].ravel().tolist(),
        },
        "EmAx_nm": {"n": len(EmAx), "range": (float(EmAx.min()), float(EmAx.max()))},
        "ExAx_nm": ExAx.tolist(),
        "notes": (
            "X is unfolded (samples x (emission*excitation)). "
            "Proc and Lab contain many NaNs (documented lag structure). "
            "X max ~1000 may indicate saturation or sentinel; inspect before modelling."
        ),
    }
    return report
