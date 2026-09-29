"""Unit tests for ChemX data validation suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_validation.validators import (
    check_missing,
    check_duplicates,
    check_constant_features,
    check_value_range,
    check_wavelength_order,
    check_shape_consistency,
    summarize_dataset,
    audit_mlnir,
    audit_corn,
    audit_sugar,
)

RAW = ROOT / "data" / "raw"


def test_check_missing_none():
    X = np.random.randn(10, 5)
    r = check_missing(X)
    assert r["n_missing"] == 0
    assert not r["has_missing"]


def test_check_missing_some():
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    r = check_missing(X)
    assert r["n_missing"] == 1
    assert r["has_missing"]


def test_check_duplicates():
    X = np.array([[1, 2], [1, 2], [3, 4]])
    r = check_duplicates(X)
    assert r["n_duplicates"] == 1
    assert r["has_duplicates"]


def test_check_constant_features():
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    r = check_constant_features(X)
    assert r["n_constant"] == 1
    assert 0 in r["constant_indices"]


def test_check_value_range():
    X = np.array([0.1, 0.5, 1.2])
    r = check_value_range(X, min_allowed=0.0, max_allowed=1.0)
    assert r["n_out_of_range"] == 1
    assert r["has_out_of_range"]


def test_check_wavelength_order_increasing():
    ax = np.linspace(1100, 2500, 100)
    r = check_wavelength_order(ax, expect_increasing=True)
    assert r["is_monotonic"]


def test_check_wavelength_order_decreasing():
    ax = np.linspace(9000, 4000, 50)
    r = check_wavelength_order(ax, expect_increasing=False)
    assert r["is_monotonic"]


def test_check_shape_consistency():
    X = np.zeros((80, 700))
    r = check_shape_consistency(X, (80, 700))
    assert r["matches"]
    r2 = check_shape_consistency(X, (80, 701))
    assert not r2["matches"]


def test_summarize_dataset():
    X = np.random.randn(20, 10)
    y = np.random.randn(20)
    s = summarize_dataset(X, y, name="toy")
    assert s["shape"] == (20, 10)
    assert "missing" in s
    assert "y_shape" in s


@pytest.mark.skipif(not (RAW / "mlnirdata" / "MLNIR_matrixX_NirSpectrumData.csv").exists(), reason="MLNIR data missing")
def test_audit_mlnir():
    report = audit_mlnir(RAW / "mlnirdata")
    assert report["n_samples"] == 208
    assert report["n_features"] == 2635
    assert report["X_summary"]["missing"]["n_missing"] == 0
    assert report["X_summary"]["duplicates"]["n_duplicates"] == 0
    assert report["wavelength_order"]["is_monotonic"]


@pytest.mark.skipif(not (RAW / "corn" / "corn.mat").exists(), reason="Corn data missing")
def test_audit_corn():
    report = audit_corn(RAW / "corn" / "corn.mat")
    assert report["m5spec"]["shape"] == (80, 700)
    assert report["properties"]["missing"]["n_missing"] == 0
    assert report["wavelength_nm"]["n"] == 700


@pytest.mark.skipif(not (RAW / "sugar" / "process" / "data.mat").exists(), reason="Sugar data missing")
def test_audit_sugar():
    report = audit_sugar(RAW / "sugar" / "process" / "data.mat")
    assert report["n_samples"] == 268
    assert report["X_unfolded_shape"] == (268, 3997)
    assert report["X_summary"]["missing"]["n_missing"] == 0
    assert report["Proc"]["missing"]["has_missing"]  # expected NaNs
    assert report["y"]["columns"][1].strip() == "Color"
