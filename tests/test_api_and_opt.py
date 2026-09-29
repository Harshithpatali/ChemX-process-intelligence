"""API smoke tests and optimization constraint tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization.process_opt import optimize_deterministic, optimize_risk_aware
from src.physics_informed.reactor import cstr_steady_state, simulate_campaign
from src.utils.io import load_mlnir, train_test_split_indices
from src.chemometrics.pls import PLSModel


def test_cstr_physical_bounds():
    r = cstr_steady_state(1.0, 350.0, 2.0)
    assert 0 <= r["conversion"] <= 1
    assert r["C_A"] >= 0
    assert r["yield"] >= 0


def test_deterministic_opt_feasible():
    res = optimize_deterministic()
    assert res["success"] or res["yield"] > 0
    assert 320 <= res["T"] <= 420
    assert 0.5 <= res["tau"] <= 6.0
    assert 0 <= res["yield"] <= 1


def test_simulation_label():
    sim = simulate_campaign(50, seed=1)
    assert sim["label"] == "MECHANISTIC_SIMULATION"
    assert sim["spectra"].shape[0] == 50


def test_pls_predict_shape():
    data = load_mlnir()
    X, y = data["X"], data["y"]
    tr, te = train_test_split_indices(len(y), 0.3, 0)
    mu = X[tr].mean(0)
    pls = PLSModel(n_components=5).fit(X[tr] - mu, y[tr])
    pred = pls.predict(X[te] - mu)
    assert pred.shape == (len(te),)


@pytest.mark.skipif(True, reason="Optional FastAPI TestClient; run manually if httpx available")
def test_api_health():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
