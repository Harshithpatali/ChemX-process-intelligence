"""Runtime-independent API, model and optimization smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chemometrics.pls import PLSModel
from src.optimization.process_opt import optimize_deterministic
from src.physics_informed.reactor import cstr_steady_state, simulate_campaign
from src.utils.model_registry import load_bundle


def test_cstr_physical_bounds():
    result = cstr_steady_state(1.0, 350.0, 2.0)
    assert 0 <= result["conversion"] <= 1
    assert result["C_A"] >= 0
    assert result["yield"] >= 0


def test_deterministic_opt_feasible():
    result = optimize_deterministic()
    assert result["success"]
    assert 320 <= result["T"] <= 420
    assert 0.5 <= result["tau"] <= 6.0
    assert 0 <= result["yield"] <= 1


def test_simulation_label():
    sim = simulate_campaign(50, seed=1)
    assert sim["label"] == "MECHANISTIC_SIMULATION"
    assert sim["spectra"].shape == (50, 50)


def test_pls_predict_shape_on_synthetic_matrix():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 20))
    y = X[:, 0] * 0.4 - X[:, 1] * 0.2 + rng.normal(0, 0.05, 40)
    model = PLSModel(n_components=3).fit(X[:30], y[:30])
    pred = model.predict(X[30:])
    assert pred.shape == (10,)


def test_production_model_bundle_loads_without_raw_data():
    bundle, metrics = load_bundle()
    assert "pls" in bundle
    assert "bayes" in bundle
    assert "monitor" in bundle
    assert metrics["n_features"] == bundle["n_features"]


def test_api_health_and_model_info():
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["model_loaded"] is True

        info = client.get("/model-info")
        assert info.status_code == 200
        assert info.json()["runtime_data_required"] is False


def test_api_predict_and_monitor():
    from fastapi.testclient import TestClient
    from src.api.main import app

    bundle, _ = load_bundle()
    spectrum = np.asarray(bundle["mu"], dtype=float).tolist()

    with TestClient(app) as client:
        prediction = client.post("/predict", json={"spectrum": spectrum})
        assert prediction.status_code == 200
        assert "prediction" in prediction.json()
        assert "uncertainty" in prediction.json()

        monitoring = client.post("/monitor", json={"spectrum": spectrum})
        assert monitoring.status_code == 200
        assert "T2" in monitoring.json()
        assert "Q" in monitoring.json()
