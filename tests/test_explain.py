"""Tests for config and stakeholder explainer (offline / fallback)."""

from __future__ import annotations

from src.config import get_settings
from src.explain.stakeholder import StakeholderExplainer


def test_settings_load():
    s = get_settings()
    assert s.pls_components >= 1
    assert s.model_dir.exists()


def test_explainer_fallback_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Clear cached settings
    get_settings.cache_clear()
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    exp = StakeholderExplainer(api_key="")
    assert not exp.available
    out = exp.explain({"type": "prediction", "prediction": 0.5, "status": "ok"})
    assert out["ok"] is False
    assert "fallback" in out
    assert "0.5" in out["fallback"]


def test_explainer_payload_structure():
    exp = StakeholderExplainer(api_key="")
    fb = exp._fallback(
        {
            "type": "monitoring",
            "anomaly": True,
            "T2": 12.0,
            "top_contributor_indices": [1, 2, 3],
        }
    )
    assert "anomaly" in fb.lower() or "Anomaly" in fb
