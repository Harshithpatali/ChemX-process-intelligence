"""ChemX FastAPI — industry-oriented service layer.

Endpoints:
  GET  /health
  GET  /ready
  GET  /model-info
  GET  /metrics
  POST /predict
  POST /monitor
  POST /anomaly
  POST /optimize
  POST /explain   (Groq stakeholder narrative; requires GROQ_API_KEY)
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.utils.logging_setup import setup_logging
from src.utils.model_registry import load_bundle, train_and_save
from src.optimization.process_opt import optimize_deterministic, optimize_risk_aware
from src.explain.stakeholder import StakeholderExplainer

settings = get_settings()
logger = setup_logging(settings.log_level)

app = FastAPI(
    title="ChemX API",
    description=(
        "Physics-Informed Chemometric Process Intelligence Platform. "
        "Public research datasets only — not industrial plant data."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if settings.cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle: Dict[str, Any] = {}
_metrics: Dict[str, Any] = {}
_explainer = StakeholderExplainer()


@app.on_event("startup")
def _startup() -> None:
    global _bundle, _metrics
    logger.info("ChemX starting (env=%s)", settings.env)
    _bundle, _metrics = load_bundle()
    logger.info(
        "Models ready | n_features=%s | PLS RMSE=%.4f | Groq=%s",
        _metrics.get("n_features"),
        _metrics.get("pls_test_rmse", -1),
        "configured" if _explainer.available else "not configured",
    )


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


class SpectrumRequest(BaseModel):
    spectrum: List[float] = Field(..., description="NIR spectrum; length must match model")
    explain: bool = Field(False, description="If true, attach Groq stakeholder explanation")
    audience: str = Field("operations", description="operations | quality | management")


class ProcessRequest(BaseModel):
    T: float = 360.0
    tau: float = 2.0
    C_A0: float = 1.0
    risk_aware: bool = False
    min_prob: float = 0.85
    quality_threshold: float = 0.50
    explain: bool = False
    audience: str = "management"


class ExplainRequest(BaseModel):
    context: Dict[str, Any] = Field(..., description="Any ChemX result JSON to narrate")
    audience: str = "operations"
    detail: str = Field("standard", description="brief | standard | detailed")


def _check_spectrum(spectrum: List[float]) -> np.ndarray:
    n = int(_metrics.get("n_features") or _bundle["n_features"])
    x = np.asarray(spectrum, dtype=float)
    if x.ndim != 1 or x.shape[0] != n:
        raise HTTPException(
            status_code=400,
            detail=f"Expected spectrum length {n}, got {x.shape}",
        )
    return (x.reshape(1, -1) - _bundle["mu"]).astype(float)


def _maybe_explain(payload: Dict[str, Any], explain: bool, audience: str) -> Dict[str, Any]:
    if not explain:
        return payload
    exp = _explainer.explain(payload, audience=audience)
    out = dict(payload)
    out["stakeholder_explanation"] = exp
    return out


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ChemX",
        "version": "0.2.0",
        "env": settings.env,
    }


@app.get("/ready")
def ready():
    ready_ok = bool(_bundle) and "pls" in _bundle
    return {
        "ready": ready_ok,
        "models_loaded": ready_ok,
        "groq_configured": _explainer.available,
        "n_features": _metrics.get("n_features"),
    }


@app.get("/model-info")
def model_info():
    return {
        "soft_sensor": "PLS + BayesianRidge on MLNIRdata density",
        "n_features": _metrics.get("n_features"),
        "n_train": _metrics.get("n_train"),
        "preprocessing": "mean-centering",
        "data_source": _metrics.get("source", "MLNIRdata (public)"),
        "disclaimer": "Public research data only — not industrial plant data.",
        "groq_explanations": _explainer.available,
    }


@app.get("/metrics")
def metrics():
    return _metrics


@app.post("/predict")
def predict(req: SpectrumRequest):
    x = _check_spectrum(req.spectrum)
    yhat = float(_bundle["pls"].predict(x)[0])
    interval = _bundle["bayes"].predict_interval(x)
    payload = {
        "type": "prediction",
        "prediction": yhat,
        "uncertainty": {
            "lower": float(interval["lower"][0]),
            "upper": float(interval["upper"][0]),
            "std": float(interval["std"][0]),
        },
        "status": "ok",
        "target": "normalized_density",
        "units_note": "Normalized density on public MLNIRdata scale [0,1]",
    }
    return _maybe_explain(payload, req.explain, req.audience)


@app.post("/monitor")
def monitor(req: SpectrumRequest):
    x = _check_spectrum(req.spectrum)
    m = _bundle["monitor"].monitor(x)
    anomaly = bool(m["anomaly"][0])
    payload = {
        "type": "monitoring",
        "T2": float(m["T2"][0]),
        "Q": float(m["Q"][0]),
        "T2_limit": float(m["T2_limit"][0]),
        "Q_limit": float(m["Q_limit"][0]),
        "anomaly": anomaly,
        "status": "anomaly" if anomaly else "normal",
    }
    return _maybe_explain(payload, req.explain, req.audience)


@app.post("/anomaly")
def anomaly(req: SpectrumRequest):
    x = _check_spectrum(req.spectrum)
    rc = _bundle["monitor"].root_cause(x, sample_idx=0, top_k=10)
    rc["type"] = "root_cause"
    return _maybe_explain(rc, req.explain, req.audience)


@app.post("/optimize")
def optimize(req: ProcessRequest):
    if req.risk_aware:
        result = optimize_risk_aware(
            min_prob=req.min_prob,
            quality_threshold=req.quality_threshold,
        )
    else:
        result = optimize_deterministic()
    result["type"] = "optimization"
    result["note"] = "MECHANISTIC_SIMULATION — not industrial data"
    return _maybe_explain(result, req.explain, req.audience)


@app.post("/explain")
def explain(req: ExplainRequest):
    result = _explainer.explain(req.context, audience=req.audience, detail=req.detail)
    if not result.get("ok") and result.get("error") == "groq_not_configured":
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/admin/retrain")
def retrain():
    if settings.is_production:
        raise HTTPException(403, "Retrain disabled in production")
    global _bundle, _metrics
    train_and_save(force=True)
    _bundle, _metrics = load_bundle()
    return {"status": "retrained", "metrics": _metrics}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
