"""ChemX production API.

The API is the runtime boundary: raw research datasets are not required by the
deployed service. The versioned model bundle under models/ is loaded at startup.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

from src.config import get_settings
from src.explain.stakeholder import StakeholderExplainer
from src.optimization.process_opt import optimize_deterministic, optimize_risk_aware
from src.utils.logging_setup import setup_logging
from src.utils.model_registry import load_bundle

settings = get_settings()
logger = setup_logging(settings.log_level)

APP_VERSION = "1.1.0"

@asynccontextmanager
async def lifespan(_: FastAPI):
    global _bundle, _metrics
    _bundle, _metrics = load_bundle()
    logger.info(
        "ChemX ready model=%s features=%s groq=%s",
        _metrics.get("model_version"),
        _metrics.get("n_features"),
        settings.groq_configured,
    )
    yield


app = FastAPI(
    title="ChemX API",
    version=APP_VERSION,
    description=(
        "Research-grade chemometric process intelligence API. "
        "Public experimental datasets only; mechanistic optimization is synthetic."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
)

_bundle: Dict[str, Any] = {}
_metrics: Dict[str, Any] = {}
_explainer = StakeholderExplainer()


class SpectrumRequest(BaseModel):
    spectrum: List[float] = Field(
        ..., min_length=1, max_length=10000,
        description="NIR spectrum in model wavelength order",
    )
    explain: bool = False
    audience: str = Field("operations", pattern="^(operations|quality|management)$")


class ProcessRequest(BaseModel):
    risk_aware: bool = False
    min_prob: float = Field(0.85, ge=0.0, le=1.0)
    quality_threshold: float = Field(0.50, ge=0.0, le=1.0)
    explain: bool = False
    audience: str = Field("management", pattern="^(operations|quality|management)$")


class ExplainRequest(BaseModel):
    context: Dict[str, Any]
    audience: str = Field("operations", pattern="^(operations|quality|management)$")
    detail: str = Field("standard", pattern="^(brief|standard|detailed)$")


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request error id=%s path=%s", request_id, request.url.path)
        raise
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
    return response


def _validate_spectrum(values: List[float]) -> np.ndarray:
    expected = int(_metrics.get("n_features") or _bundle.get("n_features", 0))
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size != expected:
        raise HTTPException(422, f"Expected {expected} spectral values; received {x.size}.")
    if not np.isfinite(x).all():
        raise HTTPException(422, "Spectrum contains NaN or infinite values.")
    return x.reshape(1, -1)


def _with_explanation(payload: Dict[str, Any], requested: bool, audience: str) -> Dict[str, Any]:
    if not requested:
        return payload
    result = _explainer.explain(payload, audience=audience)
    return {**payload, "stakeholder_explanation": result}


@app.get("/")
def root():
    return {"service": "ChemX API", "version": APP_VERSION, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "chemx-api", "version": APP_VERSION}


@app.get("/ready")
def ready():
    ok = bool(_bundle) and "pls" in _bundle and "bayes" in _bundle and "monitor" in _bundle
    return {
        "ready": ok,
        "model_loaded": ok,
        "model_version": _metrics.get("model_version"),
        "n_features": _metrics.get("n_features"),
        "groq_configured": settings.groq_configured,
    }


@app.get("/model-info")
def model_info():
    axis = np.asarray(_bundle.get("axis", []), dtype=float)
    return {
        "model_version": _metrics.get("model_version"),
        "model_type": "PLS + BayesianRidge + PCA-MSPC",
        "target": "normalized_density",
        "n_features": _metrics.get("n_features"),
        "n_train": _metrics.get("n_train"),
        "wavelength_axis": {
            "unit": "cm-1",
            "min": float(axis.min()) if axis.size else None,
            "max": float(axis.max()) if axis.size else None,
        },
        "data_source": _metrics.get("source"),
        "runtime_data_required": False,
        "disclaimer": "Public research data only; not industrial plant data.",
    }


@app.get("/demo-spectrum")
def demo_spectrum():
    """Return the training-mean spectrum stored in the model bundle.

    This is a compact derived artifact, not the raw training dataset.
    """
    axis = np.asarray(_bundle["axis"], dtype=float)
    mean_spectrum = np.asarray(_bundle["mu"], dtype=float)
    return {
        "type": "derived_demo_spectrum",
        "source": "MLNIRdata training-set mean",
        "axis_unit": "cm-1",
        "axis": axis.tolist(),
        "spectrum": mean_spectrum.tolist(),
        "n_features": len(mean_spectrum),
        "note": "Derived model artifact for UI demonstration; not a raw dataset export.",
    }


@app.get("/metrics")
def metrics():
    return _metrics


@app.post("/predict")
def predict(req: SpectrumRequest):
    x = _validate_spectrum(req.spectrum)
    centered = x - _bundle["mu"]
    pred = float(_bundle["pls"].predict(centered)[0])
    interval = _bundle["bayes"].predict_interval(centered)
    payload = {
        "type": "prediction",
        "model_version": _metrics.get("model_version"),
        "prediction": pred,
        "uncertainty": {
            "lower": float(interval["lower"][0]),
            "upper": float(interval["upper"][0]),
            "std": float(interval["std"][0]),
            "nominal_level": 0.95,
        },
        "target": "normalized_density",
        "status": "ok",
        "data_classification": "PUBLIC_RESEARCH_MODEL",
    }
    return _with_explanation(payload, req.explain, req.audience)


@app.post("/monitor")
def monitor(req: SpectrumRequest):
    x = _validate_spectrum(req.spectrum)
    m = _bundle["monitor"].monitor(x - _bundle["mu"])
    payload = {
        "type": "monitoring",
        "T2": float(m["T2"][0]),
        "Q": float(m["Q"][0]),
        "T2_limit": float(m["T2_limit"][0]),
        "Q_limit": float(m["Q_limit"][0]),
        "anomaly": bool(m["anomaly"][0]),
        "status": "anomaly" if bool(m["anomaly"][0]) else "normal",
        "interpretation": "Statistical deviation from the model reference space; not proof of causality.",
    }
    return _with_explanation(payload, req.explain, req.audience)


@app.post("/anomaly")
def anomaly(req: SpectrumRequest):
    x = _validate_spectrum(req.spectrum)
    result = _bundle["monitor"].root_cause(x - _bundle["mu"], sample_idx=0, top_k=10)
    result["type"] = "root_cause"
    result["interpretation"] = "Largest statistical contributors to Q residual; not causal attribution."
    return _with_explanation(result, req.explain, req.audience)


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
    result["data_classification"] = "MECHANISTIC_SIMULATION"
    result["note"] = "Synthetic CSTR/Arrhenius demonstration only; not plant operating guidance."
    return _with_explanation(result, req.explain, req.audience)


@app.post("/explain")
def explain(req: ExplainRequest):
    # Bound the serialized prompt size before it reaches an external LLM provider.
    serialized = json.dumps(req.context, default=str)
    if len(serialized) > 6000:
        raise HTTPException(413, "Explanation context is too large; limit it to 6000 characters.")
    return _explainer.explain(req.context, audience=req.audience, detail=req.detail)


@app.post("/admin/retrain")
def retrain():
    if settings.is_production:
        raise HTTPException(403, "Retraining is disabled in production.")
    if not settings.allow_runtime_training:
        raise HTTPException(403, "Set CHEMX_ALLOW_RUNTIME_TRAINING=true for local development.")
    from src.utils.model_registry import train_and_save
    global _bundle, _metrics
    _metrics = train_and_save(force=True)
    _bundle, _metrics = load_bundle()
    return {"status": "retrained", "metrics": _metrics}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error path=%s", request.url.path)
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )
    response.headers["X-Request-ID"] = request_id
    return response
