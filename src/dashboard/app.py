"""ChemX Streamlit control tower.

This frontend is intentionally API-driven. It does not require raw datasets or
training libraries at runtime; Render hosts the FastAPI model service.
"""

from __future__ import annotations

import json
import os
from io import StringIO
from typing import Any, Dict

import httpx
import numpy as np
import streamlit as st

st.set_page_config(page_title="ChemX | Process Intelligence", page_icon="⚗️", layout="wide")

API_BASE_URL = os.getenv("CHEMX_API_URL", "").rstrip("/")
if not API_BASE_URL:
    try:
        API_BASE_URL = str(st.secrets.get("CHEMX_API_URL", "")).rstrip("/")
    except Exception:
        API_BASE_URL = ""

if not API_BASE_URL:
    API_BASE_URL = "http://localhost:8000"

TIMEOUT = httpx.Timeout(20.0, connect=5.0)


def api_get(path: str) -> Dict[str, Any]:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.get(f"{API_BASE_URL}{path}")
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.post(f"{API_BASE_URL}{path}", json=payload)
            if r.status_code >= 400:
                try:
                    detail = r.json().get("detail", r.text)
                except Exception:
                    detail = r.text
                return {"_error": f"HTTP {r.status_code}: {detail}"}
            return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


@st.cache_data(ttl=60, show_spinner=False)
def cached_get(path: str):
    return api_get(path)


def error_box(result: Dict[str, Any]) -> bool:
    if "_error" in result:
        st.error(result["_error"])
        return True
    return False


def kpi(label: str, value: str, help_text: str = ""):
    st.metric(label, value, help=help_text or None)


def explanation(result: Dict[str, Any]):
    text = result.get("stakeholder_explanation")
    if not text:
        return
    st.info(text.get("explanation", text.get("message", "No narrative returned.")))


@st.cache_data(ttl=300, show_spinner="Loading model metadata…")
def model_info():
    return cached_get("/model-info")


@st.cache_data(ttl=300, show_spinner="Loading demo spectrum…")
def demo_spectrum():
    return cached_get("/demo-spectrum")


def parse_uploaded_spectrum(raw: str) -> np.ndarray:
    """Parse a headerless or header-containing CSV/TXT spectrum safely."""
    attempts = [
        lambda: np.genfromtxt(StringIO(raw), delimiter=",", dtype=float),
        lambda: np.genfromtxt(StringIO(raw), delimiter=None, dtype=float),
    ]
    for parser in attempts:
        values = np.asarray(parser(), dtype=float).reshape(-1)
        if values.size:
            finite = values[np.isfinite(values)]
            if finite.size:
                return finite
    raise ValueError("No numeric spectral values were found in the uploaded file.")


st.markdown(
    """
<style>
.block-container {max-width: 1380px; padding-top: 1.5rem;}
.hero {padding: 1.5rem 1.8rem; border-radius: 16px; background: #10232F; color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0 0 .25rem; font-size:2rem;}
.hero p {color:#B9CBD4; margin:0; max-width:850px;}
.badge {display:inline-block; padding:.25rem .6rem; border-radius:999px; background:#E8F5F4; color:#0E706D; font-weight:700; font-size:.78rem;}
.small {color:#637783; font-size:.85rem;}
</style>
""",
    unsafe_allow_html=True,
)

ready = cached_get("/ready")
health = cached_get("/health")
info = model_info()

with st.sidebar:
    st.markdown("## ⚗️ ChemX")
    st.caption("Physics-informed chemometric process intelligence")
    page = st.radio(
        "Navigate",
        ["Overview", "Spectroscopy", "Soft Sensor", "Process Monitoring", "Root Cause", "Chemometrics", "Physics + Optimization", "Stakeholder Explain"],
    )
    st.divider()
    st.caption(f"Backend: {API_BASE_URL}")
    if health.get("status") == "ok":
        st.success("API online")
    else:
        st.error("API unavailable")
    if ready.get("ready"):
        st.success("Model ready")
    else:
        st.warning("Model not ready")
    st.caption("Public research data only.")
    st.caption("Synthetic CSTR outputs are explicitly labelled.")

st.markdown(
    """
<div class="hero">
<h1>ChemX — Process Intelligence Control Tower</h1>
<p>Spectroscopy → Chemometrics → Bayesian soft sensing → multivariate monitoring →
statistical contribution analysis → mechanistic optimization.</p>
</div>
""",
    unsafe_allow_html=True,
)

if page == "Overview":
    st.subheader("Deployment status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("API", "Online" if health.get("status") == "ok" else "Offline")
    with c2:
        kpi("Model", "Ready" if ready.get("ready") else "Unavailable")
    with c3:
        kpi("Features", str(info.get("n_features", "—")))
    with c4:
        kpi("Model version", str(info.get("model_version", "—")))

    st.markdown("### Scientific boundary")
    st.markdown(
        '<span class="badge">REAL PUBLIC DATA</span> MLNIRdata supports the deployed NIR '
        'density model. <span class="badge">MECHANISTIC_SIMULATION</span> is used only for '
        "the CSTR physics and optimization demonstration.",
        unsafe_allow_html=True,
    )
    st.markdown("### Runtime architecture")
    st.code(
        "\n".join(
            [
                "Streamlit UI",
                "    ↓ HTTPS JSON",
                "Render FastAPI",
                "    ↓",
                "Versioned MLNIR model bundle",
                "    ↓",
                "Prediction / UQ / MSPC / optimization",
            ]
        ),
        language="text",
    )
    st.markdown("### Model provenance")
    st.json(
        {
            "source": info.get("data_source"),
            "target": info.get("target"),
            "model": info.get("model_type"),
            "runtime_data_required": info.get("runtime_data_required"),
            "disclaimer": info.get("disclaimer"),
        }
    )

elif page == "Spectroscopy":
    st.subheader("Spectroscopy")
    d = demo_spectrum()
    if error_box(d):
        st.stop()
    axis = np.asarray(d["axis"], dtype=float)
    spectrum = np.asarray(d["spectrum"], dtype=float)
    st.caption("Derived training-set mean spectrum is used for demonstration; raw datasets are not bundled with the frontend.")
    st.line_chart(spectrum)
    st.write(
        f"**{len(spectrum):,} spectral variables** · "
        f"axis: {d['axis_unit']} · range: {axis.min():.0f}–{axis.max():.0f}"
    )
    uploaded = st.file_uploader("Upload a single-column or comma-separated spectrum", type=["csv", "txt"])
    if uploaded:
        raw = uploaded.getvalue().decode("utf-8")
        try:
            values = parse_uploaded_spectrum(raw)
            if len(values) != int(info["n_features"]):
                st.error(f"Expected {info['n_features']} values; received {len(values)}.")
            else:
                st.session_state["spectrum"] = values.tolist()
                st.success("Spectrum loaded. Use Soft Sensor or Process Monitoring.")
        except Exception as exc:
            st.error(f"Could not parse spectrum: {exc}")

elif page == "Soft Sensor":
    st.subheader("Bayesian soft sensor")
    d = demo_spectrum()
    if error_box(d):
        st.stop()
    default = st.session_state.get("spectrum", d["spectrum"])
    with st.form("predict"):
        st.caption(f"Input must contain exactly {info.get('n_features')} values in model wavelength order.")
        explain_flag = st.checkbox("Generate stakeholder explanation", value=False)
        submitted = st.form_submit_button("Predict density", type="primary")
    if submitted:
        result = api_post("/predict", {"spectrum": default, "explain": explain_flag, "audience": "operations"})
        if not error_box(result):
            u = result["uncertainty"]
            a, b, c = st.columns(3)
            with a:
                kpi("Prediction", f"{result['prediction']:.4f}")
            with b:
                kpi("95% lower", f"{u['lower']:.4f}")
            with c:
                kpi("95% upper", f"{u['upper']:.4f}")
            st.caption("Interval calibration is empirical on the held-out public dataset; nominal 95% does not imply guaranteed coverage.")
            explanation(result)

elif page == "Process Monitoring":
    st.subheader("Multivariate process monitoring")
    d = demo_spectrum()
    spectrum = st.session_state.get("spectrum", d.get("spectrum", []))
    if st.button("Run monitoring", type="primary"):
        result = api_post("/monitor", {"spectrum": spectrum, "explain": True, "audience": "operations"})
        if not error_box(result):
            a, b, c = st.columns(3)
            with a:
                kpi("Hotelling T²", f"{result['T2']:.3f}")
            with b:
                kpi("Q residual", f"{result['Q']:.3f}")
            with c:
                kpi("Status", result["status"].upper())
            st.json(result)
            explanation(result)

elif page == "Root Cause":
    st.subheader("Statistical contribution analysis")
    d = demo_spectrum()
    spectrum = st.session_state.get("spectrum", d.get("spectrum", []))
    st.warning("Contributions identify variables with large statistical influence on the residual; they do not establish physical causality.")
    if st.button("Analyze contributions", type="primary"):
        result = api_post("/anomaly", {"spectrum": spectrum, "explain": True, "audience": "quality"})
        if not error_box(result):
            st.write("### Largest statistical contributors")
            rows = [
                {"rank": i + 1, "spectral_index": idx, "contribution": value}
                for i, (idx, value) in enumerate(
                    zip(result.get("top_contributor_indices", []), result.get("top_contributor_values", []))
                )
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.json(result)
            explanation(result)

elif page == "Chemometrics":
    st.subheader("Chemometric model card")
    metrics = cached_get("/metrics")
    if error_box(metrics):
        st.stop()
    a, b, c, d = st.columns(4)
    with a:
        kpi("PLS test RMSE", f"{metrics.get('pls_test_rmse', float('nan')):.4f}")
    with b:
        kpi("Bayesian coverage", f"{100*metrics.get('bayesian_coverage_95', float('nan')):.1f}%")
    with c:
        kpi("Train samples", str(metrics.get("n_train", "—")) )
    with d:
        kpi("Spectral variables", str(metrics.get("n_features", "—")) )
    st.markdown("### Methods")
    st.markdown(
        "- PLS latent-variable calibration\n"
        "- BayesianRidge soft sensor with empirical interval coverage\n"
        "- PCA-based Hotelling T² / Q monitoring\n"
        "- VIP wavelength ranking\n"
        "- Corn multi-instrument transfer is evaluated offline, not served by the runtime API"
    )
    st.markdown("### Provenance")
    st.code(str(metrics.get("source", "unknown")))

elif page == "Physics + Optimization":
    st.subheader("Physics-informed decision demonstration")
    st.warning("MECHANISTIC_SIMULATION ONLY — outputs are not plant recommendations.")
    risk = st.checkbox("Use risk-aware Monte Carlo constraint", value=True)
    threshold = st.slider("Quality threshold", 0.0, 1.0, 0.70, 0.01)
    probability = st.slider("Minimum probability", 0.50, 0.99, 0.90, 0.01)
    if st.button("Run optimization", type="primary"):
        result = api_post(
            "/optimize",
            {"risk_aware": risk, "min_prob": probability, "quality_threshold": threshold, "explain": True, "audience": "management"},
        )
        if not error_box(result):
            st.json(result)
            if result.get("success") is False:
                st.error("The requested risk-constrained problem was not solved. No optimum is being presented as valid.")
            explanation(result)

elif page == "Stakeholder Explain":
    st.subheader("Stakeholder explanation")
    st.caption("Groq is optional. The backend always returns a deterministic fallback when no API key is configured.")
    default_context = {
        "type": "prediction",
        "prediction": 0.5,
        "uncertainty": {"lower": 0.45, "upper": 0.55},
        "status": "illustrative",
        "note": "Illustrative context only.",
    }
    context_text = st.text_area("Result JSON", json.dumps(default_context, indent=2), height=220)
    audience = st.selectbox("Audience", ["operations", "quality", "management"])
    detail = st.selectbox("Detail", ["brief", "standard", "detailed"])
    if st.button("Explain", type="primary"):
        try:
            context = json.loads(context_text)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        else:
            result = api_post("/explain", {"context": context, "audience": audience, "detail": detail})
            if not error_box(result):
                st.markdown(result.get("explanation", result.get("message", "")))
                st.json(result)
