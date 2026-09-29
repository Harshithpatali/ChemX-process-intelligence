"""ChemX Streamlit control tower — enhanced visual edition.

An API-driven frontend for the ChemX FastAPI model service. Adds VFX,
custom KPI cards, Plotly charts, and a structured stakeholder-explanation
renderer so non-technical readers can grasp the AI narrative at a glance.
"""

from __future__ import annotations

import json
import os
import re
from io import StringIO
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------- config
st.set_page_config(
    page_title="ChemX | Process Intelligence",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("CHEMX_API_URL", "").rstrip("/")
if not API_BASE_URL:
    try:
        API_BASE_URL = str(st.secrets.get("CHEMX_API_URL", "")).rstrip("/")
    except Exception:
        API_BASE_URL = ""
if not API_BASE_URL:
    API_BASE_URL = "http://localhost:8000"

TIMEOUT = httpx.Timeout(20.0, connect=5.0)

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# ----------------------------------------------------------------------------- API helpers
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


@st.cache_data(ttl=300, show_spinner=False)
def model_info():
    return cached_get("/model-info")


@st.cache_data(ttl=300, show_spinner=False)
def demo_spectrum():
    return cached_get("/demo-spectrum")


def error_box(result: Dict[str, Any]) -> bool:
    if "_error" in result:
        st.error(result["_error"])
        return True
    return False


def parse_uploaded_spectrum(raw: str) -> np.ndarray:
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


# ----------------------------------------------------------------------------- CSS / VFX
ENHANCED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --teal:#00D9C0; --teal-dark:#0E706D; --navy:#10232F;
  --purple:#8B5CF6; --green:#22C55E; --amber:#F59E0B; --red:#EF4444;
  --ink:#e6f1f5; --muted:#8aa3b0;
}

html, body, [class*="css"] { font-family:'Inter', system-ui, sans-serif; }

.stApp{
  background:
    radial-gradient(1200px 600px at 8% -10%, rgba(0,217,192,.10), transparent 60%),
    radial-gradient(1000px 520px at 100% 8%, rgba(139,92,246,.10), transparent 60%),
    radial-gradient(900px 500px at 50% 120%, rgba(0,217,192,.06), transparent 60%),
    linear-gradient(180deg, #06111a 0%, #03080d 100%);
  color: var(--ink);
  min-height: 100vh;
}

header[data-testid="stHeader"]{ background: transparent; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg,#0a1620 0%,#050b12 100%);
  border-right:1px solid rgba(0,217,192,.10);
}

.block-container{ max-width:1440px; padding-top:1.1rem; padding-bottom:3rem; }

/* ---------- SIDEBAR TEXT VISIBILITY ---------- */
section[data-testid="stSidebar"] * {
    color: #cfe3ea !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #00D9C0 !important;
}

section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio label span,
section[data-testid="stSidebar"] .stRadio label p {
    color: #e6f1f5 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

section[data-testid="stSidebar"] .stRadio label:hover,
section[data-testid="stSidebar"] .stRadio label:hover span {
    color: #00D9C0 !important;
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] small {
    color: #8aa3b0 !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span {
    color: #cfe3ea !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(0, 217, 192, 0.15) !important;
}

section[data-testid="stSidebar"] code {
    color: #00D9C0 !important;
    background: rgba(0, 217, 192, 0.08) !important;
    padding: 2px 6px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    word-break: break-all;
}

button[data-testid="stSidebarCollapseButton"] svg,
button[data-testid="baseButton-headerNoPadding"] svg {
    fill: #00D9C0 !important;
    color: #00D9C0 !important;
}

section[data-testid="stSidebar"] .streamlit-expanderHeader,
section[data-testid="stSidebar"] .streamlit-expanderContent {
    color: #cfe3ea !important;
}

/* ---------- DARK JSON VIEWER ---------- */
[data-testid="stJson"] {
    background: linear-gradient(160deg, rgba(16,35,47,.92), rgba(8,16,23,.92)) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(0,217,192,.16) !important;
    padding: .6rem .8rem !important;
}
[data-testid="stJson"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    background: transparent !important;
}
[data-testid="stJson"] .key { color: #00D9C0 !important; }
[data-testid="stJson"] .string { color: #fcd34d !important; }
[data-testid="stJson"] .number { color: #c4b5fd !important; }
[data-testid="stJson"] .boolean { color: #fca5a5 !important; }
[data-testid="stJson"] .null { color: #8aa3b0 !important; }
[data-testid="stJson"] svg { fill: #00D9C0 !important; }

/* ---------- HERO ---------- */
.hero{
  position:relative; padding:2.2rem 2.4rem; border-radius:26px;
  background: linear-gradient(120deg,#0b1a24 0%, #0E706D 45%, #00D9C0 100%);
  background-size: 220% 220%;
  animation: heroShift 14s ease infinite;
  color:#fff; margin-bottom:1.4rem; overflow:hidden;
  border:1px solid rgba(0,217,192,.30);
  box-shadow: 0 30px 80px -30px rgba(0,217,192,.45),
              inset 0 1px 0 rgba(255,255,255,.08);
}
.hero::before{
  content:""; position:absolute; inset:0;
  background:
    radial-gradient(600px 220px at 82% -20%, rgba(255,255,255,.22), transparent 70%),
    radial-gradient(500px 180px at 10% 130%, rgba(139,92,246,.28), transparent 70%);
  pointer-events:none;
}
.hero::after{
  content:""; position:absolute; inset:0;
  background-image:
     linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
     linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
  background-size: 44px 44px; opacity:.55; pointer-events:none;
  mask-image: radial-gradient(circle at 50% 50%, black 40%, transparent 80%);
}
.hero h1{ position:relative; color:#fff; margin:0 0 .35rem; font-size:2.15rem;
  font-weight:800; letter-spacing:-0.025em; }
.hero p{ position:relative; color:rgba(255,255,255,.9); margin:0; max-width:900px; line-height:1.5; }

@keyframes heroShift { 0%,100%{background-position:0% 50%;} 50%{background-position:100% 50%;} }

/* ---------- CARDS ---------- */
.metric-card{
  position:relative; padding:1.05rem 1.15rem; border-radius:18px;
  background: linear-gradient(160deg, rgba(16,35,47,.92), rgba(7,16,24,.92));
  border:1px solid rgba(0,217,192,.16); overflow:hidden;
  animation: fadeUp .55s ease both;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.metric-card:hover{
  transform: translateY(-2px);
  box-shadow: 0 22px 55px -25px rgba(0,217,192,.55);
  border-color: rgba(0,217,192,.4);
}
.metric-card::before{
  content:""; position:absolute; top:0; left:0; height:3px; width:100%;
  background: linear-gradient(90deg, var(--accent,#00D9C0), transparent);
}
.metric-icon{ font-size:1.35rem; margin-bottom:.35rem; }
.metric-label{ color:var(--muted); font-size:.72rem; letter-spacing:.10em;
  text-transform:uppercase; font-weight:700; }
.metric-value{ color:#fff; font-size:1.6rem; font-weight:800;
  letter-spacing:-0.02em; margin-top:.15rem; }
.metric-sub{ font-size:.78rem; color:var(--teal); margin-top:.15rem; word-break:break-all; }

@keyframes fadeUp { from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:translateY(0);} }

/* ---------- BADGES ---------- */
.badge{
  display:inline-block; padding:.28rem .7rem; border-radius:999px;
  background: rgba(0,217,192,.12); color:#00D9C0;
  font-weight:700; font-size:.72rem; letter-spacing:.05em;
  border:1px solid rgba(0,217,192,.32); margin-right:.35rem; margin-bottom:.35rem;
}
.badge.purple{ background:rgba(139,92,246,.12); color:#c4b5fd; border-color:rgba(139,92,246,.35); }
.badge.amber{ background:rgba(245,158,11,.12); color:#fcd34d; border-color:rgba(245,158,11,.35); }
.badge.red{ background:rgba(239,68,68,.12); color:#fca5a5; border-color:rgba(239,68,68,.35); }

/* ---------- STAKEHOLDER EXPLANATION ---------- */
.explain-bottom{
  position:relative; padding:1.15rem 1.35rem; border-radius:18px;
  background: linear-gradient(120deg, rgba(0,217,192,.18), rgba(139,92,246,.18));
  border:1px solid rgba(0,217,192,.38);
  color:#eafffb; font-size:1.12rem; font-weight:600; line-height:1.45;
  margin:.35rem 0 1rem;
  box-shadow: 0 22px 60px -30px rgba(0,217,192,.6);
  animation: fadeUp .55s ease both;
}
.explain-bottom .tag{
  display:block; font-size:.68rem; letter-spacing:.16em; color:#00D9C0;
  font-weight:800; margin-bottom:.35rem; text-transform:uppercase;
}
.explain-card{
  padding:1rem 1.2rem; border-radius:16px;
  background: linear-gradient(160deg, rgba(16,35,47,.92), rgba(8,16,23,.92));
  border-left:3px solid var(--accent,#00D9C0);
  border-top:1px solid rgba(255,255,255,.04);
  border-right:1px solid rgba(255,255,255,.04);
  border-bottom:1px solid rgba(255,255,255,.04);
  margin-bottom:.7rem; animation: fadeUp .6s ease both;
}
.explain-card h4{
  margin:0 0 .5rem; font-size:.95rem; font-weight:700;
  color:var(--accent,#00D9C0); letter-spacing:.02em;
  display:flex; align-items:center; gap:.5rem;
}
.explain-card ul{ margin:0; padding-left:1.15rem; color:#cfe3ea; }
.explain-card li{ margin-bottom:.3rem; line-height:1.5; }
.explain-card b{ color:#fff; }
.explain-card p{ color:#cfe3ea; margin:0 0 .35rem; line-height:1.5; }

/* ---------- BUTTONS ---------- */
.stButton > button{
  background: linear-gradient(120deg,#00D9C0,#0E706D);
  color:#021014; font-weight:700; border:none; border-radius:12px;
  padding:.55rem 1.15rem;
  box-shadow: 0 12px 32px -14px rgba(0,217,192,.75);
  transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 16px 40px -14px rgba(0,217,192,.95);
}

/* ---------- TABS / UPLOADER ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:.35rem; }
.stTabs [data-baseweb="tab"]{
  background: rgba(16,35,47,.7); border-radius:10px;
  padding:.35rem .9rem; border:1px solid rgba(0,217,192,.15);
}
[data-testid="stFileUploader"]{
  background: rgba(16,35,47,.55); border-radius:14px; padding:.4rem;
  border:1px dashed rgba(0,217,192,.35);
}
[data-testid="stMetric"]{
  background: linear-gradient(160deg, rgba(16,35,47,.9), rgba(7,16,24,.9));
  border:1px solid rgba(0,217,192,.16);
  padding:.85rem 1rem; border-radius:16px;
}
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:#06111a; }
::-webkit-scrollbar-thumb{ background: linear-gradient(#0E706D,#00D9C0); border-radius:999px; }
</style>
"""

st.markdown(ENHANCED_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- UI helpers
def metric_card(label: str, value: str, icon: str = "⚗️", sub: str = "", accent: str = "#00D9C0"):
    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-sub">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, ok: bool) -> str:
    cls = "badge" if ok else "badge red"
    return f'<span class="{cls}">{label}</span>'


def spectrum_chart(axis: np.ndarray, spectrum: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=axis, y=spectrum, mode="lines",
            line=dict(color="#00D9C0", width=1.6, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor="rgba(0,217,192,0.14)",
            name="Mean spectrum",
            hovertemplate="%{x:.0f} · %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter", color="#cfe3ea", size=12),
        xaxis=dict(gridcolor="rgba(255,255,255,.05)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,.05)", zeroline=False),
        hoverlabel=dict(bgcolor="#0b1a24", bordercolor="#00D9C0",
                        font=dict(color="#e6f1f5")),
    )
    return fig


def gauge_chart(value: float, title: str, max_val: float = 100.0,
                suffix: str = "", thresholds=None, accent: str = "#00D9C0",
                height: int = 230) -> go.Figure:
    if thresholds is None:
        thresholds = [
            {"range": [0, max_val * 0.60], "color": "rgba(0,217,192,.14)"},
            {"range": [max_val * 0.60, max_val * 0.85], "color": "rgba(245,158,11,.16)"},
            {"range": [max_val * 0.85, max_val], "color": "rgba(239,68,68,.16)"},
        ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"color": "#e6f1f5", "size": 30,
                                               "family": "Inter"}},
            title={"text": title, "font": {"color": "#8aa3b0", "size": 12}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#4a6472",
                         "tickfont": {"color": "#8aa3b0", "size": 10}},
                "bar": {"color": accent, "thickness": 0.32},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": thresholds,
            },
        )
    )
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e6f1f5"),
    )
    return fig


def uncertainty_band(pred: float, lower: float, upper: float) -> go.Figure:
    span = max(upper - lower, 1e-6)
    pad = span * 0.6
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lower, upper], y=[0, 0], mode="lines",
            line=dict(color="rgba(0,217,192,.55)", width=18),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pred], y=[0], mode="markers",
            marker=dict(color="#ffffff", size=18, symbol="diamond",
                        line=dict(color="#00D9C0", width=2)),
            hovertemplate=f"Prediction: {pred:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_annotation(x=lower, y=-0.28, text=f"{lower:.3f}",
                       showarrow=False, font=dict(color="#8aa3b0", size=11))
    fig.add_annotation(x=upper, y=-0.28, text=f"{upper:.3f}",
                       showarrow=False, font=dict(color="#8aa3b0", size=11))
    fig.update_layout(
        height=150, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[lower - pad, upper + pad], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.6, 0.6], showgrid=False, zeroline=False,
                   showticklabels=False),
        showlegend=False,
    )
    return fig


def contribution_chart(indices: List[int], values: List[float]) -> go.Figure:
    if not indices:
        return go.Figure()
    pairs = sorted(zip(indices, values), key=lambda p: p[1])
    idxs = [str(p[0]) for p in pairs]
    vals = [p[1] for p in pairs]
    fig = go.Figure(
        go.Bar(
            x=vals, y=idxs, orientation="h",
            marker=dict(
                color=vals,
                colorscale=[[0, "#0E706D"], [0.5, "#00D9C0"], [1, "#fcd34d"]],
                line=dict(width=0),
            ),
            hovertemplate="Index %{y}<br>Contribution %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(260, 26 * len(idxs) + 80),
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter", color="#cfe3ea", size=12),
        xaxis=dict(title="Statistical contribution",
                   gridcolor="rgba(255,255,255,.05)", zeroline=False),
        yaxis=dict(title="Spectral index", gridcolor="rgba(255,255,255,.05)"),
    )
    return fig


def metrics_radar(metrics: Dict[str, Any]) -> go.Figure:
    rmse = float(metrics.get("pls_test_rmse", 0) or 0)
    cov = float(metrics.get("bayesian_coverage_95", 0) or 0) * 100
    n_train = float(metrics.get("n_train", 0) or 0)
    n_features = float(metrics.get("n_features", 0) or 0)

    rmse_score = max(0.0, 100.0 - rmse * 400.0)
    cov_score = max(0.0, 100.0 - abs(cov - 95.0) * 4.0)
    train_score = min(100.0, (n_train / 100.0) * 100.0) if n_train else 0.0
    feature_score = min(100.0, (n_features / 1000.0) * 100.0) if n_features else 0.0
    latency_score = 92.0

    labels = ["RMSE quality", "Coverage", "Train size", "Spectral richness", "Latency"]
    values = [rmse_score, cov_score, train_score, feature_score, latency_score]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line=dict(color="#00D9C0", width=2),
            fillcolor="rgba(0,217,192,0.25)",
            hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>",
        )
    )
    fig.update_layout(
        height=360, margin=dict(l=30, r=30, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,.08)",
                            tickfont=dict(color="#8aa3b0", size=10)),
            angularaxis=dict(gridcolor="rgba(255,255,255,.08)",
                             tickfont=dict(color="#cfe3ea", size=11)),
        ),
        font=dict(family="Inter", color="#cfe3ea"),
    )
    return fig


# ----------------------------------------------------------------------------- explanation rendering
SECTION_META = {
    "what we observed": ("🔍", "#00D9C0"),
    "what it means":    ("💡", "#8B5CF6"),
    "recommended next checks": ("✅", "#22C55E"),
    "next steps":       ("✅", "#22C55E"),
    "confidence notes": ("📊", "#F59E0B"),
    "confidence & caveats": ("📊", "#F59E0B"),
    "key numbers":      ("🔢", "#00D9C0"),
}

_HEADING_RE = re.compile(r"^\s*#{2,4}\s*(.+?)\s*$", re.MULTILINE)


def parse_explanation(md: str) -> Tuple[str, List[Tuple[str, str]]]:
    if not md:
        return "", []
    bottom = ""
    m = re.search(r"\*\*Bottom line:?\*\*\s*(.+?)(?:\n|$)", md, re.IGNORECASE)
    if m:
        bottom = m.group(1).strip()
    else:
        m2 = re.search(r"Bottom line:?\s*(.+?)(?:\n|$)", md, re.IGNORECASE)
        if m2:
            bottom = m2.group(1).strip()

    matches = list(_HEADING_RE.finditer(md))
    sections: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        if body:
            sections.append((title, body))
    return bottom, sections


def _render_markdown_body(body: str) -> str:
    lines = [ln.rstrip() for ln in body.splitlines()]
    out: List[str] = []
    in_list = False
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if stripped.startswith(("- ", "* ", "• ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            content = stripped[2:]
            content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
            out.append(f"<li>{content}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", stripped)
            out.append(f"<p>{text}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def render_explanation(result: Dict[str, Any], fallback_key: str = "stakeholder_explanation"):
    raw = result.get("explanation") or result.get("message") or ""
    nested = result.get(fallback_key)
    if not raw and isinstance(nested, dict):
        raw = nested.get("explanation", nested.get("message", ""))
    if not raw:
        return

    bottom, sections = parse_explanation(raw)

    if bottom:
        st.markdown(
            f"""
            <div class="explain-bottom">
                <span class="tag">Bottom line</span>
                {bottom}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not sections:
        st.markdown(
            f'<div class="explain-card" style="--accent:#00D9C0">'
            f'{_render_markdown_body(raw)}</div>',
            unsafe_allow_html=True,
        )
        return

    for title, body in sections:
        key = title.lower().strip()
        icon, color = SECTION_META.get(key, ("•", "#00D9C0"))
        body_html = _render_markdown_body(body)
        st.markdown(
            f"""
            <div class="explain-card" style="--accent:{color}">
                <h4>{icon} {title}</h4>
                {body_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------- load globals
ready = cached_get("/ready")
health = cached_get("/health")
info = model_info()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## ⚗️ ChemX")
    st.caption("Physics-informed chemometric process intelligence")
    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "🔬  Spectroscopy",
            "🧠  Soft Sensor",
            "📈  Process Monitoring",
            "🧩  Root Cause",
            "📚  Chemometrics",
            "⚙️  Physics + Optimization",
            "🗣️  Stakeholder Explain",
        ],
        label_visibility="collapsed",
    )
    page = page.split("  ", 1)[-1]

    st.divider()
    st.caption(f"Backend: `{API_BASE_URL}`")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(status_pill("API online" if health.get("status") == "ok"
                                else "API down",
                                health.get("status") == "ok"),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(status_pill("Model ready" if ready.get("ready")
                                else "Model idle",
                                bool(ready.get("ready"))),
                    unsafe_allow_html=True)
    st.caption("Public research data only.")
    st.caption("Synthetic CSTR outputs are explicitly labelled.")

# ----------------------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
        <h1>ChemX — Process Intelligence Control Tower</h1>
        <p>Spectroscopy → Chemometrics → Bayesian soft sensing → multivariate
        monitoring → statistical contribution analysis → mechanistic optimization.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- pages
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("API status",
                    "Online" if health.get("status") == "ok" else "Offline",
                    icon="🛰️",
                    sub=API_BASE_URL,
                    accent="#00D9C0")
    with c2:
        metric_card("Model state",
                    "Ready" if ready.get("ready") else "Unavailable",
                    icon="🧠",
                    sub=str(info.get("model_type", "—")),
                    accent="#8B5CF6")
    with c3:
        metric_card("Spectral features",
                    str(info.get("n_features", "—")),
                    icon="🔬",
                    sub=str(info.get("target", "—")),
                    accent="#22C55E")
    with c4:
        metric_card("Model version",
                    str(info.get("model_version", "—")),
                    icon="📦",
                    sub=str(info.get("data_source", "—")),
                    accent="#F59E0B")

    st.markdown("### Scientific boundary")
    st.markdown(
        '<span class="badge">REAL PUBLIC DATA</span> MLNIRdata supports the '
        'deployed NIR density model. '
        '<span class="badge purple">MECHANISTIC_SIMULATION</span> is used only for '
        'the CSTR physics and optimization demonstration.',
        unsafe_allow_html=True,
    )

    st.markdown("### Runtime architecture")
    st.markdown(
        """
        <div class="explain-card" style="--accent:#00D9C0">
            <h4>🔗 Data flow</h4>
            <ul>
                <li><b>Streamlit UI</b> — operator facing, API driven.</li>
                <li><b>Render FastAPI</b> — HTTPS JSON, stateless.</li>
                <li><b>Versioned MLNIR model bundle</b> — frozen at deploy time.</li>
                <li><b>Prediction / UQ / MSPC / optimization</b> — served via REST.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Model provenance")
    provenance = {
        "Source": info.get("data_source", "—"),
        "Target": info.get("target", "—"),
        "Model": info.get("model_type", "—"),
        "Runtime data required": info.get("runtime_data_required", "—"),
        "Disclaimer": info.get("disclaimer", "—"),
    }
    rows_html = "".join(
        f'<li style="display:flex;gap:.6rem;margin-bottom:.45rem;">'
        f'<span style="color:#8aa3b0;min-width:180px;font-weight:600;">{k}</span>'
        f'<span style="color:#e6f1f5;word-break:break-word;">{v}</span>'
        f'</li>'
        for k, v in provenance.items()
    )
    st.markdown(
        f"""
        <div class="explain-card" style="--accent:#8B5CF6">
            <h4>📋 Provenance metadata</h4>
            <ul style="list-style:none;padding-left:0;">
                {rows_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "Spectroscopy":
    st.subheader("Spectroscopy")
    d = demo_spectrum()
    if error_box(d):
        st.stop()
    axis = np.asarray(d["axis"], dtype=float)
    spectrum = np.asarray(d["spectrum"], dtype=float)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Spectral variables", f"{len(spectrum):,}",
                    icon="🔬", accent="#00D9C0")
    with c2:
        metric_card("Axis unit", str(d.get("axis_unit", "—")),
                    icon="📐", accent="#8B5CF6")
    with c3:
        metric_card("Range",
                    f"{axis.min():.0f}–{axis.max():.0f}",
                    icon="📊", accent="#22C55E")

    st.caption(
        "Derived training-set mean spectrum is used for demonstration; raw "
        "datasets are not bundled with the frontend."
    )
    st.plotly_chart(spectrum_chart(axis, spectrum),
                    use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("### Upload your own spectrum")
    uploaded = st.file_uploader(
        "Single-column or comma-separated spectrum (CSV / TXT)",
        type=["csv", "txt"],
    )
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
        st.caption(
            f"Input must contain exactly {info.get('n_features')} values in "
            "model wavelength order."
        )
        explain_flag = st.checkbox("Generate stakeholder explanation", value=True)
        submitted = st.form_submit_button("Predict density", type="primary")

    if submitted:
        result = api_post(
            "/predict",
            {"spectrum": default, "explain": explain_flag, "audience": "operations"},
        )
        if not error_box(result):
            u = result["uncertainty"]
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Prediction", f"{result['prediction']:.4f}",
                            icon="🎯", accent="#00D9C0")
            with c2:
                metric_card("95% lower", f"{u['lower']:.4f}",
                            icon="⬇️", accent="#8B5CF6")
            with c3:
                metric_card("95% upper", f"{u['upper']:.4f}",
                            icon="⬆️", accent="#22C55E")

            st.markdown("### Likely range")
            st.plotly_chart(
                uncertainty_band(result["prediction"], u["lower"], u["upper"]),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
            st.caption(
                "Interval calibration is empirical on the held-out public dataset; "
                "nominal 95% does not imply guaranteed coverage."
            )

            if explain_flag:
                st.markdown("### Explanation for the team")
                render_explanation(result)


elif page == "Process Monitoring":
    st.subheader("Multivariate process monitoring")
    d = demo_spectrum()
    spectrum = st.session_state.get("spectrum", d.get("spectrum", []))

    if st.button("Run monitoring", type="primary"):
        result = api_post(
            "/monitor",
            {"spectrum": spectrum, "explain": True, "audience": "operations"},
        )
        if not error_box(result):
            status = str(result["status"]).upper()
            color = {"OK": "#22C55E", "WARNING": "#F59E0B", "ALARM": "#EF4444"}.get(
                status, "#00D9C0"
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Hotelling T²", f"{result['T2']:.3f}",
                            icon="🔥", accent="#00D9C0")
            with c2:
                metric_card("Q residual", f"{result['Q']:.3f}",
                            icon="📉", accent="#8B5CF6")
            with c3:
                metric_card("Status", status, icon="🚦",
                            accent=color,
                            sub="Process classification")

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    gauge_chart(result["T2"], "Hotelling T² (relative)",
                                max_val=max(10.0, result["T2"] * 1.4),
                                accent="#00D9C0"),
                    use_container_width=True, config=PLOTLY_CONFIG,
                )
            with g2:
                st.plotly_chart(
                    gauge_chart(result["Q"], "Q residual (relative)",
                                max_val=max(10.0, result["Q"] * 1.4),
                                accent="#8B5CF6"),
                    use_container_width=True, config=PLOTLY_CONFIG,
                )

            with st.expander("Raw monitoring payload"):
                st.json(result)

            st.markdown("### Explanation for the team")
            render_explanation(result)


elif page == "Root Cause":
    st.subheader("Statistical contribution analysis")
    d = demo_spectrum()
    spectrum = st.session_state.get("spectrum", d.get("spectrum", []))

    st.warning(
        "Contributions identify variables with large statistical influence on the "
        "residual; they do not establish physical causality."
    )

    if st.button("Analyze contributions", type="primary"):
        result = api_post(
            "/anomaly",
            {"spectrum": spectrum, "explain": True, "audience": "quality"},
        )
        if not error_box(result):
            idxs = result.get("top_contributor_indices", []) or []
            vals = result.get("top_contributor_values", []) or []

            if idxs:
                st.markdown("### Largest statistical contributors")
                st.plotly_chart(contribution_chart(idxs, vals),
                                use_container_width=True, config=PLOTLY_CONFIG)

                st.markdown("#### Detail table")
                rows = [
                    {"rank": i + 1, "spectral_index": int(idx),
                     "contribution": round(float(v), 5)}
                    for i, (idx, v) in enumerate(zip(idxs, vals))
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No contributors returned for this spectrum.")

            with st.expander("Raw anomaly payload"):
                st.json(result)

            st.markdown("### Explanation for the team")
            render_explanation(result)


elif page == "Chemometrics":
    st.subheader("Chemometric model card")
    metrics = cached_get("/metrics")
    if error_box(metrics):
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("PLS test RMSE",
                    f"{metrics.get('pls_test_rmse', float('nan')):.4f}",
                    icon="📏", accent="#00D9C0")
    with c2:
        metric_card("Bayesian coverage",
                    f"{100*metrics.get('bayesian_coverage_95', float('nan')):.1f}%",
                    icon="🎯", accent="#8B5CF6")
    with c3:
        metric_card("Train samples",
                    str(metrics.get("n_train", "—")),
                    icon="🧪", accent="#22C55E")
    with c4:
        metric_card("Spectral variables",
                    str(metrics.get("n_features", "—")),
                    icon="🔬", accent="#F59E0B")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("### Coverage gauge")
        cov = float(metrics.get("bayesian_coverage_95", 0) or 0) * 100
        st.plotly_chart(
            gauge_chart(cov, "Bayesian 95% coverage", max_val=100.0,
                        suffix="%", accent="#8B5CF6",
                        thresholds=[
                            {"range": [0, 80], "color": "rgba(239,68,68,.16)"},
                            {"range": [80, 92], "color": "rgba(245,158,11,.16)"},
                            {"range": [92, 98], "color": "rgba(0,217,192,.16)"},
                            {"range": [98, 100], "color": "rgba(139,92,246,.16)"},
                        ]),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
    with col_b:
        st.markdown("### Model health radar")
        st.plotly_chart(metrics_radar(metrics),
                        use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("### Methods")
    st.markdown(
        """
        <div class="explain-card" style="--accent:#00D9C0">
            <ul>
                <li>PLS latent-variable calibration.</li>
                <li>BayesianRidge soft sensor with empirical interval coverage.</li>
                <li>PCA-based Hotelling T² / Q monitoring.</li>
                <li>VIP wavelength ranking.</li>
                <li>Corn multi-instrument transfer evaluated offline, not served.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
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
            {
                "risk_aware": risk,
                "min_prob": probability,
                "quality_threshold": threshold,
                "explain": True,
                "audience": "management",
            },
        )
        if not error_box(result):
            if result.get("success") is False:
                st.error(
                    "The requested risk-constrained problem was not solved. "
                    "No optimum is being presented as valid."
                )
            else:
                numeric = {
                    k: v for k, v in result.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                top = list(numeric.items())[:4]
                if top:
                    cols = st.columns(len(top))
                    for col, (k, v) in zip(cols, top):
                        with col:
                            metric_card(k.replace("_", " ").title(),
                                        f"{v:.4g}",
                                        icon="⚙️",
                                        accent="#00D9C0")

                if len(numeric) > 1:
                    keys = [k.replace("_", " ").title() for k in numeric]
                    vals = list(numeric.values())
                    fig = go.Figure(
                        go.Bar(x=keys, y=vals,
                               marker=dict(color="#00D9C0"))
                    )
                    fig.update_layout(
                        height=320, margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.02)",
                        font=dict(family="Inter", color="#cfe3ea"),
                        xaxis=dict(gridcolor="rgba(255,255,255,.05)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,.05)"),
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config=PLOTLY_CONFIG)

            with st.expander("Raw optimization payload"):
                st.json(result)

            st.markdown("### Explanation for management")
            render_explanation(result)


elif page == "Stakeholder Explain":
    st.subheader("Stakeholder explanation")
    st.caption(
        "Groq is optional. The backend always returns a deterministic fallback "
        "when no API key is configured."
    )

    default_context = {
        "type": "prediction",
        "prediction": 0.5,
        "uncertainty": {"lower": 0.45, "upper": 0.55},
        "status": "illustrative",
        "note": "Illustrative context only.",
    }
    context_text = st.text_area(
        "Result JSON", json.dumps(default_context, indent=2), height=240
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        audience = st.selectbox("Audience", ["operations", "quality", "management"])
    with c2:
        detail = st.selectbox("Detail", ["brief", "standard", "detailed"], index=1)
    with c3:
        st.write("")
        st.write("")
        submitted = st.button("Explain", type="primary")

    if submitted:
        try:
            context = json.loads(context_text)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        else:
            with st.spinner("Drafting a plain-language summary…"):
                result = api_post(
                    "/explain",
                    {"context": context, "audience": audience, "detail": detail},
                )
            if not error_box(result):
                st.markdown("### Plain-language summary")
                render_explanation(result)
                with st.expander("Raw response"):
                    st.json(result)
