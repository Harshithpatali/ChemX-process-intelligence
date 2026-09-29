"""ChemX Control Tower — redesigned Streamlit UI (same backend API as before)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.utils.io import load_mlnir, train_test_split_indices
from src.utils.model_registry import load_bundle
from src.preprocessing.spectral import SNV
from src.optimization.process_opt import optimize_deterministic, optimize_risk_aware
from src.physics_informed.reactor import simulate_campaign, cstr_steady_state
from src.explain.stakeholder import StakeholderExplainer

settings = get_settings()
st.set_page_config(page_title="ChemX Control Tower", page_icon="⚗️", layout="wide")

# ── Design tokens ───────────────────────────────────────────────────────────
INK, PANEL, CANVAS, LINE = "#10232F", "#FFFFFF", "#EEF2F5", "#D8E0E6"
TEAL, AMBER, RED, MUTED = "#0E8A87", "#D98A1F", "#C8432F", "#5B6F7D"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'Manrope',system-ui,sans-serif; }}
.stApp {{ background:{CANVAS}; color:{INK}; }}
.block-container {{ padding:1.6rem 2.4rem 4rem; max-width:1320px; }}
header[data-testid="stHeader"] {{ background:transparent; }}
h1,h2,h3 {{ letter-spacing:-0.02em; color:{INK}; font-weight:800; }}
h2 {{ font-size:1.5rem !important; margin-top:.2rem; }}
[data-testid="stSidebar"] {{ background:#0D1B25; border-right:1px solid #16303E; }}
[data-testid="stSidebar"] * {{ color:#8FA6B4; }}
[data-testid="stSidebar"] h3 {{ color:#E6EEF2 !important; }}
[data-testid="stSidebar"] hr {{ border-color:#16303E; }}
[data-testid="stSidebar"] .stRadio label {{ padding:.35rem .7rem; border-radius:8px; width:100%; font-size:.9rem;
  border-left:3px solid transparent; }}
[data-testid="stSidebar"] .stRadio label:hover {{ background:rgba(255,255,255,.05); }}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {{ background:rgba(14,138,135,.16); border-left-color:{TEAL}; }}
[data-testid="stSidebar"] .stRadio label:has(input:checked) * {{ color:#E6EEF2; font-weight:700; }}
[data-testid="stSidebar"] .stRadio label > div:first-child {{ display:none; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background:#152A38 !important; border:1px solid #1F3B4C; }}
[data-testid="stSidebar"] [data-baseweb="select"] * {{ color:#DCE7ED !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill:#8FA6B4; }}
.bottom {{ background:{PANEL}; border:1px solid {LINE}; border-left:5px solid {TEAL}; border-radius:12px;
  padding:1rem 1.3rem; font-size:1.15rem; font-weight:700; line-height:1.4; margin:.4rem 0 1rem; }}
.hero {{ background:{INK}; border-radius:16px; padding:1.6rem 1.9rem; color:#fff; display:flex;
  justify-content:space-between; align-items:center; gap:2rem; margin-bottom:1.4rem; }}
.hero h1 {{ color:#fff; font-size:1.9rem; margin:0 0 .3rem; }}
.hero p {{ color:#9FB4C1; margin:0; font-size:.92rem; max-width:640px; line-height:1.5; }}
.verdict {{ text-align:right; min-width:190px; }}
.verdict b {{ display:block; font-size:1.7rem; font-weight:800; }}
.verdict span {{ color:#9FB4C1; font-size:.82rem; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }}
.kpi {{ background:{PANEL}; border:1px solid {LINE}; border-left:5px solid var(--c,{TEAL});
  border-radius:12px; padding:1rem 1.2rem; height:100%; }}
.kpi small {{ color:{MUTED}; font-weight:600; font-size:.82rem; }}
.kpi div {{ font-size:1.9rem; font-weight:800; font-variant-numeric:tabular-nums; line-height:1.25; }}
.kpi em {{ font-style:normal; color:{MUTED}; font-size:.78rem; }}
.note {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:.85rem 1.1rem;
  color:{MUTED}; font-size:.88rem; margin:1rem 0; }}
.badge {{ display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.76rem; font-weight:700;
  background:#FBEBD0; color:#8A5610; }}
.pipe {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.9rem; }}
.pipe span {{ background:rgba(255,255,255,.1); color:#CFE0E8; border-radius:999px; padding:.2rem .7rem; font-size:.76rem; }}
.stButton > button {{ background:{TEAL}; color:#fff; border:none; border-radius:10px; font-weight:700;
  padding:.55rem 1.2rem; }}
.stButton > button:hover {{ background:#0B706E; color:#fff; }}
.stButton > button:focus-visible {{ outline:3px solid {AMBER}; }}
[data-testid="stTabs"] button[role="tab"] {{ font-weight:700; }}
.stPlotlyChart, [data-testid="stImage"] img {{ border-radius:12px; }}
@media (max-width:800px) {{ .hero {{ flex-direction:column; align-items:flex-start; }} .verdict {{ text-align:left; }}
  .block-container {{ padding:1rem; }} }}
</style>
""",
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif", "axes.facecolor": PANEL, "figure.facecolor": PANEL,
        "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#EDF1F4",
        "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
        "axes.prop_cycle": plt.cycler(color=[TEAL, AMBER, INK, RED]),
    }
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def kpi(col, label: str, value: str, hint: str = "", color: str = TEAL):
    col.markdown(
        f'<div class="kpi" style="--c:{color}"><small>{label}</small><div>{value}</div><em>{hint}</em></div>',
        unsafe_allow_html=True,
    )


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def show(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def synthetic_banner():
    st.markdown(
        '<span class="badge">Simulated data</span> &nbsp;<span style="color:#5B6F7D;font-size:.88rem">'
        "Mechanistic simulation only. Not industrial plant data.</span>",
        unsafe_allow_html=True,
    )
    st.write("")


def render_result(res: dict):
    """Scalars as KPI cards, everything else in an expander."""
    scalars = {k: v for k, v in res.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if scalars:
        cols = st.columns(min(4, len(scalars)))
        for i, (k, v) in enumerate(scalars.items()):
            kpi(cols[i % len(cols)], k.replace("_", " ").capitalize(), f"{v:,.4g}")
            if i % 4 == 3 and i < len(scalars) - 1:
                cols = st.columns(min(4, len(scalars) - i - 1))
    with st.expander("Full result (JSON)"):
        st.json(res)
    st.download_button("Download JSON", json.dumps(res, indent=2, default=str), "chemx_result.json")


@st.cache_resource(show_spinner="Loading models and data…")
def load_all():
    data = load_mlnir()
    bundle, metrics = load_bundle()
    explainer = StakeholderExplainer()
    X = data["X"]
    mu = bundle["mu"]
    mon_all = bundle["monitor"].monitor(X - mu)  # computed once, reused across pages
    return data, bundle, metrics, explainer, mon_all


data, bundle, metrics, explainer, mon_all = load_all()
X, y = data["X"], data["y"]
mu = bundle["mu"]
tr, te = train_test_split_indices(len(y), settings.test_size, settings.random_seed)

# ── Sidebar ─────────────────────────────────────────────────────────────────
PAGES = [
    "Overview", "Spectroscopy", "Chemometrics", "Process monitoring", "Soft sensor",
    "Root cause", "Physics-informed AI", "Optimization", "Stakeholder explain",
]
with st.sidebar:
    st.markdown("### ⚗️ ChemX")
    st.caption("Process intelligence control tower")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    audience = st.selectbox("Explain for", ["operations", "quality", "management"])
    detail = st.select_slider("Detail", ["brief", "standard", "detailed"], value="standard")
    st.divider()
    ok = explainer.available
    st.markdown(
        f'<span class="dot" style="background:{"#3CCB8A" if ok else AMBER}"></span>'
        f'Groq {"connected" if ok else "not configured"}',
        unsafe_allow_html=True,
    )
    st.caption(f"Environment: {settings.env}")
    st.caption(f"{metrics.get('n_features')} features · PLS test RMSE {metrics.get('pls_test_rmse', 0):.4f}")
    if not ok:
        st.caption("Add GROQ_API_KEY to .env to enable narratives.")
    st.caption("Public research data only.")

# ── Hero ────────────────────────────────────────────────────────────────────
anom_rate = float(mon_all["anomaly"][te].mean()) if len(mon_all["anomaly"]) == len(y) else float(mon_all["anomaly"].mean())
state, colour = (
    ("Nominal", "#3CCB8A") if anom_rate < 0.05 else ("Watch", AMBER) if anom_rate < 0.15 else ("Alarm", "#FF6B57")
)
st.markdown(
    f"""
<div class="hero"><div>
<h1>Process intelligence control tower</h1>
<p>From raw spectra to a decision someone can act on: monitoring, calibrated soft sensing,
physics-informed models and plain-language explanations.</p>
<div class="pipe"><span>Spectroscopy</span><span>Chemometrics</span><span>Monitoring</span>
<span>Bayesian soft sensor</span><span>Physics-informed ML</span><span>Optimization</span><span>Explanations</span></div>
</div>
<div class="verdict"><b><span class="dot" style="background:{colour}"></span>{state}</b>
<span>{anom_rate:.1%} of samples flagged</span></div></div>
""",
    unsafe_allow_html=True,
)


def show_explanation(payload: dict, key: str):
    """Runs Groq once per click and keeps the result across reruns."""
    if not explainer.available:
        st.warning("Narratives are off. Add GROQ_API_KEY to `.env` and restart the app.")
        st.code(explainer._fallback(payload))
        return
    with st.spinner("Writing explanation…"):
        result = explainer.explain(payload, audience=audience, detail=detail)
    st.session_state[key] = result


SECTIONS = [
    ("What we observed", "🔎"), ("What it means", "💡"),
    ("Recommended next checks", "✅"), ("Confidence notes", "🎯"),
]


def parse_sections(text: str):
    bottom, out, cur = "", {}, None
    for line in text.splitlines():
        t = line.strip().strip("#*: ").strip()
        low = t.lower()
        if low.startswith("bottom line"):
            bottom = t.split(":", 1)[1].strip(" *") if ":" in t else ""
            cur = None
            continue
        hit = next((n for n, _ in SECTIONS if low.startswith(n.lower()) and len(t) < len(n) + 3), None)
        if hit:
            cur = hit
            out[cur] = []
        elif cur:
            out[cur].append(line)
    return bottom, {k: "\n".join(v).strip() for k, v in out.items() if "".join(v).strip()}


def explanation_panel(key: str):
    result = st.session_state.get(key)
    if not result:
        return
    st.write("")
    if not result.get("ok"):
        st.error(result.get("message", "The explanation failed. Check your API key and connection, then try again."))
        if result.get("fallback"):
            st.code(result["fallback"])
        return
    bottom, secs = parse_sections(result["explanation"])
    if bottom:
        st.markdown(f'<div class="bottom">{bottom}</div>', unsafe_allow_html=True)
    if secs:
        cols = st.columns(2)
        for i, (name, icon) in enumerate(SECTIONS):
            if name in secs:
                with cols[i % 2].container(border=True):
                    st.markdown(f"**{icon} {name}**")
                    st.markdown(secs[name])
    else:
        with st.container(border=True):
            st.markdown(result["explanation"])
    st.caption(f"For {result.get('audience')} · {result.get('detail')} detail · {result.get('model')} · {result.get('latency_s')}s")
    st.download_button("Download as text", result["explanation"], f"chemx_{key}.txt", key=f"dl_{key}")


def facts(ctx: dict, prefix: str = ""):
    """Flatten a result into (label, value) pairs a human can read."""
    rows = []
    for k, v in ctx.items():
        if k in ("type", "status") and not prefix:
            continue
        label = (prefix + k).replace("_", " ").capitalize()
        if isinstance(v, dict):
            rows += facts(v, prefix=f"{k} ")
        elif isinstance(v, bool):
            rows.append((label, "Yes" if v else "No"))
        elif isinstance(v, (int, float)):
            rows.append((label, f"{v:,.4g}"))
        elif isinstance(v, str) and len(v) < 40:
            rows.append((label, v))
        elif isinstance(v, (list, tuple)) and len(v) <= 6:
            rows.append((label, ", ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in v)))
    return rows


# ── Pages ───────────────────────────────────────────────────────────────────
if page == "Overview":
    st.header("Process health")
    pred = bundle["pls"].predict(X[te] - mu)
    cov = bundle["bayes"].coverage(X[te] - mu, y[te])
    c = st.columns(4)
    kpi(c[0], "Mean predicted density", f"{pred.mean():.3f}", "PLS, test set")
    kpi(c[1], "Interval coverage", f"{cov:.0%}", "Bayesian, test set", TEAL if cov >= 0.9 else AMBER)
    kpi(c[2], "Anomaly rate", f"{anom_rate:.1%}", "T² / Q limits", colour)
    kpi(c[3], "Test samples", f"{len(te)}", f"of {len(y)} total", INK)
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.scatter(y[te], pred, s=18, alpha=0.75)
        lim = [min(y[te].min(), pred.min()), max(y[te].max(), pred.max())]
        ax.plot(lim, lim, color=AMBER, lw=1.2, ls="--")
        ax.set_xlabel("Measured density")
        ax.set_ylabel("Predicted density")
        ax.set_title("Predicted vs measured (test set)")
        show(fig)
    with right:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.hist(pred - y[te].reshape(pred.shape), bins=24, color=TEAL, alpha=0.85)
        ax.axvline(0, color=INK, lw=1)
        ax.set_xlabel("Prediction error")
        ax.set_title("Error distribution")
        show(fig)
    note("Data: MLNIRdata, Corn and Sugar process (public). Optimization uses mechanistic simulation only.")

elif page == "Spectroscopy":
    st.header("Near-infrared spectra")
    idx = st.slider("Sample", 0, len(y) - 1, 0)
    st.caption(f"Sample {idx} · density {y[idx]:.3f}")
    snv = SNV().fit_transform(X[idx : idx + 1])
    t1, t2, t3 = st.tabs(["Raw", "SNV-corrected", "Overlay"])
    with t1:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(data["axis"], X[idx], lw=1)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Absorbance (a.u.)")
        show(fig)
    with t2:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(data["axis"], snv[0], lw=1, color=AMBER)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Standardised absorbance")
        show(fig)
    with t3:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(data["axis"], X.T[:, ::max(1, len(y) // 40)], color=LINE, lw=0.6)
        ax.plot(data["axis"], X[idx], lw=1.6, color=TEAL)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Absorbance (a.u.)")
        show(fig)

elif page == "Chemometrics":
    st.header("PCA and PLS")
    scores = bundle["pca"].transform(X - mu)
    vip = bundle["pls"].vip_scores()
    a, b = st.columns([2, 3])
    with a:
        fig, ax = plt.subplots(figsize=(5, 4.4))
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=y, cmap="viridis", s=18)
        fig.colorbar(sc, ax=ax, label="Density")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Score plot")
        show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(data["axis"], vip, lw=0.9)
        ax.axhline(1, color=AMBER, ls="--", lw=1)
        ax.fill_between(data["axis"], vip, 1, where=vip > 1, color=TEAL, alpha=0.2)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("VIP")
        ax.set_title("Variables that drive the model (VIP > 1)")
        show(fig)

elif page == "Process monitoring":
    st.header("Multivariate monitoring")
    mon = mon_all
    c = st.columns(3)
    kpi(c[0], "Anomaly rate", f"{mon['anomaly'].mean():.1%}", "all samples", colour)
    kpi(c[1], "Flagged samples", f"{int(mon['anomaly'].sum())}", f"of {len(mon['anomaly'])}", AMBER)
    kpi(c[2], "Confidence limit", "95%", "T² and Q", INK)
    st.write("")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, key, lim, title in [
        (axes[0], "T2", "T2_limit", "Hotelling T²"), (axes[1], "Q", "Q_limit", "Q residuals"),
    ]:
        v = np.asarray(mon[key])
        lv = float(np.ravel(mon[lim])[0])
        ax.plot(v, lw=0.8)
        ax.axhline(lv, color=RED, ls="--", lw=1.1, label="95% limit")
        over = np.where(v > lv)[0]
        ax.scatter(over, v[over], color=RED, s=14, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.legend(frameon=False)
    show(fig)

elif page == "Soft sensor":
    st.header("Bayesian soft sensor")
    idx = st.slider("Test sample", 0, len(te) - 1, 0)
    i = te[idx]
    iv = bundle["bayes"].predict_interval(X[i : i + 1] - mu)
    pred, lo, hi, sd = (float(iv[k][0]) for k in ("prediction", "lower", "upper", "std"))
    inside = lo <= y[i] <= hi
    c = st.columns(3)
    kpi(c[0], "Predicted density", f"{pred:.4f}", f"± {sd:.4f} (1σ)")
    kpi(c[1], "95% credible interval", f"{lo:.3f} – {hi:.3f}", f"width {hi - lo:.4f}", INK)
    kpi(c[2], "Measured density", f"{y[i]:.4f}", "inside interval" if inside else "outside interval",
        TEAL if inside else RED)
    st.write("")
    fig, ax = plt.subplots(figsize=(10, 1.9))
    ax.hlines(0, lo, hi, color=TEAL, lw=10, alpha=0.35)
    ax.scatter([pred], [0], s=110, color=TEAL, zorder=3, label="Prediction")
    ax.scatter([y[i]], [0], s=110, marker="D", color=AMBER, zorder=3, label="Measured")
    ax.set_yticks([])
    ax.set_xlabel("Density")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    show(fig)
    payload = {
        "type": "prediction", "prediction": pred,
        "uncertainty": {"lower": lo, "upper": hi, "std": sd},
        "true_value": float(y[i]), "target": "normalized_density", "status": "ok",
    }
    st.session_state["last_pred"] = payload
    if st.button("Explain to stakeholders"):
        show_explanation(payload, "exp_soft")
    explanation_panel("exp_soft")

elif page == "Root cause":
    st.header("Root-cause analysis")
    anom_idx = np.where(mon_all["anomaly"])[0]
    if len(anom_idx) == 0:
        note("No samples are outside the current limits, so there is nothing to investigate.")
    else:
        pick = st.selectbox("Flagged sample", anom_idx.tolist())
        rc = bundle["monitor"].root_cause(X - mu, sample_idx=int(pick), top_k=10)
        rc["type"] = "root_cause"
        st.session_state["last_rc"] = rc
        if rc.get("message"):
            note(rc["message"])
        render_result(rc)
        if st.button("Explain anomaly to stakeholders"):
            show_explanation(rc, "exp_rc")
        explanation_panel("exp_rc")

elif page == "Physics-informed AI":
    st.header("Mechanistic reactor (CSTR)")
    synthetic_banner()
    sim = simulate_campaign(150, seed=0)
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        sc = ax.scatter(sim["T"], sim["conversion"], c=sim["anomaly"], cmap="coolwarm", s=18)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Conversion")
        ax.set_title("Simulated campaign (red = anomalous)")
        show(fig)
    with right:
        st.markdown("##### Try an operating point")
        T = st.slider("Temperature (K)", 320, 450, 360)
        tau = st.slider("Residence time", 0.5, 6.0, 2.0)
        r = cstr_steady_state(1.0, float(T), float(tau))
        if isinstance(r, dict):
            render_result(r)
        else:
            st.write(r)

elif page == "Optimization":
    st.header("Constrained optimization")
    synthetic_banner()
    mode = st.radio("Mode", ["Deterministic", "Risk-aware"], horizontal=True)
    st.caption("Risk-aware mode samples 80 Monte Carlo scenarios, so it takes longer.")
    if st.button("Run optimization"):
        with st.spinner("Optimizing…"):
            res = optimize_deterministic() if mode == "Deterministic" else optimize_risk_aware(n_mc=80)
        res["type"] = "optimization"
        res["note"] = "MECHANISTIC_SIMULATION, not industrial data"
        st.session_state["last_opt"] = res
        st.session_state.pop("exp_opt", None)
    if st.session_state.get("last_opt"):
        render_result(st.session_state["last_opt"])
        if st.button("Explain optimization to stakeholders"):
            show_explanation(st.session_state["last_opt"], "exp_opt")
        explanation_panel("exp_opt")

elif page == "Stakeholder explain":
    st.header("Explanation studio")
    st.write("Pick a result and get a short, plain-language summary for the audience chosen in the sidebar.")
    example = {
        "type": "prediction", "prediction": 0.82,
        "uncertainty": {"lower": 0.80, "upper": 0.84, "std": 0.01},
        "status": "ok", "target": "normalized_density",
    }
    sources = {"Example soft-sensor reading": example}
    for label, k in [("Latest soft-sensor reading", "last_pred"), ("Latest root-cause result", "last_rc"),
                     ("Latest optimization", "last_opt")]:
        if st.session_state.get(k):
            sources = {label: st.session_state[k], **sources}
    choice = st.radio("Result to explain", list(sources), horizontal=True)
    ctx = sources[choice]
    with st.expander("Advanced: edit the raw JSON"):
        raw = st.text_area("Result JSON", value=json.dumps(ctx, indent=2, default=str), height=220, key=f"json_{choice}")
        try:
            ctx = json.loads(raw)
        except json.JSONDecodeError as e:
            st.error(f"That isn't valid JSON (line {e.lineno}, column {e.colno}): {e.msg}")
            ctx = None
    if ctx:
        rows = facts(ctx)[:8]
        for start in range(0, len(rows), 4):
            cols = st.columns(4)
            for col, (lab, val) in zip(cols, rows[start : start + 4]):
                kpi(col, lab, val)
            st.write("")
        if st.button("Explain this result"):
            show_explanation(ctx, "exp_studio")
    explanation_panel("exp_studio")