# ChemX — Physics-Informed Chemometric Process Intelligence Platform

**Subtitle:** Spectroscopy → Chemometrics → Process Monitoring → Bayesian Soft Sensors → Physics-Informed ML → Uncertainty-Aware Optimization

Research-oriented portfolio platform demonstrating computational chemometrics, multivariate statistics, Bayesian uncertainty quantification, process monitoring, and physics-informed optimization capabilities relevant to a **Computational Chemometrician / Digital Chemistry** role.

> **Scientific honesty:** All experimental results use **public real datasets** (MLNIRdata, Eigenvector Corn, UCPH Sugar Process). Optimization and physics-informed sections use an explicitly labelled **MECHANISTIC_SIMULATION**. No Shell, refinery, LNG, or proprietary industrial data is used or claimed.

---

## Problem & Scientific Motivation

High-dimensional spectroscopic and process measurements must be transformed into reliable chemical-property predictions, latent process understanding, uncertainty estimates, anomaly alerts, and decision support under domain shift and physical constraints. ChemX connects:

**Mathematics → Statistics → Chemometrics → Machine Learning → Physics → Optimization → Software Engineering**

## Why Chemometrics?

Classical latent-variable models (PCA, PLS) remain the industry standard for interpretable multivariate calibration of NIR/fluorescence data. They are complemented—not replaced—by Bayesian soft sensors, calibration transfer, and physics-constrained optimization.

## Datasets (Provenance)

| Dataset | Source | License | Role in ChemX |
|---------|--------|---------|---------------|
| **MLNIRdata** | [Zenodo 10.5281/zenodo.16781223](https://doi.org/10.5281/zenodo.16781223) | CC BY | NIR spectra (208 × 2635) + density — core calibration, UQ, VIP |
| **Corn multi-instrument** | [Eigenvector](https://eigenvector.com/resources/data-sets/nir-of-corn-samples-for-standardization-benchmarking/) | Public research | 80 samples × 3 instruments — calibration transfer |
| **Sugar process** | [UCPH Chemometrics](https://ucphchemometrics.com/sugar-process-data/) (Bro 1999) | Public research | Fluorescence EEM + process vars — monitoring, regimes, root-cause |
| **Mechanistic CSTR** | Simulated in-repo | N/A | Physics-informed ML & constrained optimization only |

Full variable dictionary, units, checksums, and quality audit: [`data/data_dictionary.md`](data/data_dictionary.md).

## Methodology

- **Spectral preprocessing:** raw, SNV, MSC, Savitzky–Golay (1st/2nd derivative); selected by test RMSE.
- **PCA:** scores, loadings, explained variance, Hotelling’s T², Q residuals, contribution plots.
- **PLS / PCR / Ridge / Elastic Net:** nested comparison; VIP for wavelength importance.
- **Bayesian soft sensor:** Bayesian Ridge (+ optional PCA reduction) → mean + 95% credible interval; coverage reported.
- **Calibration transfer:** no-transfer vs SNV/MSC vs Direct Standardization across corn instruments.
- **Process monitoring:** PCA-MSPC on sugar fluorescence; GMM regime hints; statistical root-cause contributions.
- **Physics-informed demonstration:** first-order Arrhenius–CSTR; black-box NN vs physics residual.
- **Optimization:** deterministic SLSQP vs risk-aware (MC constraint on P(quality ≥ threshold)).

## Key Results (from `scripts/run_experiments.py`)

| Component | Baseline | Proposed | Metric (observed) |
|-----------|----------|----------|-------------------|
| Density prediction | PCR | PLS / Ridge | RMSE ≈ 0.035 (R² ≈ 0.98) |
| Uncertainty | Point estimate | Bayesian interval | Coverage ≈ 0.90 (95% nominal) |
| Instrument transfer (Moisture m5→mp5) | Raw model RMSE ≈ 1.82 | Direct Standardization | RMSE ≈ 0.25 |
| Process monitoring | — | PCA T²/Q | Anomaly flags on hold-out |
| Physics consistency | Black-box NN | Physics residual | Mean \|r\| ≈ 0.08 |
| Optimization | Deterministic yield | Risk-aware | Constraint probability reported |

Full JSON: [`reports/results.json`](reports/results.json).

## Repository Structure

```
ChemX/
├── data/           # raw (real), processed, mechanistic, dictionary, checksums
├── src/
│   ├── data_validation/
│   ├── preprocessing/
│   ├── chemometrics/   # PCA, PLS
│   ├── bayesian/
│   ├── monitoring/
│   ├── transfer/
│   ├── physics_informed/
│   ├── optimization/
│   ├── api/            # FastAPI
│   └── dashboard/      # Streamlit
├── scripts/run_experiments.py
├── tests/
├── reports/
├── Dockerfile
└── .github/workflows/ci.yml
```

## Quick Start

```bash
# Install (recommended: fresh venv)
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# If Streamlit fails with ImportError on starlette.middleware.gzip:
pip install -U "streamlit>=1.40" "starlette>=0.46"

# Run unit tests
set PYTHONPATH=.
pytest tests/ -v

# Run full experiment suite (writes reports/results.json)
python scripts/run_experiments.py

# API
uvicorn src.api.main:app --reload --port 8000
# GET /health  POST /predict  POST /monitor  POST /optimize  GET /metrics

# Dashboard
streamlit run src/dashboard/app.py

# Docker
docker build -t chemx .
docker run -p 8000:8000 chemx
```

### Troubleshooting: Streamlit `DEFAULT_EXCLUDED_CONTENT_TYPES` ImportError

This means your **Streamlit** and **Starlette** versions are mismatched (common on Windows when packages were installed at different times).

```powershell
# From the ChemX folder, with your venv active:
pip uninstall -y streamlit starlette
pip install "streamlit>=1.40,<1.65" "starlette>=0.46"
streamlit run src/dashboard/app.py
```

Alternatively use Docker (isolates dependencies completely):

```powershell
docker build -t chemx .
docker run -p 8501:8501 chemx streamlit run src/dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/model-info` | Soft-sensor metadata |
| GET | `/metrics` | Cached validation metrics |
| POST | `/predict` | Density + credible interval from spectrum |
| POST | `/monitor` | T² / Q / anomaly flag |
| POST | `/anomaly` | Contribution-based root-cause summary |
| POST | `/optimize` | Deterministic or risk-aware CSTR optimum (synthetic) |

## Limitations

- Dataset sizes are modest (typical of public chemometrics benchmarks); results illustrate methodology, not plant-scale performance.
- Bayesian intervals use a normal approximation from BayesianRidge; coverage is measured, not assumed perfect.
- Sugar process anomaly rate depends on the chosen reference window; high rates indicate distribution shift, which is informative.
- Risk-aware optimizer is a demonstration; industrial stochastic programming would use richer uncertainty models.
- No claim of improved industrial efficiency or Shell-specific results.

## Reproducibility

- Fixed random seeds (42) for splits and simulation.
- SHA256 checksums of raw data files in `data/checksums.sha256`.
- Deterministic experiment script; CI runs unit tests on push.

## Future Research Directions

- Sparse PLS / interval PLS for wavelength selection with chemical assignment references.
- Multi-way PARAFAC on sugar EEM with non-negativity constraints.
- Conformal prediction for distribution-free intervals.
- Active learning loop on the mechanistic surrogate.
- Model cards and continuous monitoring drift detection.

## Resume Bullets (honest)

- Built an end-to-end chemometrics platform (ChemX) on public NIR and process datasets: spectral preprocessing, PCA/PLS, VIP wavelength analysis, and density soft sensing (R² ≈ 0.98).
- Implemented Bayesian soft sensors with calibrated credible intervals (empirical coverage ≈ 90% at 95% nominal) and multivariate process monitoring (Hotelling’s T², Q residuals, contribution-based root-cause analysis).
- Quantified multi-instrument domain shift on the Eigenvector corn benchmark and reduced transfer RMSE via direct standardization (e.g. Moisture m5→mp5: 1.82 → 0.25).
- Developed a physics-informed CSTR demonstration (Arrhenius kinetics) comparing black-box neural predictors with physical residuals, plus deterministic and risk-aware constrained optimization.
- Deployed FastAPI prediction/monitoring/optimization endpoints and a Streamlit scientific dashboard; Dockerized with automated pytest CI.



## Industry-ready deployment notes

ChemX is structured for portfolio / pilot deployment patterns used in digital chemistry teams:

| Capability | Implementation |
|------------|----------------|
| Config via environment | `.env` / `.env.example`, `src/config.py` |
| Model persistence | `models/mlnir_bundle.joblib` via `model_registry` |
| API readiness | `GET /health`, `GET /ready` |
| Request tracing | `X-Request-ID`, `X-Response-Time-Ms` headers |
| CORS | `CHEMX_CORS_ORIGINS` |
| Structured logging | `src/utils/logging_setup.py` |
| Stakeholder narratives | Groq LLM via `POST /explain` and dashboard |
| Secrets | Never bake `GROQ_API_KEY` into Docker images |
| Retrain guard | `/admin/retrain` blocked when `CHEMX_ENV=production` |

**Disclaimer:** This remains a research portfolio platform on **public** datasets. Industry readiness here means engineering practices (config, health, artifacts, explanations), not certification for plant control.

### Groq stakeholder explanations

1. Copy `.env.example` → `.env`
2. Set `GROQ_API_KEY=gsk_...` (from https://console.groq.com)
3. Optional: `GROQ_MODEL=llama-3.3-70b-versatile`

**API**

```bash
# Predict + auto-explain for operations
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"spectrum": [...], "explain": true, "audience": "management"}'

# Standalone explanation of any result JSON
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"context": {"type": "monitoring", "anomaly": true, "T2": 12.5}, "audience": "operations"}'
```

**Dashboard:** pages Soft Sensor, Root Cause, Optimization, and **Stakeholder Explain** call Groq when the key is present. Without a key, a deterministic technical fallback is shown.

Docker with Groq:

```bash
docker run -p 8000:8000 -e GROQ_API_KEY=$GROQ_API_KEY chemx
```


## License

Code: MIT. Datasets retain their original licenses (CC BY for MLNIRdata; Eigenvector and UCPH terms for the others).

---

*ChemX is a research/engineering portfolio project. It demonstrates transferable computational capability for computational chemometrics and digital chemistry roles—without fabricating industrial experience or proprietary data.*
