# ChemX — Physics-Informed Chemometric Process Intelligence

**Spectroscopy → Chemometrics → Uncertainty → Process Monitoring → Physics-Informed Decision Support**

ChemX is a research-grade portfolio platform for computational chemometrics and digital chemistry. It combines classical multivariate calibration with uncertainty quantification, multivariate statistical process monitoring, calibration-transfer experiments, and a clearly separated mechanistic simulation layer.

> **Scientific boundary:** MLNIRdata, the Eigenvector Corn benchmark, and the UCPH Sugar Process dataset are public experimental data. The CSTR/Arrhenius physics and optimization layer is **MECHANISTIC_SIMULATION** only. ChemX does not contain or claim Shell, refinery, LNG, or proprietary plant data.

## Architecture

```text
PUBLIC RESEARCH DATA → offline audit / experiments → versioned artifacts
                                         ↓
                              Render FastAPI model service
                                         ↓ HTTPS JSON
                              Streamlit control tower

MECHANISTIC_SIMULATION → CSTR / Arrhenius → physics residuals → optimization
```

The deployed runtime **does not require data/raw/**. The backend loads a versioned MLNIR model bundle from models/; the Streamlit frontend communicates with the backend over HTTPS.

## Scientific workflow

1. Data provenance and validation
2. Spectral preprocessing: raw, SNV, MSC, Savitzky–Golay
3. PCA and multivariate diagnostics
4. PLS / PCR / Ridge / Elastic Net comparison
5. VIP wavelength ranking
6. BayesianRidge soft sensor with empirical interval coverage
7. Hotelling T² / Q residual monitoring
8. Statistical contribution analysis
9. Corn instrument-transfer benchmark
10. Sugar temporal/process monitoring
11. CSTR physics residual demonstration
12. Deterministic and risk-aware optimization
13. FastAPI model serving
14. Streamlit stakeholder dashboard
15. Docker + CI + deployment configuration

## Public data provenance

| Dataset | Role | Runtime |
|---|---|---|
| MLNIRdata — https://doi.org/10.5281/zenodo.16783068 | Hydrocarbon NIR density calibration and UQ | Model artifact only |
| Eigenvector Corn — https://eigenvector.com/resources/data-sets/nir-of-corn-samples-for-standardization-benchmarking/ | Multi-instrument calibration transfer | Offline experiments |
| UCPH Sugar Process — https://ucphchemometrics.com/sugar-process-data/ | Fluorescence/process monitoring | Offline experiments |
| CSTR / Arrhenius | Physics-informed optimization | MECHANISTIC_SIMULATION |

The verified MLNIRdata Zenodo record is DOI 10.5281/zenodo.16783068; the record describes the underlying dataset DOI as 10.5281/zenodo.16781222. citeturn1search0

Raw datasets are intentionally excluded from Git. See data/README.md and data/data_dictionary.md.

## Current experiment snapshot

The checked-in reports/results.json is an experiment snapshot, not a claim of industrial performance.

- MLNIR: 156/52 train/test split; PLS test RMSE ≈ 0.0349 and R² ≈ 0.9787.
- Bayesian soft sensor: empirical 95% interval coverage ≈ 90.4% on the held-out split.
- Corn moisture transfer, m5 → mp5: raw cross-instrument RMSE ≈ 1.816; direct standardization RMSE ≈ 0.252.
- Sugar monitoring: the current simple NOC/reference construction flags a very high fraction of the hold-out samples. This is treated as evidence of distribution shift / an inadequate reference definition, **not** as a plant alarm-rate claim.
- The current risk-aware CSTR optimization run can be infeasible for the requested probability/quality constraint. ChemX deliberately reports solver failure rather than presenting an invalid optimum.

These numbers are public-dataset benchmark observations and can change when preprocessing, split strategy, or reference-window methodology changes.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Liveness |
| GET | /ready | Model readiness |
| GET | /model-info | Model/provenance metadata |
| GET | /metrics | Validation snapshot |
| GET | /demo-spectrum | Compact derived spectrum for UI demo |
| POST | /predict | Density + uncertainty |
| POST | /monitor | T² / Q / anomaly |
| POST | /anomaly | Statistical contributions |
| POST | /optimize | MECHANISTIC_SIMULATION optimization |
| POST | /explain | Groq narrative or deterministic fallback |
| POST | /admin/retrain | Local-only retraining; blocked in production |

Interactive OpenAPI documentation is available at /docs.

## Local development

### Backend

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn src.api.main:app --reload --port 8000
```

### Frontend

```bash
pip install -r src/dashboard/requirements.txt
```

PowerShell:

```powershell
$env:CHEMX_API_URL='http://localhost:8000'
streamlit run src/dashboard/app.py
```

The frontend can run without the raw datasets.

### Offline science / retraining

Install the full development environment from requirements.txt, download the public datasets into data/raw/, validate them, then run:

```bash
PYTHONPATH=. pytest tests/ -v
python scripts/run_experiments.py
```

Runtime retraining is intentionally disabled in production.

## Render backend

The repository includes render.yaml and a production Dockerfile.

- Runtime: Docker
- Health check: /health
- Port: Render-provided $PORT
- Environment: CHEMX_ENV=production
- Runtime training: disabled
- Secrets: GROQ_API_KEY and CHEMX_CORS_ORIGINS are supplied through Render; the default narrative model is `openai/gpt-oss-120b`.

Render supports Docker-based services and HTTP health checks; the Blueprint keeps this configuration version-controlled. citeturn0search4turn2search1turn2search0

After deployment, record:

```text
https://<your-service>.onrender.com
```

Then set CHEMX_CORS_ORIGINS to the exact Streamlit frontend origin.

## Streamlit frontend

Deploy src/dashboard/app.py.

The dependency file beside the entrypoint is src/dashboard/requirements.txt.

Set the Streamlit secret:

```toml
CHEMX_API_URL = 'https://<your-render-service>.onrender.com'
```

Streamlit supports repository-local dependency files and secrets outside source control. citeturn3search0turn0search3

## Environment variables

Production-important variables:

```text
CHEMX_ENV=production
CHEMX_CORS_ORIGINS=https://<your-streamlit-app>.streamlit.app
CHEMX_ALLOW_RUNTIME_TRAINING=false
GROQ_API_KEY=<optional>
GROQ_MODEL=openai/gpt-oss-120b
```

Never commit .env, Streamlit secrets, or provider credentials.

## Model governance

docs/model_card.md documents model purpose, provenance, validation methodology, uncertainty interpretation, and limitations.

The application intentionally uses language such as **largest statistical contributors** rather than causal claims.

## Engineering controls

- Environment-driven configuration
- Explicit CORS allowlist
- Request IDs and response timing headers
- Production retraining guard
- Model bundle validation at startup
- Docker health check
- Render HTTP health check
- Separate backend/frontend dependency sets
- Secrets outside Git
- Raw-data exclusion
- Fixed random seeds for reproducible offline experiments
- Public-data provenance and checksums
- Deterministic explanation fallback when Groq is unavailable

## Limitations

ChemX is a research/portfolio platform, not a plant-certified control system.

- Public datasets are small relative to industrial deployments.
- Calibration-transfer results depend on the experimental split and paired samples.
- T²/Q thresholds are reference-set dependent.
- Statistical contributions are not causal explanations.
- Bayesian intervals require empirical calibration checks.
- Mechanistic optimization is a simplified CSTR demonstration.
- No production process, safety, yield, energy, or financial improvement is claimed.

## Honest resume framing

- Built a research-grade chemometric process-intelligence platform using public NIR and process datasets, combining PLS/PCA, Bayesian uncertainty, MSPC and calibration transfer.
- Quantified cross-instrument NIR domain shift and evaluated direct standardization on the Eigenvector Corn benchmark.
- Implemented uncertainty-aware density soft sensing with empirical interval-coverage evaluation.
- Developed a clearly separated Arrhenius/CSTR physics simulation with constrained and Monte-Carlo risk-aware optimization.
- Deployed the model-serving layer as FastAPI/Docker and built an API-driven Streamlit control tower with CI, health checks, provenance, and secret management.

## License

Code: MIT. Dataset rights and attribution remain subject to each original source's terms.