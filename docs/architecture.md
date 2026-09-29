# ChemX Architecture

```
REAL CHEMICAL DATA (MLNIR, Corn, Sugar)
        │
        ▼
 DATA VALIDATION ──► SPECTRAL PREPROCESSING (SNV/MSC/SG)
        │
        ▼
 EXPLORATORY CHEMOMETRICS (PCA scores/loadings/T²/Q)
        │
        ├──────────────────────┐
        ▼                      ▼
 PLS / PCR / Ridge      BAYESIAN SOFT SENSOR
 VIP wavelength sel.    Credible intervals
        │                      │
        ▼                      ▼
 INSTRUMENT TRANSFER     UNCERTAINTY CALIBRATION
 (Corn multi-instrument)
        │
        ▼
 PROCESS MONITORING (Sugar) ──► ANOMALY ──► ROOT-CAUSE (contributions)
        │
        ▼
 OPERATING REGIME (GMM on PCA scores)
        │
        ▼
 ════════════════════════════════════════
 MECHANISTIC_SIMULATION (Arrhenius CSTR)
        │
        ├─ Black-box NN vs physics residual
        ├─ Deterministic constrained opt
        └─ Risk-aware opt (P(quality) constraint)
        │
        ▼
 FastAPI  (/predict /monitor /optimize /anomaly /explain)
 Streamlit Control Tower
 Docker + pytest CI
```

**Separation rule:** Real experimental artefacts live under `data/raw/`. Synthetic campaign data is generated at runtime and labelled `MECHANISTIC_SIMULATION` in every response and report.
