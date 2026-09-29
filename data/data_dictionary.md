# ChemX Data Dictionary

**Project:** ChemX — Physics-Informed Chemometric Process Intelligence Platform  
**Phase:** 1 — Data Acquisition and Audit (completed 2026-09-29)  
**Status:** ALL THREE real experimental datasets acquired, audited, and checksummed. Validation suite + 12 unit tests passing. Mechanistic simulation reserved for later phases and will be clearly labelled.

---

## 1. MLNIRdata (Primary Hydrocarbon NIR Dataset)

| Field | Value |
|-------|-------|
| **Dataset name** | MLNIRdata |
| **Source URL** | https://doi.org/10.5281/zenodo.16781223 |
| **Zenodo record** | 16781223 |
| **Authors** | Laurent Duval, Louna Alsouki, Jérémy Laxalde, Noémie Caillol |
| **Publication year** | 2025 (curated release); original data from Laxalde thesis |
| **License** | CC BY (Creative Commons Attribution) |
| **Local path** | `data/raw/mlnirdata/` |
| **Files acquired** | `MLNIR_matrixX_NirSpectrumData.csv`, `MLNIR_matrixX_NirSpectrumDataAxis.csv`, `MLNIR_matrixX_NirSpectrumDerivative.csv`, `MLNIR_matrixX_NirSpectrumDerivativeAxis.csv`, `MLNIR_matrixY_NirPropertyDensityNormalized.csv`, `MLNIRdata_matrixXY_NirSpectrum_DensityNormalized.mat`, README PDF, display script, figure |

### Dimensions & Variables

| Variable | Meaning | Unit | Type | Shape / Count | Notes |
|----------|---------|------|------|---------------|-------|
| Spectra (X) | NIR absorbance spectra of hydrocarbon mixtures | Arbitrary absorbance units (a.u.) | Continuous, high-dimensional | 2635 wavelengths × 208 samples | CSV is wavelengths × samples; transpose for conventional n×p |
| Wavenumber axis | Discrete wavenumber grid (increasing values; often plotted reversed) | cm⁻¹ | Continuous | 2635 | Range ≈ 3961.5 – 9041.6 cm⁻¹ |
| Derivative spectra | Savitzky-Golay first derivative | a.u. | Continuous | 2594 × 208 | Provided for convenience |
| Density (y) | Normalized macroscopic density of mixture | Normalized [0, 1] | Continuous scalar | 208 | Original density measured under identical conditions; released normalized |

### Quality Audit (performed)

- Missing values: **0**
- Duplicate spectra: **0**
- Wavelength ordering: monotonic (reversed cm⁻¹ as documented)
- Value range spectra: ≈ 0.0004 – 2.44
- Density range: [0.0, 1.0]
- Observations: independent samples (no temporal/batch grouping stated)
- Train/test leakage risk: low if random split used carefully; preferred approach = nested CV or repeated hold-out with fixed seeds
- Measurement limitations: laboratory NIR under controlled conditions; not on-line process spectra

### Mapping to ChemX components

Spectral preprocessing, PCA, PLS / sparse PLS, wavelength selection (VIP), Bayesian soft sensor for density, uncertainty quantification, basic anomaly analysis.

---

## 2. Corn NIR Multi-Instrument Dataset (Eigenvector)

| Field | Value |
|-------|-------|
| **Dataset name** | NIR of Corn Samples for Standardization Benchmarking |
| **Source URL** | https://eigenvector.com/resources/data-sets/nir-of-corn-samples-for-standardization-benchmarking/ |
| **Direct download** | https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip |
| **Original collection** | Cargill (redistributed by Eigenvector Research with permission) |
| **License** | Public redistribution for research / benchmarking (Eigenvector archive terms) |
| **Local path** | `data/raw/corn/corn.mat` |

### Dimensions & Variables

| Variable | Meaning | Unit | Type | Shape / Count | Notes |
|----------|---------|------|------|---------------|-------|
| m5spec | NIR spectra on instrument m5 (FOSS NIRSystems 5000) | Absorbance | Continuous | 80 × 700 | Wavelengths 1100–2498 nm, 2 nm steps |
| mp5spec | NIR spectra on instrument mp5 | Absorbance | Continuous | 80 × 700 | Same samples, different instrument |
| mp6spec | NIR spectra on instrument mp6 (FOSS NIRSystems 6000) | Absorbance | Continuous | 80 × 700 | Same samples, different instrument |
| Wavelength axis | Spectral axis | nm | Continuous | 700 | 1100, 1102, …, 2498 |
| Moisture | Laboratory reference moisture | % | Continuous | 80 | Mean ≈ 10.23, range ≈ 9.38–10.99 |
| Oil | Laboratory reference oil content | % | Continuous | 80 | Mean ≈ 3.50, range ≈ 3.09–3.83 |
| Protein | Laboratory reference protein | % | Continuous | 80 | Mean ≈ 8.67, range ≈ 7.65–9.71 |
| Starch | Laboratory reference starch | % | Continuous | 80 | Mean ≈ 64.70, range ≈ 62.83–66.47 |
| NBS glass standards | Reference glass spectra per instrument | Absorbance | Continuous | 3–4 × 700 | Useful for transfer diagnostics |

### Quality Audit (performed)

- Missing values (spectra & properties): **0**
- Duplicate spectra (m5): **0**
- Instrument differences clearly present (min/max ranges differ slightly across m5/mp5/mp6)
- Observations: same 80 physical samples measured on three instruments → perfect for calibration-transfer experiments
- Leakage risk: must never mix instruments in the same training fold without explicit transfer modelling; instrument-wise hold-out is mandatory for transfer evaluation

### Mapping to ChemX components

Instrument variability, domain shift, calibration transfer (DS, PDS, preprocessing, domain adaptation baselines), multi-property soft sensors, robustness testing.

---

## 3. Sugar Process Fluorescence + Process Data (University of Copenhagen)

| Field | Value |
|-------|-------|
| **Dataset name** | Sugar Process Data |
| **Source URL** | https://ucphchemometrics.com/sugar-process-data/ |
| **Download link** | https://sid.erda.dk/cgi-sid/ls.py?share_id=E0z0V0negi (ERDA share; interactive download required) |
| **Reference** | R. Bro, *Chemom. Intell. Lab. Syst.* 46 (1999) 133–147 |
| **Additional refs** | Bro PhD thesis (1998); Munck et al. (1998) |
| **License** | Public research use (UCPH Chemometrics standard terms: “as is”, report use as courtesy) |
| **Local path** | `data/raw/sugar/process/data.mat` |
| **SHA256** | `2ea507a1db477350411575be291300a990277d8f7ac4f2c32b6e8b633c16f711` |

### Dimensions & Variables (from official documentation)

| Variable | Meaning | Unit | Type | Shape / Count | Notes |
|----------|---------|------|------|---------------|-------|
| X (fluorescence) | Emission spectra at 7 excitation wavelengths | Fluorescence intensity | Continuous, multi-way | 268 × 571 × 7 (or unfolded 268 × 3997) | Emission 275–560 nm (0.5 nm steps); Excitations: 230, 240, 255, 290, 305, 325, 340 nm |
| EmAx | Emission wavelength labels | nm | Continuous | 571 | |
| ExAx | Excitation wavelength labels | nm | Continuous | 7 | |
| y | Quality parameters (color, ash, …) | Color (absorbance-derived units), Ash (%) | Continuous | 268 × 3 | Color at 420 nm (pH 7); ash by conductivity |
| Proc | Automatically sampled process variables (T, flow, pH, …) | Mixed | Continuous + missing | 268 × 39 × 7 (lags) | NaN for missing; 7 temporal lags |
| Lab | Auxiliary laboratory measurements | Mixed | Continuous + missing | 268 × 48 × 7 (lags) | NaN for missing |
| time | Sampling time code | Coded (month/day/shift) | Discrete | 268 | Enables temporal / regime analysis |
| Known upsets | Documented break-downs | Dates | Categorical | Listed on source page | Useful for supervised anomaly evaluation |

### Quality / Structure Notes (from source)

- N ≈ 268 shift-average samples over a 3-month campaign (autumn 1995, Scandinavian sugar plant).
- Temporal structure present → regime detection, drift, process monitoring natural.
- Missing values exist (explicitly coded as NaN in Proc and Lab).
- Three samples commonly discarded as extreme outliers in literature.
- Fluorescence measured after dissolution of final sugar product (centrifuge output).

### Mapping to ChemX components

Multivariate process monitoring (PCA / T² / Q), anomaly detection, root-cause contribution analysis, operating-regime detection, multi-way chemometrics (PARAFAC / MCR-ALS), soft sensing of quality from fluorescence + process variables.

### Quality Audit (performed 2026-09-29)

- Fluorescence X (unfolded 268 × 3997): **0 missing**, **0 duplicate rows**
- y (Date, Color, Ash*1000): **0 missing**; Color range 17–50; Ash*1000 range 8–33
- Proc (268 × 273): **26 717 NaNs** (documented lag structure, expected)
- Lab (268 × 336): **80 653 NaNs** (documented lag structure, expected)
- Emission axis: 275–560 nm (571 points); Excitation: [230, 240, 255, 290, 305, 325, 340] nm
- Temporal codes cover months 2–5 (Oct–Jan campaign)
- X maximum value ≈ 999.999 — possible saturation or instrument sentinel; inspect before modelling
- DimX confirms reshape target 268 × 571 × 7



**Acquisition status:** COMPLETE. Binary `data.mat` acquired via ERDA share redirect, extracted, and fully audited (see Quality Audit below).

---

## 4. Mechanistic Simulation Dataset (to be generated later)

| Field | Value |
|-------|-------|
| **Status** | Not yet generated (Phase 15+) |
| **Purpose** | Physics-informed ML, constrained / risk-aware optimization, DOE, active learning, controlled robustness stress tests only |
| **Labelling rule** | All files, notebooks, API fields and README sections will be prefixed `MECHANISTIC_` or `SYNTHETIC_`. Never mixed with real experimental folders. |
| **Model outline** | First-order kinetics \( dC_A/dt = -k(T)C_A \), Arrhenius \( k = A\exp(-E_a/RT) \), extended CSTR/batch variables (T, feed, residence time, conversion, yield, energy). Sensor noise, drift, missing values, regime changes, bias injected and documented. |

---

## Summary of Provenance & Scientific Honesty

| Dataset | Real experimental? | Domain | Primary use in ChemX |
|---------|--------------------|--------|----------------------|
| MLNIRdata | Yes | Hydrocarbon NIR + density | Core chemometrics, soft sensor, UQ |
| Corn multi-instrument | Yes | Agricultural NIR, 3 instruments | Calibration transfer, robustness |
| Sugar process | Yes | Industrial sugar refining (fluorescence + process) | Process monitoring, anomaly, regimes, multi-way |
| Mechanistic | No (simulated) | Simplified reaction engineering | Physics-informed ML & optimization only |

**No Shell, refinery, LNG or proprietary industrial data is used or claimed.**

---

## Next Actions After This Audit

1. Complete interactive download of sugar MATLAB files and re-run full numeric audit.
2. Implement `src/data_validation/` reusable functions + unit tests.
3. Freeze raw data (checksums) and proceed only after validation suite passes.

**Phase 1 data acquisition and metadata audit is complete for the two fully downloaded datasets; sugar metadata is verified pending binary acquisition.**


---

## Checksums (SHA256)

Recorded in `data/checksums.sha256`:

| File | SHA256 |
|------|--------|
| MLNIR spectra CSV | `c4cd4aab0423fb45388a309c1f736d1b56acc5e76d985438e36b44182cfe3e24` |
| MLNIR density CSV | `bb77ed81f5a41e745b10a6d92a29d02e8167856da19e0adea54ff108225d3691` |
| Corn corn.mat | `e28fd4be274a54ca57b1f2c67ef5a8bf4981f8314bcc73e4c64836fe658c46b5` |
| Sugar data.mat | `2ea507a1db477350411575be291300a990277d8f7ac4f2c32b6e8b633c16f711` |

## Validation Suite

- Location: `src/data_validation/`
- Unit tests: `tests/test_data_validation.py` — **12/12 passed**
- Entry points: `audit_mlnir`, `audit_corn`, `audit_sugar`
