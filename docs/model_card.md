# Model Card — ChemX Soft Sensor (MLNIR Density)

## Model details
- **Type:** PLS regression + Bayesian Ridge soft sensor
- **Input:** NIR spectrum (2635 wavenumbers, cm⁻¹)
- **Output:** Normalized density ∈ [0, 1] + 95% credible interval
- **Training data:** MLNIRdata public hydrocarbon mixtures (n=208)
- **Not trained on:** any industrial/Shell/proprietary data

## Intended use
Demonstration of chemometric calibration and uncertainty quantification for portfolio and research purposes.

## Metrics (held-out test, seed=42)
- PLS / PCR RMSE ≈ 0.035, R² ≈ 0.98
- Bayesian 95% interval empirical coverage ≈ 0.90

## Limitations
- Small public dataset; not validated for process control decisions
- Interval calibration is approximate (normal assumption)
- Spectral regions flagged by VIP are statistical, not automatically chemical assignments

## Ethical / safety
Do not use for safety-critical or regulatory decisions without independent validation on application-specific data.
