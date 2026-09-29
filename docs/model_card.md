# Model Card — ChemX Soft Sensor (MLNIR Density)

## Model details
- **Type:** PLS regression + Bayesian Ridge soft sensor
- **Input:** NIR spectrum (2635 wavenumbers, cm⁻¹)
- **Output:** Normalized density ∈ [0, 1] + approximate 95% predictive interval
- **Training data:** MLNIRdata public hydrocarbon mixtures (n=208)
- **Not trained on:** any industrial/Shell/proprietary data

## Intended use
Demonstration of chemometric calibration and uncertainty quantification for portfolio and research purposes.

## Metrics (held-out test, seed=42)
- PLS / PCR RMSE ≈ 0.035, R² ≈ 0.98
- Bayesian approximate 95% predictive interval empirical coverage ≈ 0.90

## Limitations
- Small public dataset; not validated for process control decisions
- Predictive intervals use a Gaussian approximation and should be recalibrated on application-specific data
- Spectral regions flagged by VIP are statistical, not automatically chemical assignments

## Ethical / safety
Do not use for safety-critical or regulatory decisions without independent validation on application-specific data.
