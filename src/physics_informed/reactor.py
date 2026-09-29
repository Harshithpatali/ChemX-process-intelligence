"""Mechanistic CSTR / first-order reaction simulation for ChemX.

EXPLICITLY SYNTHETIC — not industrial or Shell data.
Model: dC_A/dt = -k(T) C_A , k = A exp(-E_a / RT)
Extended with conversion, yield proxy, energy, sensor noise.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# Physical constants
R = 8.314  # J/(mol·K)


def arrhenius(T: np.ndarray, A: float = 1e6, Ea: float = 5e4) -> np.ndarray:
    """k = A exp(-Ea / RT). T in Kelvin."""
    return A * np.exp(-Ea / (R * T))


def cstr_steady_state(
    C_A0: float,
    T: float,
    tau: float,
    A: float = 1e6,
    Ea: float = 5e4,
) -> Dict[str, float]:
    """Steady-state CSTR for irreversible first-order reaction.

    C_A = C_A0 / (1 + k tau)
    X = 1 - C_A/C_A0
    """
    k = float(arrhenius(np.array([T]), A, Ea)[0])
    C_A = C_A0 / (1.0 + k * tau)
    conversion = 1.0 - C_A / C_A0
    # Simple yield proxy (= conversion for single reaction)
    yield_ = conversion
    # Relative energy: heating to T from 300 K (arbitrary)
    energy = max(0.0, (T - 300.0) * 0.01)
    return {
        "C_A": C_A,
        "conversion": conversion,
        "yield": yield_,
        "k": k,
        "energy": energy,
        "T": T,
        "tau": tau,
        "C_A0": C_A0,
    }


def simulate_campaign(
    n_samples: int = 200,
    seed: int = 42,
    noise_level: float = 0.02,
    include_anomalies: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic campaign of operating points + soft measurements.

    Returns process inputs, true outputs, and noisy observations.
    Clearly labelled MECHANISTIC_SIMULATION.
    """
    rng = np.random.RandomState(seed)

    # Operating ranges
    T = rng.uniform(320, 400, n_samples)          # K
    tau = rng.uniform(0.5, 5.0, n_samples)        # residence time
    C_A0 = rng.uniform(0.8, 1.2, n_samples)       # feed concentration

    true = np.array([
        list(cstr_steady_state(c, t, tau_i).values())
        for c, t, tau_i in zip(C_A0, T, tau)
    ])
    # columns: C_A, conversion, yield, k, energy, T, tau, C_A0
    conversion = true[:, 1]
    yield_ = true[:, 2]
    energy = true[:, 4]

    # Noisy soft-sensor-like observations (e.g. spectral proxy of conversion)
    y_obs = conversion + rng.normal(0, noise_level, n_samples)
    y_obs = np.clip(y_obs, 0, 1)

    # Feature matrix: process variables (+ optional synthetic "spectral" channels)
    # Simple synthetic spectra: linear combination of conversion + noise
    n_wl = 50
    base = np.linspace(0, 1, n_wl)
    spectra = (
        conversion[:, None] * base[None, :]
        + (1 - conversion[:, None]) * (1 - base[None, :])
        + rng.normal(0, noise_level, (n_samples, n_wl))
    )

    # Inject a few anomalies (extreme T or tau)
    anomaly = np.zeros(n_samples, dtype=int)
    if include_anomalies:
        idx = rng.choice(n_samples, size=max(3, n_samples // 20), replace=False)
        T[idx] = rng.uniform(450, 500, len(idx))
        anomaly[idx] = 1
        # recompute those points
        for i in idx:
            r = cstr_steady_state(C_A0[i], T[i], tau[i])
            conversion[i] = r["conversion"]
            yield_[i] = r["yield"]
            energy[i] = r["energy"]
            y_obs[i] = np.clip(conversion[i] + rng.normal(0, noise_level), 0, 1)

    return {
        "label": "MECHANISTIC_SIMULATION",
        "T": T,
        "tau": tau,
        "C_A0": C_A0,
        "conversion": conversion,
        "yield": yield_,
        "energy": energy,
        "y_obs": y_obs,
        "spectra": spectra,
        "anomaly": anomaly,
        "X_process": np.column_stack([T, tau, C_A0]),
        "feature_names": ["T_K", "tau", "C_A0"],
    }


def physics_residual(
    T: np.ndarray,
    tau: np.ndarray,
    C_A0: np.ndarray,
    conversion_pred: np.ndarray,
    A: float = 1e6,
    Ea: float = 5e4,
) -> np.ndarray:
    """Physics residual: predicted conversion vs Arrhenius-CSTR conversion."""
    k = arrhenius(T, A, Ea)
    conversion_phys = 1.0 - 1.0 / (1.0 + k * tau)
    return conversion_pred - conversion_phys
