"""Mechanistic CSTR / first-order reaction simulation for ChemX.

EXPLICITLY SYNTHETIC — not industrial or Shell data.
Model: dC_A/dt = -k(T) C_A , k = A exp(-E_a / RT)
Extended with conversion, yield proxy, energy, sensor noise.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# Physical constants
R = 8.314  # J/(mol·K)


def arrhenius(T: np.ndarray, A: float = 1e6, Ea: float = 5e4) -> np.ndarray:
    """k = A exp(-Ea / RT). T must be strictly positive Kelvin."""
    T = np.asarray(T, dtype=float)
    if not np.isfinite(T).all() or np.any(T <= 0):
        raise ValueError("Temperature must contain finite values greater than 0 K.")
    if not np.isfinite(A) or A <= 0 or not np.isfinite(Ea) or Ea < 0:
        raise ValueError("Arrhenius parameters must be finite, A > 0 and Ea >= 0.")
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
    if not np.isfinite(C_A0) or C_A0 <= 0:
        raise ValueError("C_A0 must be finite and greater than 0.")
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and greater than 0 K.")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative.")

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
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative.")

    rng = np.random.RandomState(seed)

    # Operating ranges
    T = rng.uniform(320, 400, n_samples)          # K
    tau = rng.uniform(0.5, 5.0, n_samples)        # residence time
    C_A0 = rng.uniform(0.8, 1.2, n_samples)       # feed concentration

    conversion = np.array([
        cstr_steady_state(c, t, tau_i)["conversion"]
        for c, t, tau_i in zip(C_A0, T, tau)
    ])
    yield_ = conversion.copy()
    energy = np.array([
        cstr_steady_state(c, t, tau_i)["energy"]
        for c, t, tau_i in zip(C_A0, T, tau)
    ])

    # Inject a few anomalies (extreme T or tau) and recompute all dependent
    # outputs before generating observations and synthetic spectra.
    anomaly = np.zeros(n_samples, dtype=int)
    if include_anomalies:
        n_anomalies = min(n_samples, max(3, n_samples // 20))
        idx = rng.choice(n_samples, size=n_anomalies, replace=False)
        T[idx] = rng.uniform(450, 500, len(idx))
        anomaly[idx] = 1
        for i in idx:
            r = cstr_steady_state(C_A0[i], T[i], tau[i])
            conversion[i] = r["conversion"]
            yield_[i] = r["yield"]
            energy[i] = r["energy"]

    y_obs = np.clip(conversion + rng.normal(0, noise_level, n_samples), 0, 1)

    # Synthetic spectra are derived from the final process state, including anomalies.
    n_wl = 50
    base = np.linspace(0, 1, n_wl)
    spectra = (
        conversion[:, None] * base[None, :]
        + (1 - conversion[:, None]) * (1 - base[None, :])
        + rng.normal(0, noise_level, (n_samples, n_wl))
    )

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
