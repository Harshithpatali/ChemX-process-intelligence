"""Constrained and uncertainty-aware process optimization.

Uses the mechanistic CSTR model as surrogate.
Deterministic vs risk-aware formulations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from src.physics_informed.reactor import cstr_steady_state


def objective_yield_energy(
    x: np.ndarray,
    w_yield: float = 1.0,
    w_energy: float = 0.1,
) -> float:
    """Maximize yield - energy penalty. x = [T, tau, C_A0]."""
    T, tau, C_A0 = x
    r = cstr_steady_state(C_A0, T, tau)
    return -(w_yield * r["yield"] - w_energy * r["energy"])


def optimize_deterministic(
    bounds: Optional[list] = None,
    w_yield: float = 1.0,
    w_energy: float = 0.1,
    x0: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Deterministic constrained optimization of yield vs energy."""
    if bounds is None:
        bounds = [(320.0, 420.0), (0.5, 6.0), (0.8, 1.2)]  # T, tau, C_A0
    if x0 is None:
        x0 = np.array([360.0, 2.0, 1.0])

    cons = [
        {"type": "ineq", "fun": lambda x: x[0] - 320},   # T >= 320
        {"type": "ineq", "fun": lambda x: 420 - x[0]},   # T <= 420
        {"type": "ineq", "fun": lambda x: x[1] - 0.5},
        {"type": "ineq", "fun": lambda x: 6.0 - x[1]},
    ]

    res = minimize(
        lambda x: objective_yield_energy(x, w_yield, w_energy),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    T, tau, C_A0 = res.x
    r = cstr_steady_state(C_A0, T, tau)
    return {
        "success": bool(res.success),
        "T": float(T),
        "tau": float(tau),
        "C_A0": float(C_A0),
        "yield": float(r["yield"]),
        "conversion": float(r["conversion"]),
        "energy": float(r["energy"]),
        "objective": float(-res.fun),
        "message": res.message,
        "method": "deterministic",
    }


def optimize_risk_aware(
    n_mc: int = 200,
    noise_std: float = 0.02,
    min_prob: float = 0.90,
    quality_threshold: float = 0.50,
    seed: int = 42,
    w_yield: float = 1.0,
    w_energy: float = 0.1,
) -> Dict[str, object]:
    """Risk-aware: maximize expected yield s.t. P(conversion >= threshold) >= min_prob.

    Uncertainty is modelled as Gaussian noise on the effective rate constant.
    """
    if n_mc < 10:
        raise ValueError("n_mc must be at least 10.")
    if not 0.0 <= min_prob <= 1.0:
        raise ValueError("min_prob must be between 0 and 1.")
    if not 0.0 <= quality_threshold <= 1.0:
        raise ValueError("quality_threshold must be between 0 and 1.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    rng = np.random.RandomState(seed)
    bounds = [(320.0, 420.0), (0.5, 6.0), (0.8, 1.2)]

    # Common random numbers keep candidate evaluations reproducible and
    # comparable during SLSQP; the objective/constraint no longer move
    # because fresh Monte Carlo noise is sampled on every evaluation.
    Ea_samples = 5e4 + rng.normal(0, 1500, n_mc)
    observation_noise = rng.normal(0, noise_std, n_mc)

    def mc_stats(x: np.ndarray) -> Tuple[float, float, float]:
        T, tau, C_A0 = x
        conversions = []
        yields = []
        energies = []
        for Ea in Ea_samples:
            r = cstr_steady_state(C_A0, T, tau, Ea=float(Ea))
            conversions.append(r["conversion"])
            yields.append(r["yield"])
            energies.append(r["energy"])
        conversions = np.array(conversions)
        yields = np.array(yields)
        energies = np.array(energies)
        conversions = np.clip(conversions + observation_noise, 0, 1)
        prob = float((conversions >= quality_threshold).mean())
        return float(yields.mean()), float(energies.mean()), prob

    def objective(x: np.ndarray) -> float:
        ey, ee, _ = mc_stats(x)
        return -(w_yield * ey - w_energy * ee)

    def constraint_prob(x: np.ndarray) -> float:
        _, _, prob = mc_stats(x)
        return prob - min_prob

    cons = [{"type": "ineq", "fun": constraint_prob}]
    x0 = np.array([400.0, 5.0, 1.0])
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 100, "ftol": 1e-8},
    )
    ey, ee, prob = mc_stats(res.x)
    T, tau, C_A0 = res.x
    success = bool(res.success) and prob >= min_prob
    return {
        "success": success,
        "T": float(T),
        "tau": float(tau),
        "C_A0": float(C_A0),
        "expected_yield": ey,
        "expected_energy": ee,
        "P_quality": prob,
        "quality_threshold": quality_threshold,
        "min_prob": min_prob,
        "method": "risk_aware",
        "message": res.message,
    }
