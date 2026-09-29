#!/usr/bin/env python3
"""ChemX end-to-end experiment runner.

Produces real metrics on real datasets + mechanistic simulation.
Writes results to reports/results.json and prints a summary table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_mlnir, load_corn_all, load_sugar, train_test_split_indices
from src.preprocessing.spectral import apply_preprocessing
from src.chemometrics.pca import ChemometricPCA
from src.chemometrics.pls import PLSModel, compare_regressors, regression_metrics
from src.bayesian.soft_sensor import BayesianSoftSensor
from src.monitoring.process_monitor import ProcessMonitor
from src.transfer.calibration_transfer import transfer_experiment
from src.physics_informed.reactor import simulate_campaign, physics_residual
from src.optimization.process_opt import optimize_deterministic, optimize_risk_aware


def run_mlnir() -> dict:
    print("\n=== MLNIRdata: density prediction ===")
    data = load_mlnir()
    X, y = data["X"], data["y"]
    tr, te = train_test_split_indices(len(y), test_size=0.25, seed=42)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    # Preprocessing comparison
    prep_results = {}
    for method in ["raw", "snv", "msc", "sg1", "snv_sg1"]:
        Xt, Xs, _ = apply_preprocessing(Xtr, Xte, method)
        # mean-center for PLS
        mu = Xt.mean(0)
        pls = PLSModel(n_components=10)
        pls.fit(Xt - mu, ytr)
        m = pls.evaluate(Xs - mu, yte)
        prep_results[method] = m
        print(f"  Prep {method:8s}: RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")

    best_prep = min(prep_results, key=lambda k: prep_results[k]["RMSE"])
    Xt, Xs, _ = apply_preprocessing(Xtr, Xte, best_prep)
    mu = Xt.mean(0)
    Xt, Xs = Xt - mu, Xs - mu

    # Model comparison
    model_cmp = compare_regressors(Xt, ytr, Xs, yte, n_components=10)
    print("  Model comparison:")
    for name, m in model_cmp.items():
        print(f"    {name:12s}: RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")

    # Bayesian soft sensor
    bayes = BayesianSoftSensor(n_features_pca=30)
    bayes.fit(Xt, ytr)
    interval = bayes.predict_interval(Xs)
    cov = bayes.coverage(Xs, yte)
    print(f"  Bayesian coverage (95%): {cov:.3f}")
    print(f"  Example pred={interval['prediction'][0]:.3f} "
          f"[{interval['lower'][0]:.3f}, {interval['upper'][0]:.3f}]")

    # PCA monitoring
    pca = ChemometricPCA(n_components=5)
    pca.fit(Xt)
    mon = pca.monitor(Xs)
    print(f"  PCA anomalies on test: {mon['anomaly'].sum()}/{len(mon['anomaly'])}")

    # VIP
    pls = PLSModel(n_components=10)
    pls.fit(Xt, ytr)
    vip = pls.vip_scores()
    top_wl = np.argsort(vip)[::-1][:5]

    return {
        "dataset": "MLNIRdata",
        "n_train": len(tr),
        "n_test": len(te),
        "best_preprocessing": best_prep,
        "preprocessing": prep_results,
        "models": model_cmp,
        "bayesian_coverage_95": cov,
        "pca_anomaly_rate_test": float(mon["anomaly"].mean()),
        "top_vip_indices": top_wl.tolist(),
    }


def run_corn_transfer() -> dict:
    print("\n=== Corn multi-instrument calibration transfer ===")
    all_inst = load_corn_all()
    y = all_inst["m5"]["y"]
    results = {}
    for target, prop_idx in [("Moisture", 0), ("Protein", 2)]:
        print(f"  Target: {target}")
        for pair in [("m5", "mp5"), ("m5", "mp6"), ("mp5", "mp6")]:
            r = transfer_experiment(
                all_inst[pair[0]]["X"],
                all_inst[pair[1]]["X"],
                y,
                property_idx=prop_idx,
            )
            key = f"{target}_{pair[0]}_to_{pair[1]}"
            results[key] = r
            print(f"    {pair[0]}→{pair[1]}: no_transfer={r['no_transfer_RMSE']:.4f}  "
                  f"DS={r['direct_std_RMSE']:.4f}  same={r['same_instrument_RMSE']:.4f}")
    return {"dataset": "Corn", "transfer": results}


def run_sugar_monitoring() -> dict:
    print("\n=== Sugar process monitoring ===")
    data = load_sugar()
    X = data["X"]
    # Use first 180 as reference (NOC), rest as test
    Xref, Xtest = X[:180], X[180:]
    mon = ProcessMonitor(n_components=8)
    mon.fit(Xref)
    status = mon.summary_status(Xtest)
    print(f"  Test anomaly rate: {status['anomaly_rate']:.3f}  "
          f"(n_anom={status['n_anomalies']})")
    # Root cause on first flagged sample if any
    flags = mon.monitor(Xtest)["anomaly"]
    rc = None
    if flags.any():
        idx = int(np.where(flags)[0][0])
        rc = mon.root_cause(Xtest, sample_idx=idx, top_k=5)
        print(f"  Root-cause sample {idx}: top vars {rc['top_contributor_indices']}")
    return {
        "dataset": "Sugar",
        "monitoring": status,
        "root_cause_example": rc,
    }


def run_physics_and_opt() -> dict:
    print("\n=== Mechanistic simulation + optimization ===")
    sim = simulate_campaign(n_samples=250, seed=42)
    print(f"  Simulated {len(sim['conversion'])} points "
          f"(anomalies injected: {sim['anomaly'].sum()})")

    # Black-box vs physics residual
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    X = sim["X_process"]
    y = sim["conversion"]
    tr, te = train_test_split_indices(len(y), 0.25, 42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    nn = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    nn.fit(Xtr, y[tr])
    pred = nn.predict(Xte)
    nn_metrics = regression_metrics(y[te], pred)
    resid = physics_residual(sim["T"][te], sim["tau"][te], sim["C_A0"][te], pred)
    print(f"  NN RMSE={nn_metrics['RMSE']:.4f}  mean |physics residual|={np.abs(resid).mean():.4f}")

    det = optimize_deterministic()
    print(f"  Deterministic opt: yield={det['yield']:.4f}  T={det['T']:.1f}  tau={det['tau']:.2f}")
    risk = optimize_risk_aware(n_mc=100, min_prob=0.90, quality_threshold=0.70)
    print(f"  Risk-aware opt: E[yield]={risk['expected_yield']:.4f}  "
          f"P(quality)={risk['P_quality']:.3f}")

    return {
        "dataset": "MECHANISTIC_SIMULATION",
        "nn_metrics": nn_metrics,
        "mean_abs_physics_residual": float(np.abs(resid).mean()),
        "deterministic_opt": det,
        "risk_aware_opt": risk,
    }


def main():
    results = {
        "mlnir": run_mlnir(),
        "corn_transfer": run_corn_transfer(),
        "sugar_monitoring": run_sugar_monitoring(),
        "physics_opt": run_physics_and_opt(),
    }
    out = ROOT / "reports" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS TABLE")
    print("=" * 70)
    m = results["mlnir"]["models"]
    print(f"{'Component':<35} {'Baseline':<12} {'Proposed':<12} {'Metric'}")
    print("-" * 70)
    print(f"{'Chemical property (density)':<35} {'PCR':<12} {'PLS':<12} "
          f"RMSE {m['PLS']['RMSE']:.4f} vs {m['PCR']['RMSE']:.4f}")
    print(f"{'Nonlinear / regularized':<35} {'PLS':<12} {'Ridge':<12} "
          f"RMSE {m['Ridge']['RMSE']:.4f}")
    print(f"{'Uncertainty':<35} {'Point':<12} {'Bayesian':<12} "
          f"Coverage {results['mlnir']['bayesian_coverage_95']:.3f}")
    print(f"{'Monitoring':<35} {'Univariate':<12} {'PCA T2/Q':<12} "
          f"anom rate {results['sugar_monitoring']['monitoring']['anomaly_rate']:.3f}")
    # transfer
    tr = results["corn_transfer"]["transfer"]
    key = "Moisture_m5_to_mp5"
    if key in tr:
        print(f"{'Instrument transfer (Moist m5→mp5)':<35} {'Raw':<12} {'DS':<12} "
              f"RMSE {tr[key]['direct_std_RMSE']:.4f} (raw {tr[key]['no_transfer_RMSE']:.4f})")
    po = results["physics_opt"]
    print(f"{'Physics consistency':<35} {'Black-box NN':<12} {'Phys residual':<12} "
          f"|r|={po['mean_abs_physics_residual']:.4f}")
    print(f"{'Optimization':<35} {'Deterministic':<12} {'Risk-aware':<12} "
          f"P(qual)={po['risk_aware_opt']['P_quality']:.3f}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
