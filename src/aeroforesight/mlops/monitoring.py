"""Model & data monitoring: population-stability index (PSI) drift detection
plus a simple performance gate, the two checks most MLOps stacks run in CI/CD.
"""

from __future__ import annotations

import numpy as np


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI between a reference (training) distribution and a live one.

    < 0.10  no significant shift; 0.10–0.25 moderate; > 0.25 major drift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(expected, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(reference: dict[str, np.ndarray], live: dict[str, np.ndarray], cfg: dict) -> dict:
    warn = cfg["monitoring"]["drift_psi_warn"]
    alert = cfg["monitoring"]["drift_psi_alert"]
    features = {}
    worst = 0.0
    for name in reference:
        if name not in live:
            continue
        psi = population_stability_index(reference[name], live[name])
        status = "ok" if psi < warn else ("warn" if psi < alert else "alert")
        features[name] = {"psi": round(psi, 4), "status": status}
        worst = max(worst, psi)
    overall = "ok" if worst < warn else ("warn" if worst < alert else "alert")
    return {"overall_status": overall, "max_psi": round(worst, 4), "features": features}


def performance_gate(auc: float, cfg: dict) -> dict:
    min_auc = cfg["monitoring"]["min_auc"]
    passed = auc >= min_auc
    return {"metric": "val_auc", "value": round(auc, 4), "threshold": min_auc, "passed": bool(passed)}
