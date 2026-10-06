"""Evaluation: metrics per lead day, reliability, alert-threshold selection. Nothing hardcoded."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _f(x: float | None) -> float | None:
    """JSON-safe float (``None`` for NaN/inf)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """PR-AUC, ROC-AUC, Brier and precision/recall/F1 at ``threshold``, exactly as computed.

    ROC-AUC and PR-AUC are ``None`` when only one class is present.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    both = len(np.unique(y)) == 2
    pred = (p >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "base_rate": _f(y.mean()) if len(y) else None,
        "pr_auc": _f(average_precision_score(y, p)) if both else None,
        "roc_auc": _f(roc_auc_score(y, p)) if both else None,
        "brier": _f(brier_score_loss(y, p)) if len(y) else None,
        "threshold": _f(threshold),
        "n_alerts": int(pred.sum()),
        "precision": _f(precision_score(y, pred, zero_division=0)),
        "recall": _f(recall_score(y, pred, zero_division=0)),
        "f1": _f(f1_score(y, pred, zero_division=0)),
    }


def metrics_by_lead(
    df: pd.DataFrame, y_col: str, p_col: str, threshold: float, lead_col: str = "lead_day"
) -> dict[str, Any]:
    """Overall and per-lead-day metrics for one predictor."""
    out: dict[str, Any] = {
        "overall": binary_metrics(df[y_col].to_numpy(), df[p_col].to_numpy(), threshold),
        "by_lead": {},
    }
    for lead, g in df.groupby(lead_col):
        out["by_lead"][str(int(lead))] = binary_metrics(
            g[y_col].to_numpy(), g[p_col].to_numpy(), threshold
        )
    return out


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """Uniform-bin reliability table: mean predicted vs observed frequency per bin."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append(
                {
                    "bin_lo": float(edges[b]),
                    "bin_hi": float(edges[b + 1]),
                    "mean_pred": float(p[mask].mean()),
                    "obs_freq": float(y[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return rows


def choose_alert_threshold(
    y_val: np.ndarray, p_val: np.ndarray, precision_target: float
) -> dict[str, Any]:
    """Pick the threshold on VALIDATION data that maximises recall s.t. precision >= target.

    If no threshold reaches the target, fall back to the max-F1 threshold and report
    ``target_met = False`` (never silently).
    """
    y_val = np.asarray(y_val).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_val, np.asarray(p_val, float))
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    ok = precision >= precision_target
    if ok.any():
        candidates = np.flatnonzero(ok)
        best = candidates[np.argmax(recall[candidates])]
        rule = f"max recall with validation precision >= {precision_target}"
        met = True
    else:
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
        best = int(np.argmax(f1))
        rule = f"precision target {precision_target} not reachable on validation; max-F1 fallback"
        met = False
        logger.warning("Alert threshold: %s", rule)
    return {
        "threshold": float(thresholds[best]),
        "val_precision": float(precision[best]),
        "val_recall": float(recall[best]),
        "precision_target": float(precision_target),
        "target_met": met,
        "rule": rule,
    }


def threshold_tradeoff(
    y: np.ndarray, p: np.ndarray, grid: Sequence[float] | None = None
) -> list[dict[str, Any]]:
    """Precision/recall/F1/alert-rate over a threshold grid (for the dashboard slider)."""
    grid = grid if grid is not None else np.round(np.arange(0.02, 1.0, 0.02), 2)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    out = []
    for t in grid:
        pred = (p >= t).astype(int)
        out.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "alert_rate": float(pred.mean()),
            }
        )
    return out


def compare_to_baselines(
    by_predictor: Mapping[str, Mapping[str, Any]],
    model: str,
    baselines: Sequence[str],
    metric: str = "pr_auc",
) -> dict[str, Any]:
    """Per-lead-day verdicts: does ``model`` beat each baseline on ``metric``? (higher=better)."""
    leads = sorted(by_predictor[model]["by_lead"], key=int)
    verdict: dict[str, Any] = {"metric": metric, "model": model, "by_baseline": {}}
    for base in baselines:
        wins = []
        for lead in leads:
            m = by_predictor[model]["by_lead"][lead][metric]
            b = by_predictor[base]["by_lead"][lead][metric]
            wins.append(m is not None and b is not None and m > b)
        verdict["by_baseline"][base] = {
            "leads_won": int(sum(wins)),
            "leads_total": len(leads),
            "won_by_lead": dict(zip(leads, wins, strict=True)),
        }
    return verdict


def plot_reliability(curves: Mapping[str, list[dict[str, Any]]], path: Path, title: str) -> Path:
    """Save a reliability diagram PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    for name, rows in curves.items():
        if rows:
            ax.plot(
                [r["mean_pred"] for r in rows],
                [r["obs_freq"] for r in rows],
                marker="o",
                label=name,
            )
    ax.set_xlabel("Mean predicted bust probability")
    ax.set_ylabel("Observed bust frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
