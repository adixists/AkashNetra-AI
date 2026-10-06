"""Train the bust models, calibrate, evaluate against baselines and save every artifact.

Data usage (by year, never by random row):

* TRAIN years: feature climatologies, analog library, model fitting, baselines.
* VALIDATION years: early stopping, optional light tuning, calibration, alert threshold.
* TEST years: final metrics only. Nothing is fitted or selected on them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

from akashnetra.config import AppConfig
from akashnetra.features.analogs import AnalogIndex, neighbours_frame
from akashnetra.features.build import (
    FEATURE_COLUMNS,
    build_base_features,
    check_no_label_features,
    fit_feature_climatology,
)
from akashnetra.logging_setup import log_run_header
from akashnetra.models.baselines import ClimatologyBaseline, SpreadLogisticBaseline
from akashnetra.models.calibrate import Calibrator
from akashnetra.models.evaluate import (
    binary_metrics,
    choose_alert_threshold,
    compare_to_baselines,
    metrics_by_lead,
    plot_reliability,
    reliability_curve,
    threshold_tradeoff,
)
from akashnetra.processing.dataset import TABLE_FILENAME
from akashnetra.runinfo import get_git_hash, library_versions, set_seeds

logger = logging.getLogger(__name__)

PREDICTIONS_FILENAME = "predictions.parquet"
NEIGHBOURS_FILENAME = "analog_neighbors.parquet"
METRICS_FILENAME = "metrics.json"
ALERT_THRESHOLDS_FILENAME = "alert_thresholds.json"
RELIABILITY_PNG = "reliability.png"
TRAIN_META_FILENAME = "training_meta.json"

#: Probability columns written to the predictions table.
PRED_COLUMNS: dict[str, str] = {
    "lightgbm": "p_lightgbm",
    "xgboost": "p_xgboost",
    "spread_logistic": "p_spread_logistic",
    "climatology": "p_climatology",
}
PRIMARY_MODEL = "lightgbm"


@dataclass(frozen=True)
class TrainResult:
    """Where the outputs went, plus the computed metrics."""

    artifacts_dir: Path
    predictions_path: Path
    metrics_path: Path
    reliability_path: Path
    metrics: dict[str, Any]


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def _pos_weight(y: pd.Series) -> float:
    pos = float(y.sum())
    if pos == 0:
        raise ValueError("No positive (bust) samples in the training data")
    return (len(y) - pos) / pos


def fit_lightgbm(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_va: pd.DataFrame,
    y_va: pd.Series,
    params: dict[str, Any],
    seed: int,
    early_stopping_rounds: int,
) -> lgb.LGBMClassifier:
    """LightGBM with class weighting and early stopping on validation PR-AUC."""
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="average_precision",
        scale_pos_weight=_pos_weight(y_tr),
        subsample_freq=1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def fit_xgboost(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_va: pd.DataFrame,
    y_va: pd.Series,
    params: dict[str, Any],
    seed: int,
    early_stopping_rounds: int,
) -> XGBClassifier:
    """XGBoost cross-check on the same features, early stopping on validation PR-AUC."""
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=_pos_weight(y_tr),
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    return model


def tune_lightgbm(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_va: pd.DataFrame,
    y_va: pd.Series,
    base: dict[str, Any],
    trials: int,
    seed: int,
    early_stopping_rounds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Small random search scored on validation PR-AUC. Returns best params and the log."""
    space = {
        "num_leaves": [15, 31, 63],
        "min_child_samples": [20, 50, 100, 200],
        "learning_rate": [0.02, 0.03, 0.05],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    log: list[dict[str, Any]] = []
    best_params, best_score = dict(base), -np.inf
    for cand in ParameterSampler(space, n_iter=trials, random_state=seed):
        params = {**base, **cand}
        model = fit_lightgbm(x_tr, y_tr, x_va, y_va, params, seed, early_stopping_rounds)
        score = float(average_precision_score(y_va, model.predict_proba(x_va)[:, 1]))
        log.append({"params": cand, "val_pr_auc": score})
        logger.info("Tuning trial %s -> val PR-AUC %.4f", cand, score)
        if score > best_score:
            best_params, best_score = params, score
    return best_params, log


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def load_unified_table(cfg: AppConfig) -> pd.DataFrame:
    """Load the labelled unified table written by :func:`build_dataset`."""
    path = cfg.processed_dir / TABLE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Build the dataset first (make demo / scripts/build_dataset.py)."
        )
    table = pd.read_parquet(path)
    modes = set(table["data_mode"].unique())
    if modes != {cfg.data_mode}:
        raise ValueError(f"Dataset data_mode {modes} does not match config '{cfg.data_mode}'")
    return table


def build_features(
    table: pd.DataFrame, cfg: AppConfig
) -> tuple[pd.DataFrame, Any, AnalogIndex, pd.DataFrame]:
    """Base features + analog features for every row. Returns (features, clim, index, nbrs)."""
    train_years = cfg.years.active_split.train.years
    clim = fit_feature_climatology(table, train_years)
    feats = build_base_features(table, clim, cfg)
    feats = feats.sort_values(["init_date", "lead_day", "box_id"], ignore_index=True)
    feats.insert(0, "row_id", np.arange(len(feats), dtype=np.int64))

    index = AnalogIndex.fit(feats, train_years, cfg.features.analogs.k)
    ids, dist = index.neighbours_for(feats)
    analog = index.analog_features(ids)
    for col in analog.columns:
        feats[col] = analog[col].to_numpy(np.float32)
    nbrs = neighbours_frame(ids, dist, feats["row_id"].to_numpy())
    check_no_label_features(FEATURE_COLUMNS)
    return feats, clim, index, nbrs


def run_training(cfg: AppConfig, table: pd.DataFrame | None = None) -> TrainResult:
    """Full M2 pipeline. Writes artifacts under ``cfg.model_artifacts_dir``."""
    log_run_header(logger, cfg, title="train")
    set_seeds(cfg.model.seed)
    table = load_unified_table(cfg) if table is None else table
    split = cfg.years.active_split

    feats, clim, index, nbrs = build_features(table, cfg)
    tr, va, te = (feats[feats["split"] == s] for s in ("train", "val", "test"))
    for name, part in (("train", tr), ("val", va), ("test", te)):
        if part.empty or part["bust"].nunique() < 2:
            raise ValueError(f"Split '{name}' is empty or has a single class")
    x_tr, x_va, x_te = (p[FEATURE_COLUMNS] for p in (tr, va, te))
    y_tr, y_va, y_te = (p["bust"].astype(int) for p in (tr, va, te))
    logger.info("Rows train/val/test = %d/%d/%d", len(tr), len(va), len(te))

    seed, es = cfg.model.seed, cfg.model.early_stopping_rounds
    lgb_params, tuning_log = dict(cfg.model.lightgbm), []
    if cfg.model.tuning_trials > 0:
        lgb_params, tuning_log = tune_lightgbm(
            x_tr, y_tr, x_va, y_va, lgb_params, cfg.model.tuning_trials, seed, es
        )
    lgbm = fit_lightgbm(x_tr, y_tr, x_va, y_va, lgb_params, seed, es)
    xgbm = fit_xgboost(x_tr, y_tr, x_va, y_va, dict(cfg.model.xgboost), seed, es)
    logger.info(
        "LightGBM best_iter=%s, XGBoost best_iter=%s", lgbm.best_iteration_, xgbm.best_iteration
    )

    # Raw scores, calibration on VALIDATION only.
    raw = {
        "lightgbm": {s: lgbm.predict_proba(x)[:, 1] for s, x in (("val", x_va), ("test", x_te))},
        "xgboost": {s: xgbm.predict_proba(x)[:, 1] for s, x in (("val", x_va), ("test", x_te))},
    }
    calibrators = {m: Calibrator(cfg.model.calibration).fit(raw[m]["val"], y_va) for m in raw}
    climatology = ClimatologyBaseline().fit(tr["lead_day"], y_tr)
    spread = SpreadLogisticBaseline(seed).fit(tr["fcst_rain_spread"], y_tr)

    def predict_all(part: pd.DataFrame, raw_part: dict[str, np.ndarray]) -> pd.DataFrame:
        out = pd.DataFrame(index=part.index)
        for m in ("lightgbm", "xgboost"):
            out[f"{PRED_COLUMNS[m]}_raw"] = raw_part[m]
            out[PRED_COLUMNS[m]] = calibrators[m].transform(raw_part[m])
        out[PRED_COLUMNS["spread_logistic"]] = spread.predict_proba(part["fcst_rain_spread"])
        out[PRED_COLUMNS["climatology"]] = climatology.predict_proba(part["lead_day"])
        return out

    preds_va = predict_all(va, {m: raw[m]["val"] for m in raw})
    preds_te = predict_all(te, {m: raw[m]["test"] for m in raw})

    # Alert thresholds: chosen on VALIDATION per predictor.
    target = cfg.alerts.precision_target
    alert_thresholds = {
        name: choose_alert_threshold(y_va.to_numpy(), preds_va[col].to_numpy(), target)
        for name, col in PRED_COLUMNS.items()
    }

    # Metrics on TEST.
    eval_te = pd.concat([te[["lead_day", "bust"]], preds_te], axis=1)
    by_predictor = {
        name: metrics_by_lead(eval_te, "bust", col, alert_thresholds[name]["threshold"])
        for name, col in PRED_COLUMNS.items()
    }
    calibration_report = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    for m in ("lightgbm", "xgboost"):
        raw_col, cal_col = f"{PRED_COLUMNS[m]}_raw", PRED_COLUMNS[m]
        before = reliability_curve(y_te.to_numpy(), preds_te[raw_col].to_numpy())
        after = reliability_curve(y_te.to_numpy(), preds_te[cal_col].to_numpy())
        calibration_report[m] = {
            "test_brier_raw": binary_metrics(y_te, preds_te[raw_col], 0.5)["brier"],
            "test_brier_calibrated": binary_metrics(y_te, preds_te[cal_col], 0.5)["brier"],
            "reliability_raw": before,
            "reliability_calibrated": after,
        }
        curves[f"{m} (raw)"] = before
        curves[f"{m} (calibrated, {cfg.model.calibration})"] = after
    curves["spread-only logistic"] = reliability_curve(
        y_te.to_numpy(), preds_te[PRED_COLUMNS["spread_logistic"]].to_numpy()
    )

    out_dir = cfg.model_artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reliability_path = plot_reliability(
        curves,
        out_dir / RELIABILITY_PNG,
        f"Reliability on test years {split.test.years} ({cfg.data_label})",
    )

    comparisons = {
        metric: compare_to_baselines(
            by_predictor, PRIMARY_MODEL, ["climatology", "spread_logistic"], metric
        )
        for metric in ("pr_auc",)
    }
    brier_cmp = _compare_lower_is_better(by_predictor, PRIMARY_MODEL, "brier")
    comparisons["brier"] = brier_cmp

    importances = dict(
        sorted(
            zip(FEATURE_COLUMNS, lgbm.booster_.feature_importance("gain").tolist(), strict=True),
            key=lambda kv: -kv[1],
        )
    )
    metrics = {
        "data_mode": cfg.data_mode,
        "data_label": cfg.data_label,
        "model_version": cfg.project.model_version,
        "evaluated_on": "test",
        "years": {s: split.years_of(s) for s in ("train", "val", "test")},
        "active_year_mode": cfg.years.active,
        "random_baseline_pr_auc_note": "A random/constant predictor has PR-AUC = bust base rate.",
        "predictors": by_predictor,
        "primary_model": PRIMARY_MODEL,
        "comparisons": comparisons,
        "calibration": {"method": cfg.model.calibration, "fitted_on": "val", **calibration_report},
        "alert_thresholds": alert_thresholds,
        "threshold_tradeoff_test": threshold_tradeoff(
            y_te.to_numpy(), preds_te[PRED_COLUMNS[PRIMARY_MODEL]].to_numpy()
        ),
        "feature_importance_gain": importances,
        "features": FEATURE_COLUMNS,
        "lightgbm_params": {**lgb_params, "best_iteration": int(lgbm.best_iteration_ or 0)},
        "xgboost_params": {**cfg.model.xgboost, "best_iteration": int(xgbm.best_iteration)},
        "tuning_log": tuning_log,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": get_git_hash(cfg.root),
    }

    # Persist artifacts.
    joblib.dump(lgbm, out_dir / "lightgbm.joblib")
    joblib.dump(xgbm, out_dir / "xgboost.joblib")
    joblib.dump(calibrators, out_dir / "calibrators.joblib")
    joblib.dump(
        {"climatology": climatology, "spread_logistic": spread}, out_dir / "baselines.joblib"
    )
    joblib.dump(clim, out_dir / "feature_climatology.joblib")
    joblib.dump(index, out_dir / "analog_index.joblib")
    _write_json(out_dir / ALERT_THRESHOLDS_FILENAME, {"primary": PRIMARY_MODEL, **alert_thresholds})
    metrics_path = _write_json(out_dir / METRICS_FILENAME, metrics)
    _write_json(
        out_dir / TRAIN_META_FILENAME,
        {
            "data_mode": cfg.data_mode,
            "model_version": cfg.project.model_version,
            "years": metrics["years"],
            "features": FEATURE_COLUMNS,
            "analog_k": cfg.features.analogs.k,
            "analog_inputs": index.inputs,
            "calibration": cfg.model.calibration,
            "seed": seed,
            "git_hash": metrics["git_hash"],
            "created_utc": metrics["created_utc"],
            "libraries": library_versions(),
        },
    )

    # Predictions for VALIDATION + TEST rows (what the API serves), plus all neighbours.
    keep = [c for c in feats.columns if c not in ("month",)]
    served = pd.concat(
        [pd.concat([va[keep], preds_va], axis=1), pd.concat([te[keep], preds_te], axis=1)]
    )
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = cfg.processed_dir / PREDICTIONS_FILENAME
    served.to_parquet(predictions_path, index=False)
    nbrs.to_parquet(cfg.processed_dir / NEIGHBOURS_FILENAME, index=False)

    _log_summary(metrics)
    return TrainResult(out_dir, predictions_path, metrics_path, reliability_path, metrics)


def _compare_lower_is_better(
    by_predictor: dict[str, Any], model: str, metric: str
) -> dict[str, Any]:
    leads = sorted(by_predictor[model]["by_lead"], key=int)
    out: dict[str, Any] = {"metric": metric, "model": model, "by_baseline": {}}
    for base in ("climatology", "spread_logistic"):
        wins = [
            by_predictor[model]["by_lead"][ld][metric] < by_predictor[base]["by_lead"][ld][metric]
            for ld in leads
        ]
        out["by_baseline"][base] = {
            "leads_won": int(sum(wins)),
            "leads_total": len(leads),
            "won_by_lead": dict(zip(leads, wins, strict=True)),
        }
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def _log_summary(metrics: dict[str, Any]) -> None:
    """Log the computed headline numbers (exactly as computed)."""
    for name, res in metrics["predictors"].items():
        o = res["overall"]
        logger.info(
            "TEST %-16s PR-AUC=%s ROC-AUC=%s Brier=%s P=%s R=%s",
            name,
            _fmt(o["pr_auc"]),
            _fmt(o["roc_auc"]),
            _fmt(o["brier"]),
            _fmt(o["precision"]),
            _fmt(o["recall"]),
        )
    for metric, cmp_ in metrics["comparisons"].items():
        for base, v in cmp_["by_baseline"].items():
            logger.info(
                "%s beats %s on %s at %d/%d lead days",
                cmp_["model"],
                base,
                metric,
                v["leads_won"],
                v["leads_total"],
            )


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"
