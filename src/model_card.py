# ============================================================
# MODEL CARD — Health Insurance Cross-Sell ML System
# ============================================================
# Builds and saves a structured model card — standard practice
# in production ML systems (similar to Google Model Cards).
#
# Contains:
#   - Model metadata (name, version, date)
#   - Dataset splits info
#   - All evaluation metrics
#   - Risk decision summary
#   - Cost evaluation result
#   - Feature importances
#   - SHAP top features
#   - Threshold values (uncalibrated + calibrated)
# ============================================================

import os
import json
import time
import logging

from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# BUILD MODEL CARD
# ============================================================

def build_model_card(
    selected_name:    str,
    train_fit_size:   int,
    cal_size:         int,
    test_size:        int,
    response_rate_train:  float,
    metrics:          dict,          # test_f1, precision, recall, roc_auc, ks, recall_at_5, lift_at_5, brier
    thr_uncalibrated: float,
    thr_calibrated:   Optional[float],
    cost_result:      dict,
    decision_counts:  dict,
    feature_order:    list,
    cat_indices:      list,
    selector_k:       int,
    version:          str = "v1",
    fi_dict:          Optional[dict] = None,
    shap_dict:        Optional[dict] = None,
) -> dict:
    """
    Assembles the full model card dictionary.

    Args:
        selected_name     : winning model name (e.g. 'LightGBM')
        train_fit_size    : rows used for model training
        cal_size          : rows used for calibration
        test_size         : rows used for final evaluation
        response_rate_train   : positive class rate in training set
        metrics           : dict of evaluation metric values
        thr_uncalibrated  : threshold before calibration
        thr_calibrated    : threshold after calibration (None if calibration failed)
        cost_result       : output of cost_sensitive_evaluation()
        decision_counts   : {decision: count} from risk_engine()
        feature_order     : column order used by preprocessor
        cat_indices       : categorical feature indices for SMOTENC
        selector_k        : k used in SelectKBest
        version           : model version string
        fi_dict           : feature importances dict (optional)
        shap_dict         : SHAP top features dict (optional)

    Returns:
        model_card dict (JSON-serialisable)
    """

    card = {

        # ── Identity ──────────────────────────────────────────
        "model_version":    version,
        "model_name":       selected_name,
        "trained_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "project":          "Health Insurance Cross-Sell Prediction System",

        # ── Dataset info ──────────────────────────────────────
        "dataset": {
            "train_fit_size":  train_fit_size,
            "calibration_size": cal_size,
            "test_size":        test_size,
            "response_rate_train":  round(float(response_rate_train), 4),
        },

        # ── Evaluation metrics ────────────────────────────────
        "metrics": {
            "test_f1":        round(float(metrics.get("test_f1",    0)), 4),
            "test_precision": round(float(metrics.get("precision",  0)), 4),
            "test_recall":    round(float(metrics.get("recall",     0)), 4),
            "roc_auc":        round(float(metrics.get("roc_auc",    0)), 4),
            "ks_statistic":   round(float(metrics.get("ks",         0)), 4),
            "recall_at_5pct": round(float(metrics.get("recall_at_5", 0)), 4),
            "lift_at_5pct":   round(float(metrics.get("lift_at_5",  0)), 4),
            "brier_score":    round(float(metrics.get("brier",      0)), 4),
        },

        # ── Thresholds ────────────────────────────────────────
        "thresholds": {
            "uncalibrated": round(float(thr_uncalibrated), 4),
            "calibrated":   round(float(thr_calibrated), 4) if thr_calibrated is not None else None,
            "active":       round(float(thr_calibrated if thr_calibrated is not None else thr_uncalibrated), 4),
        },

        # ── Risk decisions ────────────────────────────────────
        "risk_decisions": decision_counts,

        # ── Cost evaluation ───────────────────────────────────
        "cost_evaluation": cost_result,

        # ── Pipeline config ───────────────────────────────────
        "pipeline_config": {
            "feature_order":      feature_order,
            "categorical_indices": cat_indices,
            "selector_k":         selector_k,
        },
    }

    # ── Optional: feature importances ────────────────────────
    if fi_dict is not None:
        card["feature_importances"] = fi_dict

    # ── Optional: SHAP ───────────────────────────────────────
    if shap_dict is not None:
        card["shap_top_features"] = shap_dict

    return card


# ============================================================
# SAVE MODEL CARD
# ============================================================

def save_model_card(card: dict, model_dir: str, selected_name: str, version: str = "v1") -> str:
    """
    Saves model card as JSON to model_dir.
    Returns path of saved file.
    """
    os.makedirs(model_dir, exist_ok=True)

    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")

    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)

    logger.info("Model card saved → %s", card_path)

    return card_path


# ============================================================
# LOAD MODEL CARD
# ============================================================

def load_model_card(model_dir: str, selected_name: str, version: str = "v1") -> dict:
    """
    Loads existing model card JSON.
    Useful for dashboard / API to read model metadata.
    """
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")

    if not os.path.exists(card_path):
        raise FileNotFoundError(f"Model card not found: {card_path}")

    with open(card_path) as f:
        card = json.load(f)

    logger.info("Model card loaded ← %s", card_path)

    return card