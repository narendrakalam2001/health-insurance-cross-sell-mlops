# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM — Health Insurance Cross-Sell ML System
# ============================================================
#
# CHALLENGER MODEL SYSTEM:
#   Real banks never blindly replace champion model with a new one.
#   They run a formal Champion vs Challenger comparison:
#
#   Champion  = current production model (latest_model.json)
#   Challenger = newly trained model (passed in from training_pipeline)
#
#   Promotion logic:
#     Challenger promoted ONLY if it beats Champion on ALL 3 gates:
#       1. F1 improvement  >= MIN_F1_IMPROVEMENT
#       2. ROC-AUC         >= MIN_ROCAUC_THRESHOLD
#       3. Train-test gap  <= MAX_GENERALIZATION_GAP
#
#   If challenger loses → champion stays, challenger archived.
#   Full comparison saved to risk_models/challenger_log.json
# ============================================================

import os
import json
import joblib
import logging
import time

from src.config import MODEL_DIR, MIN_F1_IMPROVEMENT, MIN_ROCAUC_THRESHOLD, MAX_GENERALIZATION_GAP

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model():
    """
    Reads risk_models/latest_model.json → loads .joblib + threshold.
    Returns: (model_pipeline, threshold)
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. Run train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(MODEL_DIR, registry["model_name"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model     = joblib.load(model_path)
    threshold = float(registry.get("threshold", 0.5))

    logger.info("Champion model loaded: %s  |  threshold=%.4f", model_path, threshold)

    return model, threshold


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================

def _load_champion_metrics() -> dict:
    """
    Reads the current champion model card to get its metrics.
    Returns empty dict if no champion exists yet.
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    # FIX: read model_card_path directly from registry — no brittle string parsing
    card_path = registry.get("model_card_path", "")

    # Backward compatibility — if old registry has no model_card_path, fall back to parsing
    if not card_path:
        model_name = registry.get("model_name", "")
        parts      = model_name.replace("credit_model_", "").replace(".joblib", "")
        card_path  = os.path.join(MODEL_DIR, f"model_card_{parts}.json")
        logger.warning("model_card_path missing in registry — using fallback: %s", card_path)

    if not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", card)

    return {
        "model_name":     card.get("model_name", "unknown"),
        "f1":             float(metrics.get("test_f1",   0)),
        "roc_auc":        float(metrics.get("roc_auc",   0)),
        # NOTE: train_test_gap is intentionally NOT read from the champion's
        # model card here. The model card never stores it under this key
        # (it only stores "selector_k" under pipeline_config, which is an
        # unrelated SelectKBest hyperparameter). The champion's gap is not
        # actually needed for promotion — gate3 in run_challenger_comparison()
        # only checks the CHALLENGER's freshly computed gap against the fixed
        # MAX_GENERALIZATION_GAP config threshold, so this field is unused
        # downstream. Kept out entirely to avoid a misleading dead field.
        "threshold":      float(card.get("thresholds", {}).get("active", 0.5)),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC
# ============================================================

def run_challenger_comparison(
    challenger_name:       str,
    challenger_f1:         float,
    challenger_roc_auc:    float,
    challenger_gap:        float,
    challenger_model_path: str,
    challenger_threshold:  float,
    challenger_card_path:  str = "",   # FIX: pass card path so registry stores it directly
) -> dict:
    """
    Compares challenger vs current champion.

    Args:
        challenger_name       : model name e.g. 'LightGBM'
        challenger_f1         : challenger test F1 score
        challenger_roc_auc    : challenger ROC-AUC
        challenger_gap        : challenger train-test accuracy gap
        challenger_model_path : path to challenger .joblib
        challenger_threshold  : challenger decision threshold

    Returns:
        result dict with decision: 'PROMOTED' or 'REJECTED'
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    champion = _load_champion_metrics()

    # ── If no champion exists → auto-promote ─────────────────
    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(challenger_model_path, challenger_threshold, challenger_card_path)
        result = {
            "decision":          "PROMOTED",
            "reason":            "No existing champion — first model auto-promoted",
            "challenger_name":   challenger_name,
            "challenger_f1":     round(challenger_f1,      4),
            "challenger_roc_auc":round(challenger_roc_auc, 4),
            "champion_name":     None,
            "champion_f1":       None,
            "evaluated_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(result)
        return result

    champion_f1     = champion.get("f1",    0.0)
    champion_roc    = champion.get("roc_auc", 0.0)
    champion_name   = champion.get("model_name", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-20s  F1=%.4f  ROC=%.4f",
                champion_name, champion_f1, champion_roc)
    logger.info("  Challenger: %-20s  F1=%.4f  ROC=%.4f",
                challenger_name, challenger_f1, challenger_roc_auc)
    logger.info("=" * 55)

    # ── Promotion gates ───────────────────────────────────────
    gate1_f1_improvement = (challenger_f1 - champion_f1) >= MIN_F1_IMPROVEMENT
    gate2_roc_auc        = challenger_roc_auc >= MIN_ROCAUC_THRESHOLD
    gate3_gap            = challenger_gap <= MAX_GENERALIZATION_GAP

    gates_passed = gate1_f1_improvement and gate2_roc_auc and gate3_gap

    if gates_passed:
        decision = "PROMOTED"
        reason   = (
            f"Challenger beats champion: "
            f"F1 {champion_f1:.4f} → {challenger_f1:.4f} "
            f"(+{challenger_f1 - champion_f1:.4f})"
        )
        logger.info("✅ CHALLENGER PROMOTED → new champion: %s", challenger_name)
        _update_registry(challenger_model_path, challenger_threshold, challenger_card_path)

    else:
        decision = "REJECTED"
        failed   = []
        if not gate1_f1_improvement:
            failed.append(
                f"F1 improvement {challenger_f1 - champion_f1:+.4f} < {MIN_F1_IMPROVEMENT}"
            )
        if not gate2_roc_auc:
            failed.append(f"ROC-AUC {challenger_roc_auc:.4f} < {MIN_ROCAUC_THRESHOLD}")
        if not gate3_gap:
            failed.append(f"train-test gap {challenger_gap:.4f} > {MAX_GENERALIZATION_GAP}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("❌ CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":              decision,
        "reason":                reason,
        "evaluated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name":       challenger_name,
        "challenger_f1":         round(challenger_f1,      4),
        "challenger_roc_auc":    round(challenger_roc_auc, 4),
        "challenger_gap":        round(challenger_gap,     4),
        "champion_name":         champion_name,
        "champion_f1":           round(champion_f1,        4),
        "champion_roc_auc":      round(champion_roc,       4),
        "gates": {
            "f1_improvement_passed": gate1_f1_improvement,
            "roc_auc_passed":        gate2_roc_auc,
            "gap_passed":            gate3_gap,
        }
    }

    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(model_path: str, threshold: float, model_card_path: str = None):
    # store model_card_path directly — eliminates brittle filename string parsing.
    # This is the ONLY function that should ever write latest_model.json, and it
    # must only be called from inside run_challenger_comparison() after the
    # challenger has genuinely passed the promotion gates (or is the first model).
    registry = {
        "model_name":       os.path.basename(model_path),
        "threshold":        round(threshold, 4),
        "model_card_path":  model_card_path or ""
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated → %s", registry["model_name"])


def _save_challenger_log(result: dict):
    """Appends challenger comparison result to challenger_log.json."""
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(result)

    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Challenger log saved → %s", CHALLENGER_LOG)