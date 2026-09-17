# ============================================================
# TRAINING PIPELINE — Health Insurance Cross-Sell ML System
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics         import (precision_score, recall_score, f1_score,
                                     roc_auc_score, brier_score_loss, accuracy_score,
                                     classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config       import RANDOM_STATE, N_JOBS, CV_FOLDS, MODEL_DIR, SELECT_K
from src.data_loader  import validate_input_data, add_engineered_features, detect_feature_types
from src.preprocessing import build_preprocessors, safe_k
from src.model_tuning  import scaled_models, unscaled_models, tune_models, train_mlp_pipeline
from src.evaluation    import (evaluate_models, select_best_model, calibrate_with_holdout,
                                compute_feature_importance, compute_shap,
                                save_model_and_card, save_challenger_artifact,
                                mlflow_log_run, safe_predict_proba)
from src.leakage_check import detect_leakage
from src.model_card    import build_model_card, save_model_card
from src.model_loader  import run_challenger_comparison
from src.metrics       import tune_threshold, psi, ks_statistic, recall_at_k, lift_at_k, cost_sensitive_evaluation
from src.lead_engine    import lead_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Default data path — override with env var CROSSSELL_DATA_PATH ──
DEFAULT_DATA_PATH = r"D:\Data Science Datasets\Health Insurance Cross-Sell Prediction\train.csv"


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training(data_path: str = None):

    # ─────────────────────────────────────────────────────────
    # 1. LOAD & VALIDATE DATA
    # ─────────────────────────────────────────────────────────
    DATA_PATH = data_path or os.getenv("CROSSSELL_DATA_PATH", DEFAULT_DATA_PATH)

    logger.info("Loading data from: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = validate_input_data(df)

    # ─────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────
    df = add_engineered_features(df)

    # ─────────────────────────────────────────────────────────
    # 3. FEATURE TYPE DETECTION
    # ─────────────────────────────────────────────────────────
    ord_cols, cont_cols, bin_cols = detect_feature_types(df, threshold=10)

    X = df.drop(columns=["response"])
    y = df["response"]

    logger.info("Dataset  |  shape=%s  |  response_rate=%.3f", df.shape, y.mean())

    # ─────────────────────────────────────────────────────────
    # 4. TRAIN / TEST / CALIBRATION SPLIT
    # ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Reserve 20% of train for calibration (no SMOTE on this set)
    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )

    logger.info("Split  |  train_fit=%d  cal=%d  test=%d",
                len(X_train_fit), len(X_cal), len(X_test))

    # ─────────────────────────────────────────────────────────
    # 5. LEAKAGE DETECTION
    # ─────────────────────────────────────────────────────────
    leak_warnings = detect_leakage(X_train_fit, y_train_fit)
    if leak_warnings:
        for w in leak_warnings:
            logger.warning("LEAKAGE: %s", w)
    else:
        logger.info("No obvious leakage detected")

    # ─────────────────────────────────────────────────────────
    # 6. BUILD PREPROCESSORS
    # ─────────────────────────────────────────────────────────
    pre_scaled, pre_unscaled, cat_indices, feature_order = build_preprocessors(
        ord_cols, cont_cols, bin_cols, X_train_fit
    )

    k_safe = safe_k(SELECT_K, pre_scaled, X_train_fit)

    # ─────────────────────────────────────────────────────────
    # 7. TUNE SCALED MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning scaled models ...")
    scaled_pipelines = tune_models(
        scaled_models, pre_scaled, cat_indices,
        X_train_fit, y_train_fit,
        use_smote=True, selector_k=k_safe
    )

    # ─────────────────────────────────────────────────────────
    # 8. TUNE UNSCALED MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning unscaled models ...")
    unscaled_pipelines = tune_models(
        unscaled_models, pre_unscaled, cat_indices,
        X_train_fit, y_train_fit,
        use_smote=True, selector_k=k_safe
    )

    # ─────────────────────────────────────────────────────────
    # 9. TRAIN NEURAL NETWORK (separately)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    mlp_pipe = train_mlp_pipeline(X_train_fit, y_train_fit, pre_scaled, cat_indices)

    all_pipelines = {**scaled_pipelines, **unscaled_pipelines, "NeuralNet": mlp_pipe}

    # ─────────────────────────────────────────────────────────
    # 10. EVALUATE ALL MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating all models ...")
    summary = evaluate_models(all_pipelines, X_train_fit, y_train_fit, X_test, y_test)

    print("\n" + "=" * 60)
    print("ALL MODELS SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    # ─────────────────────────────────────────────────────────
    # 11. SELECT BEST MODEL
    # ─────────────────────────────────────────────────────────
    selected_name, selected_pipe = select_best_model(
        summary, all_pipelines, scaled_pipelines, unscaled_pipelines
    )

    # ─────────────────────────────────────────────────────────
    # 12. DETAILED EVALUATION — SELECTED MODEL
    # ─────────────────────────────────────────────────────────
    y_prob_sel = safe_predict_proba(selected_pipe, X_test)
    thr_sel    = tune_threshold(y_test.values, y_prob_sel)
    y_pred_sel = (y_prob_sel >= thr_sel).astype(int)

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {selected_name}")
    print("=" * 60)
    print(classification_report(y_test, y_pred_sel))

    # ── Confusion matrix ──────────────────────────────────────
    os.makedirs(os.path.join("docs", "plots"), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred_sel)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{selected_name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join("docs", "plots", "confusion_matrix.png"))
    plt.close()

    # ── ROC + PR curves ──────────────────────────────────────
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob_sel, ax=ax[0])
    PrecisionRecallDisplay.from_predictions(y_test, y_prob_sel, ax=ax[1])
    plt.suptitle(f"{selected_name} — Curves")
    plt.tight_layout()
    plt.savefig(os.path.join("docs", "plots", "roc_pr_curves.png"))
    plt.close()

    # ─────────────────────────────────────────────────────────
    # 13. LEAD ENGINE DECISIONS
    # ─────────────────────────────────────────────────────────
    decisions = lead_engine(X_test, y_prob_sel, thr_sel)
    decision_counts = pd.Series(decisions).value_counts()

    print("\nLEAD ENGINE DECISIONS")
    print(decision_counts)

    # ─────────────────────────────────────────────────────────
    # 14. COST-SENSITIVE (BUSINESS VALUE) EVALUATION
    # ─────────────────────────────────────────────────────────
    cost_result = cost_sensitive_evaluation(X_test, y_test, y_pred_sel)
    print("\nBUSINESS VALUE EVALUATION")
    for k, v in cost_result.items():
        print(f"  {k}: {v}")

    # ─────────────────────────────────────────────────────────
    # 15. CALIBRATION
    # ─────────────────────────────────────────────────────────
    cal_pipe   = calibrate_with_holdout(selected_pipe, X_cal, y_cal)
    thr_cal    = thr_sel
    y_prob_cal = None

    if cal_pipe is not None:
        y_prob_cal = cal_pipe.predict_proba(X_test)[:, 1]
        thr_cal    = tune_threshold(y_test.values, y_prob_cal)
        y_pred_cal = (y_prob_cal >= thr_cal).astype(int)

        brier_before = brier_score_loss(y_test, y_prob_sel)
        brier_after  = brier_score_loss(y_test, y_prob_cal)

        logger.info("Brier score  |  before=%.6f  after=%.6f", brier_before, brier_after)
        logger.info("Threshold after calibration: %.4f", thr_cal)

    # ─────────────────────────────────────────────────────────
    # 16. REPEATED CV STABILITY CHECK
    # ─────────────────────────────────────────────────────────
    try:
        rskf       = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
        rep_scores = cross_val_score(
            selected_pipe, X_train_fit, y_train_fit,
            scoring = "f1",
            cv      = rskf,
            n_jobs  = 1      # n_jobs=1 avoids joblib worker crashes on heavy CV
        )
        logger.info("Repeated CV  |  mean_f1=%.4f  std=%.4f",
                    rep_scores.mean(), rep_scores.std())
    except Exception as e:
        logger.warning("Repeated CV failed: %s", e)

    # ─────────────────────────────────────────────────────────
    # 17. FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────
    fi = compute_feature_importance(selected_pipe, X_train_fit, y_train_fit)
    if fi is not None:
        print("\nTOP FEATURE IMPORTANCES")
        print(fi.head(10))

    # ─────────────────────────────────────────────────────────
    # 18. SHAP EXPLAINABILITY
    # ─────────────────────────────────────────────────────────
    shap_result = compute_shap(selected_pipe, X_train_fit, X_test.head(200))

    # ─────────────────────────────────────────────────────────
    # 19. PSI — FEATURE DRIFT
    # ─────────────────────────────────────────────────────────
    psi_scores = {}
    for col in cont_cols:
        if col in X_train_fit.columns and col in X_test.columns:
            psi_scores[col] = psi(X_train_fit[col].values, X_test[col].values)

    psi_df = pd.Series(psi_scores).sort_values(ascending=False)
    psi_df.reset_index().rename(
        columns={"index": "feature", 0: "drift_score"}
    ).to_csv(os.path.join(MODEL_DIR, "feature_drift_report.csv"), index=False)

    print("\nTOP PSI (train vs test)")
    print(psi_df.head(10))

    # ─────────────────────────────────────────────────────────
    # 20. SAVE MONITOR SCORES
    # ─────────────────────────────────────────────────────────
    final_probs    = y_prob_cal if y_prob_cal is not None else y_prob_sel
    final_preds    = (final_probs >= thr_cal).astype(int)
    final_decisions= lead_engine(X_test, final_probs, thr_cal)

    monitor_df = pd.DataFrame({
        "score":            final_probs,
        "decision":         final_decisions,
        "propensity_band":  [_prob_to_band(p) for p in final_probs],
        "label":            y_test.values
    })
    monitor_df.to_csv(os.path.join(MODEL_DIR, "monitor_scores.csv"), index=False)
    logger.info("Monitor scores saved")

    # ─────────────────────────────────────────────────────────
    # 21. MODEL CARD
    # ─────────────────────────────────────────────────────────
    model_card = build_model_card(
        selected_name        = selected_name,
        train_fit_size       = int(len(X_train_fit)),
        cal_size              = int(len(X_cal)),
        test_size              = int(len(X_test)),
        response_rate_train  = float(y_train_fit.mean()),
        metrics = {
            "test_f1":    float(f1_score(y_test, y_pred_sel)),
            "precision":  float(precision_score(y_test, y_pred_sel)),
            "recall":     float(recall_score(y_test, y_pred_sel)),
            "roc_auc":    float(roc_auc_score(y_test, y_prob_sel)),
            "ks":         float(ks_statistic(y_test, y_prob_sel)),
            "recall_at_5":float(recall_at_k(y_test, y_prob_sel, 0.05)),
            "lift_at_5":  float(lift_at_k(y_test, y_prob_sel, 0.05)),
            "brier":      float(brier_score_loss(y_test, y_prob_sel)),
        },
        thr_uncalibrated = float(thr_sel),
        thr_calibrated   = float(thr_cal) if thr_cal != thr_sel else None,
        cost_result      = cost_result,
        decision_counts  = decision_counts.to_dict(),
        feature_order    = feature_order,
        cat_indices      = cat_indices,
        selector_k       = k_safe,
        fi_dict          = fi.head(20).to_dict() if fi is not None else None,
        shap_dict        = shap_result.get("shap_top", {}) if shap_result is not None else None,
    )
    # Timestamped version so each run gets a unique model card file —
    # prevents self-comparison bug in champion vs challenger step.
    import time as _time
    _run_ts   = _time.strftime("%Y%m%d_%H%M%S")
    card_path = save_model_card(model_card, MODEL_DIR, selected_name, version=f"v1_{_run_ts}")

    # ─────────────────────────────────────────────────────────
    # 22. SAVE MODEL (challenger — registry NOT updated yet)
    # ─────────────────────────────────────────────────────────
    final_pipe = cal_pipe if cal_pipe is not None else selected_pipe
    model_path = save_challenger_artifact(selected_name, final_pipe, model_card, model_card_path=card_path)

    # ─────────────────────────────────────────────────────────
    # 23. CHALLENGER MODEL COMPARISON
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running Champion vs Challenger comparison ...")

    challenger_result = run_challenger_comparison(
        challenger_name       = selected_name,
        challenger_f1         = float(f1_score(y_test, y_pred_sel)),
        challenger_roc_auc    = float(roc_auc_score(y_test, y_prob_sel)),
        challenger_gap        = float(abs(
            accuracy_score(y_train_fit, selected_pipe.predict(X_train_fit))
            - accuracy_score(y_test, selected_pipe.predict(X_test))
        )),
        challenger_model_path = model_path,
        challenger_threshold  = float(thr_cal),
        challenger_card_path  = card_path,
    )

    print("\n" + "=" * 60)
    print(f"CHALLENGER RESULT: {challenger_result['decision']}")
    print(f"Reason: {challenger_result['reason']}")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────
    # 24. MLFLOW
    # ─────────────────────────────────────────────────────────
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{selected_name}"
    mlflow_log_run(run_name, selected_name, final_pipe, model_card, X_train_sample=X_train_fit)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE  |  Best model: {selected_name}")
    print(f"Challenger status : {challenger_result['decision']}")
    print("=" * 60)

    return selected_name, final_pipe, model_card


# ── Helper ─────────────────────────────────────────────────

def _prob_to_band(prob: float) -> str:
    from src.config import PROPENSITY_BANDS
    for band, (low, high) in PROPENSITY_BANDS.items():
        if low <= prob < high:
            return band
    return "HIGH"


if __name__ == "__main__":
    start = time.time()
    name, model, card = run_training()
    logger.info("Finished in %.1fs", time.time() - start)
