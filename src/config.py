# ============================================================
# CONFIGURATION — Health Insurance Cross-Sell ML System
# ============================================================

import os

# ── Windows fix: avoid repeated "could not find physical cores" ────
# joblib/loky tries to shell out to WMIC to detect physical core count.
# WMIC is deprecated/missing on many modern Windows installs, so this
# call fails every time a parallel worker spawns, spamming warnings +
# tracebacks in the log. Setting this env var short-circuits that
# detection entirely. Harmless no-op on Linux/Mac.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE   = 42
N_JOBS         = -1

# ── Cross-validation ─────────────────────────────────────────
# NOTE: these apply to the hyperparameter SEARCH phase only (run on a
# subsample for large datasets — see HYPERPARAM_SEARCH_SAMPLE_SIZE
# below). Reduced from 5/20 → 3/10 because searching at full 5-fold/
# 20-iteration depth on a 240K+ row dataset with SMOTENC re-computed
# per fold was taking 17+ hours end-to-end. 3 folds / 10 iterations
# is still a solid, defensible search budget for a portfolio project
# and completes in a fraction of the time.
CV_FOLDS             = 3
RANDOM_SEARCH_ITERS  = 10

# ── Large-dataset hyperparameter search optimization ──────────
# If the training set exceeds this many rows, RandomizedSearchCV runs
# on a stratified subsample of this size (fast) to find the best
# hyperparameters, then the winning params are refit ONCE on the full
# training set. This is standard practice for large-scale tabular ML —
# relative hyperparameter ranking is stable across sample sizes at this
# scale, so search quality barely changes while search cost drops by
# ~5-8x (no benefit is lost on the FINAL model, which still trains on
# all available data).
HYPERPARAM_SEARCH_SAMPLE_SIZE = 40000

# ── Feature selection ────────────────────────────────────────
SELECT_K = 10

# ── Outlier clipping ─────────────────────────────────────────
CLIP_FOLD = 1.5

# ── Feature engineering thresholds ───────────────────────────
ORDINAL_UNIQUE_THRESHOLD = 10

# ── Propensity bands (probability of cross-sell response → tier) ─
PROPENSITY_BANDS = {
    "LOW":    (0.00, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH":   (0.60, 1.01),
}

# ── Business rule thresholds (BFSI cross-sell domain) ─────────
# Customers who are already vehicle-insured almost never convert
# (industry-observed base rate < 1%) — hard business rule, not ML.
PREVIOUSLY_INSURED_AUTO_SKIP = True

# A customer without a driving license cannot legally own/insure
# a vehicle — hard business rule, not ML.
REQUIRE_DRIVING_LICENSE = True

# ── PSI drift thresholds ───────────────────────────────────────
PSI_MODERATE         = 0.10    # PSI >= 0.10 → moderate drift, monitor
PSI_HIGH             = 0.20    # PSI >= 0.20 → critical drift, retrain

# ── Score monitoring alert ─────────────────────────────────────
# Base cross-sell response rate is ~12% — alert if avg predicted
# propensity drifts far above that (targeting quality degrading).
SCORE_MEAN_ALERT     = 0.25

# ── Challenger promotion gates ──────────────────────────────────
MIN_F1_IMPROVEMENT      = 0.005
MIN_ROCAUC_THRESHOLD    = 0.83   # realistic ceiling for this dataset (~0.85-0.88 SOTA)
MAX_GENERALIZATION_GAP  = 0.10

# ── Cross-sell economics (business impact) ──────────────────────
TELECALLING_COST_PER_CALL   = 45.0     # ₹ cost per outbound call (analyst-industry avg)
CROSS_SELL_COMMISSION_RATE  = 0.15     # ~15% commission on annual premium for a converted sale

# ── Paths ────────────────────────────────────────────────────
MODEL_DIR   = "crosssell_models"
METRICS_LOG = "crosssell_models/metrics_log.csv"
TESTS_DIR   = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)