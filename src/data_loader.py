# ============================================================
# DATA LOADER + FEATURE ENGINEERING — Health Insurance Cross-Sell
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Required columns for Health Insurance Cross-Sell dataset ──
REQUIRED_COLS = [
    "gender", "age", "driving_license", "region_code", "previously_insured",
    "vehicle_age", "vehicle_damage", "annual_premium", "policy_sales_channel",
    "vintage", "response"
]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:

    # ── Normalize column names ────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.lower()
    )

    # ── Drop irrelevant ID column ─────────────────────────────
    for col in ("id",):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info("Dropped column: %s", col)

    # ── Check required columns ───────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Target validation ────────────────────────────────────
    if not set(df["response"].unique()).issubset({0, 1}):
        raise ValueError("Target 'response' must contain only 0 and 1")

    # ── Null check ───────────────────────────────────────────
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.warning("Dataset contains %d missing values — will be handled in preprocessing", nulls)

    # ── Minimum size check ───────────────────────────────────
    if df.shape[0] < 100:
        raise ValueError("Dataset too small for training (< 100 rows)")

    # ── Deduplication ────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)

    logger.info("Data validation passed  |  shape=%s  |  response_rate=%.3f",
                df.shape, df["response"].mean())

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    BFSI-grade feature engineering for health→vehicle insurance
    cross-sell propensity. All features map to real signals used
    by insurers (Bajaj Allianz / HDFC Ergo / Policybazaar style
    cross-sell scorecards).
    """
    df = df.copy()

    # ── Normalize categorical string columns ──────────────────
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.title()
    if "vehicle_age" in df.columns:
        df["vehicle_age"] = df["vehicle_age"].astype(str).str.strip()
    if "vehicle_damage" in df.columns:
        df["vehicle_damage"] = df["vehicle_damage"].astype(str).str.strip().str.title()

    # ── Binary flags from categorical text fields ─────────────
    df["vehicle_damage_flag"] = (df["vehicle_damage"] == "Yes").astype(int)
    df["is_new_vehicle"]      = (df["vehicle_age"] == "< 1 Year").astype(int)
    df["is_old_vehicle"]      = (df["vehicle_age"] == "> 2 Years").astype(int)

    # ── Strongest known cross-sell signal: damaged + uninsured ─
    df["damaged_and_uninsured"] = (
        (df["vehicle_damage_flag"] == 1) & (df["previously_insured"] == 0)
    ).astype(int)

    # ── Engagement / tenure-adjusted premium ───────────────────
    df["premium_per_vintage_day"] = df["annual_premium"] / (df["vintage"] + 1)

    # ── Age risk bins (cross-sell propensity varies by age band) ─
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(float)

    df["is_young_driver"] = (df["age"] < 25).astype(int)

    # ── Premium tier flag ───────────────────────────────────────
    premium_median = df["annual_premium"].median()
    df["high_premium_flag"] = (df["annual_premium"] > premium_median * 1.5).astype(int)

    # ── Vintage engagement bucket ────────────────────────────────
    df["long_tenure_customer"] = (df["vintage"] >= 200).astype(int)

    # ── Interaction features ─────────────────────────────────────
    df["age_x_damage"]         = df["age"] * df["vehicle_damage_flag"]
    df["prev_insured_x_new"]   = df["previously_insured"] * df["is_new_vehicle"]
    df["license_x_damage"]     = df["driving_license"] * df["vehicle_damage_flag"]

    logger.info("Feature engineering done  |  total columns=%d", df.shape[1])

    return df


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(df: pd.DataFrame, threshold: int = 10):
    """
    Auto-detect: ordinal / continuous / binary columns.
    Excludes target column 'response'.
    """
    ordinal_cols    = []
    continuous_cols = []
    binary_cols     = []

    for col in df.columns:
        if col == "response":
            continue

        dtype_name  = df[col].dtype.name
        n_unique    = df[col].nunique(dropna=False)

        if dtype_name in ("object", "category", "bool"):
            ordinal_cols.append(col)

        elif np.issubdtype(df[col].dtype, np.number):
            if n_unique == 2:
                binary_cols.append(col)
            elif 3 <= n_unique <= threshold:
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            ordinal_cols.append(col)

    logger.info("Feature types  |  ordinal=%d  continuous=%d  binary=%d",
                len(ordinal_cols), len(continuous_cols), len(binary_cols))

    return ordinal_cols, continuous_cols, binary_cols
