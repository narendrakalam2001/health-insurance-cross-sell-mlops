# ============================================================
# LEAD ENGINE — Health Insurance Cross-Sell ML System
# ============================================================
# BFSI-grade 3-tier lead prioritization system:
#   HOT_LEAD   → high propensity, call immediately (top of queue)
#   WARM_LEAD  → borderline, nurture campaign (email/SMS, low-cost)
#   COLD_LEAD  → low propensity, skip / do not call
#
# Decision hierarchy:
#   1. Hard business rules (override ML)
#   2. ML model probability + propensity bands
# ============================================================

import pandas as pd
import logging

from src.config import (
    PROPENSITY_BANDS, PREVIOUSLY_INSURED_AUTO_SKIP, REQUIRE_DRIVING_LICENSE
)

logger = logging.getLogger(__name__)


# ============================================================
# PROPENSITY BAND — probability → LOW / MEDIUM / HIGH
# ============================================================

def get_propensity_band(prob: float) -> str:
    for band, (low, high) in PROPENSITY_BANDS.items():
        if low <= prob < high:
            return band
    return "HIGH"


# ============================================================
# LEAD ENGINE — row-level decisions (batch)
# ============================================================

def lead_engine(customer_df: pd.DataFrame, probs, threshold: float) -> list:
    """
    For each customer row → returns lead priority decision string.

    Rule priority:
      1. previously_insured == 1        → COLD_LEAD_RULE  (hard rule —
         industry-observed conversion < 1% for already-insured customers)
      2. driving_license == 0           → COLD_LEAD_RULE  (hard rule —
         cannot legally own/insure a vehicle without a license)
      3. prob >= threshold              → HOT_LEAD  (ML model)
      4. prob >= threshold * 0.6        → WARM_LEAD (ML model, borderline)
      5. else                           → COLD_LEAD
    """
    decisions = []

    for idx, (_, row) in enumerate(customer_df.iterrows()):

        p = probs[idx]

        # ── Hard rule: already vehicle-insured ────────────────
        if PREVIOUSLY_INSURED_AUTO_SKIP and row.get("previously_insured", 0) == 1:
            decisions.append("COLD_LEAD_RULE")
            continue

        # ── Hard rule: no driving license ─────────────────────
        if REQUIRE_DRIVING_LICENSE and row.get("driving_license", 1) == 0:
            decisions.append("COLD_LEAD_RULE")
            continue

        # ── ML model decisions ─────────────────────────────────
        if p >= threshold:
            decisions.append("HOT_LEAD_MODEL")

        elif p >= threshold * 0.6:
            decisions.append("WARM_LEAD_MODEL")

        else:
            decisions.append("COLD_LEAD_MODEL")

    return decisions


# ============================================================
# LEAD SCORING — single customer (for API)
# ============================================================

def score_customer(row: dict, prob: float, threshold: float) -> dict:
    """
    Returns structured cross-sell lead output for a single customer.
    Used by FastAPI prediction endpoint.
    """
    propensity_band = get_propensity_band(prob)

    # ── Rule-based overrides ──────────────────────────────────
    rule_triggered = None

    if PREVIOUSLY_INSURED_AUTO_SKIP and row.get("previously_insured", 0) == 1:
        decision       = "COLD_LEAD"
        rule_triggered = "ALREADY_VEHICLE_INSURED"

    elif REQUIRE_DRIVING_LICENSE and row.get("driving_license", 1) == 0:
        decision       = "COLD_LEAD"
        rule_triggered = "NO_DRIVING_LICENSE"

    # ── ML model decision ─────────────────────────────────────
    elif prob >= threshold:
        decision = "HOT_LEAD"

    elif prob >= threshold * 0.6:
        decision = "WARM_LEAD"

    else:
        decision = "COLD_LEAD"

    return {
        "cross_sell_probability": round(float(prob), 4),
        "propensity_band":        propensity_band,
        "decision":                decision,
        "rule_triggered":          rule_triggered,
    }
