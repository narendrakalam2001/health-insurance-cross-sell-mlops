# ============================================================
# GENERATE SYNTHETIC DATA — placeholder dataset generator
# ============================================================
# Produces a dataset matching the EXACT schema of the Kaggle
# "Health Insurance Cross Sell Prediction" dataset, with
# realistic feature distributions and label correlations, so
# the full pipeline can be trained and validated end-to-end
# BEFORE the real 381,109-row CSV is downloaded.
#
# Swap-in instructions:
#   1. Download train.csv from Kaggle (anmolkumar/health-insurance-cross-sell-prediction)
#   2. Place it at data/health_insurance_train_real.csv
#   3. Set env var:  CROSSSELL_DATA_PATH=data/health_insurance_train_real.csv
#   4. Re-run: python scripts/train_model.py
#
# Column names are IDENTICAL to the real Kaggle file (after
# validate_input_data()'s lowercase/underscore normalization),
# so no code changes are needed — only the data path.
# ============================================================

import numpy as np
import pandas as pd
import os

RNG = np.random.RandomState(42)
N   = 60000   # representative sample; real Kaggle file has 381,109 rows


def generate(n=N):
    gender = RNG.choice(["Male", "Female"], size=n, p=[0.54, 0.46])
    age    = np.clip(RNG.gamma(shape=3.2, scale=11, size=n) + 18, 20, 85).astype(int)

    driving_license = RNG.choice([1, 0], size=n, p=[0.998, 0.002])

    region_code = RNG.choice(np.arange(0, 53), size=n)

    previously_insured = RNG.choice([0, 1], size=n, p=[0.46, 0.54])

    vehicle_age = RNG.choice(
        ["< 1 Year", "1-2 Year", "> 2 Years"], size=n, p=[0.43, 0.52, 0.05]
    )

    # vehicle_damage correlates with previously_insured (uninsured customers
    # skew toward having had damage — matches the real dataset's pattern)
    vehicle_damage = np.where(
        previously_insured == 1,
        RNG.choice(["Yes", "No"], size=n, p=[0.02, 0.98]),
        RNG.choice(["Yes", "No"], size=n, p=[0.85, 0.15])
    )

    # annual_premium: heavy spike at the regulatory minimum (2630) + right skew
    annual_premium = np.where(
        RNG.random(n) < 0.28,
        2630.0,
        np.clip(RNG.lognormal(mean=10.35, sigma=0.35, size=n), 2630, 540000)
    )

    policy_sales_channel = RNG.choice(
        np.arange(1, 164), size=n,
        p=_channel_weights()
    ).astype(float)

    vintage = RNG.randint(10, 300, size=n)

    # ── Response label — driven by realistic logistic combination ──
    logit = (
        -4.3
        + 3.0 * ((previously_insured == 0) & (vehicle_damage == "Yes"))
        - 1.2 * (previously_insured == 1)
        + 0.015 * (age - 35)
        - 0.02 * (age > 60)
        + 0.30 * (vehicle_age == "1-2 Year")
        - 0.20 * (vehicle_age == "< 1 Year")
        + 0.0000015 * (annual_premium - 30000)
        + RNG.normal(0, 0.6, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    response = (RNG.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "Gender": gender,
        "Age": age,
        "Driving_License": driving_license,
        "Region_Code": region_code.astype(float),
        "Previously_Insured": previously_insured,
        "Vehicle_Age": vehicle_age,
        "Vehicle_Damage": vehicle_damage,
        "Annual_Premium": np.round(annual_premium, 2),
        "Policy_Sales_Channel": policy_sales_channel,
        "Vintage": vintage,
        "Response": response,
    })
    return df


def _channel_weights():
    """A few channels dominate real-world volume — mimic that skew."""
    w = np.ones(163)
    for idx in [151, 25, 123, 159, 154]:   # 0-indexed for channels 152,26,124,160,155
        w[idx] = 40
    return w / w.sum()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    out_path = os.path.join("data", "health_insurance_train_sample.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved → {out_path}")
    print(f"Shape: {df.shape}  |  Response rate: {df['Response'].mean():.4f}")
