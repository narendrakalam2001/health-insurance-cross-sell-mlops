# ============================================================
# HEALTH INSURANCE CROSS-SELL API — FastAPI Serving
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import logging
import time
import os
import json

from src.model_loader             import load_latest_model
from services.prediction_service  import predict_customer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Health Insurance Cross-Sell Prediction API")

# ── Load model on startup ─────────────────────────────────────
try:
    model, threshold = load_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model     = None
    threshold = 0.5


# ============================================================
# INPUT SCHEMA
# ============================================================

class CustomerInput(BaseModel):
    gender:                str      # "Male" / "Female"
    age:                    int
    driving_license:        int      # 0 / 1
    region_code:            float
    previously_insured:      int      # 0 / 1
    vehicle_age:             str      # "< 1 Year" / "1-2 Year" / "> 2 Years"
    vehicle_damage:          str      # "Yes" / "No"
    annual_premium:          float
    policy_sales_channel:    float
    vintage:                  int


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Health Insurance Cross-Sell Prediction API is live 🚀",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}

@app.get("/model_info")
def model_info():
    registry_path = "crosssell_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {"error": "Model registry not found"}


# ── Prediction endpoint ───────────────────────────────────────

@app.post("/predict")
def predict(customer: CustomerInput):

    start      = time.time()
    input_data = customer.dict()

    result     = predict_customer(model, input_data, threshold)

    # ── Log prediction ────────────────────────────────────────
    log_record = {
        "timestamp":               time.time(),
        "age":                     input_data["age"],
        "previously_insured":       input_data["previously_insured"],
        "vehicle_damage":           input_data["vehicle_damage"],
        "cross_sell_probability":   result["cross_sell_probability"],
        "propensity_band":          result["propensity_band"],
        "decision":                 result["decision"],
        "rule_triggered":           result.get("rule_triggered"),
    }

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["latency_seconds"] = round(time.time() - start, 4)

    return result
