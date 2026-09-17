# ============================================================
# CUSTOMER SIMULATOR — Health Insurance Cross-Sell ML System
# ============================================================

import requests
import random
import time
import os

# ── API URL ───────────────────────────────────────────────────
# FIX: defaults to the LOCAL server now. Previously this defaulted to
# the Render URL, so running the simulator locally would silently hit
# (or fail to hit) a remote deployment instead of the server you just
# started with scripts/run_api.py.
#
# To point it at a deployed instance instead, set the env var:
#   Windows PowerShell:  $env:CROSSSELL_API_URL="https://your-app.onrender.com"
#   Linux/Mac:           export CROSSSELL_API_URL="https://your-app.onrender.com"
API_BASE = os.getenv("CROSSSELL_API_URL", "http://127.0.0.1:8000")
API_URL    = API_BASE + "/predict"
HEALTH_URL = API_BASE + "/health"

# FIX: raised from 10s. The champion ExtraTrees model is large (~352 MB),
# so the very first prediction after server start can be slow while the
# model warms up, and a remote/Render instance may be cold-starting.
REQUEST_TIMEOUT = 30


# ============================================================
# GENERATE SYNTHETIC CUSTOMER
# ============================================================

def generate_customer(scenario: str = "random") -> dict:
    """
    Scenarios:
        random      — mixed realistic customer mix
        high_intent — damaged vehicle, not previously insured (hot lead profile)
        low_intent  — already insured / no license (cold lead profile)
    """

    if scenario == "high_intent":
        return {
            "gender":               random.choice(["Male", "Female"]),
            "age":                  random.randint(28, 50),
            "driving_license":      1,
            "region_code":          float(random.randint(0, 52)),
            "previously_insured":   0,
            "vehicle_age":          random.choice(["1-2 Year", "> 2 Years"]),
            "vehicle_damage":       "Yes",
            "annual_premium":       random.uniform(25000, 45000),
            "policy_sales_channel": float(random.choice([152, 26, 124, 160])),
            "vintage":              random.randint(100, 280),
        }

    elif scenario == "low_intent":
        return {
            "gender":               random.choice(["Male", "Female"]),
            "age":                  random.randint(20, 70),
            "driving_license":      random.choice([1, 0]),
            "region_code":          float(random.randint(0, 52)),
            "previously_insured":   1,
            "vehicle_age":          random.choice(["< 1 Year", "1-2 Year"]),
            "vehicle_damage":       "No",
            "annual_premium":       random.uniform(2630, 20000),
            "policy_sales_channel": float(random.choice([160, 152, 156])),
            "vintage":              random.randint(10, 299),
        }

    else:  # random
        return {
            "gender":               random.choice(["Male", "Female"]),
            "age":                  random.randint(20, 80),
            "driving_license":      random.choices([1, 0], weights=[0.998, 0.002])[0],
            "region_code":          float(random.randint(0, 52)),
            "previously_insured":   random.choices([0, 1], weights=[0.46, 0.54])[0],
            "vehicle_age":          random.choice(["< 1 Year", "1-2 Year", "> 2 Years"]),
            "vehicle_damage":       random.choice(["Yes", "No"]),
            "annual_premium":       random.uniform(2630, 60000),
            "policy_sales_channel": float(random.randint(1, 163)),
            "vintage":              random.randint(10, 299),
        }


# ============================================================
# WARM UP — confirm the API is actually reachable first
# ============================================================

def wait_for_api(max_wait: int = 60) -> bool:
    """
    FIX: added a pre-flight health check. Without this, if the API isn't
    up (or is still loading the model) the simulator would fire all 20
    requests into a dead socket and print 20 identical timeout errors,
    which made it look like a connection/URL problem rather than a
    server-not-ready problem.
    """
    print(f"Checking API at {API_BASE} ...")
    start = time.time()

    while time.time() - start < max_wait:
        try:
            r = requests.get(HEALTH_URL, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("model_loaded"):
                    print("API is up and model is loaded.\n")
                    return True
                print("API is up but model is NOT loaded — check the server logs.")
                return False
        except Exception:
            print("  ... waiting for API to become available")
            time.sleep(3)

    print(f"\nERROR: could not reach the API at {API_BASE} within {max_wait}s.")
    print("Start it first in a SEPARATE terminal:  python scripts/run_api.py")
    return False


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_customer(customer: dict, idx: int):

    try:
        response = requests.post(API_URL, json=customer, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            result = response.json()
            print(f"[{idx+1}]  Age={customer['age']}  "
                  f"PrevInsured={customer['previously_insured']}  "
                  f"Damage={customer['vehicle_damage']}  "
                  f"→  prob={result['cross_sell_probability']:.4f}  "
                  f"band={result['propensity_band']}  "
                  f"decision={result['decision']}")
        else:
            print(f"[{idx+1}] API error: HTTP {response.status_code} — {response.text[:200]}")

    except requests.exceptions.Timeout:
        print(f"[{idx+1}] Timed out after {REQUEST_TIMEOUT}s. "
              f"If this repeats, make sure the API was started WITHOUT auto-reload "
              f"(plain 'python scripts/run_api.py').")
    except Exception as e:
        print(f"[{idx+1}] Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_customers(n: int = 20, scenario: str = "random"):

    if not wait_for_api():
        return

    print(f"Simulating {n} customers  |  scenario={scenario}\n" + "-" * 60)

    for i in range(n):
        customer = generate_customer(scenario)
        send_customer(customer, i)
        time.sleep(0.5)

    print("-" * 60 + "\nSimulation complete")


if __name__ == "__main__":
    simulate_customers(20, scenario="random")