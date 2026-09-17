# ============================================================
# RUN API — runner script
# ============================================================
# FIX: auto-reload is now OFF by default.
#
# WHY: uvicorn's --reload watcher monitors every .py file under the
# project folder and restarts the server whenever one changes. That
# includes monitoring/crosssell_dashboard.py and
# simulation/customer_simulator.py — files which are NOT part of the
# API at all. So editing the simulator or the dashboard (for example,
# to change the API URL) would restart the API server underneath you.
#
# That restart is expensive here: the champion ExtraTrees model is
# ~352 MB, so reloading it from disk takes longer than the simulator's
# request timeout. This is exactly the failure that was observed —
# the log shows a reload fired right after crosssell_dashboard.py was
# edited, request #1 completed on the outgoing worker, the server then
# shut down to restart, and every following request timed out while the
# model was being re-loaded.
#
# Running without reload gives a single stable process with the model
# loaded once — the same way it runs under Docker/Render. This is what
# you want for simulation runs, dashboard use, and demos.
#
# If you DO want auto-reload while actively editing API source code:
#     python scripts/run_api.py --reload
# In that mode the settings below keep non-API folders (dashboard,
# simulator, notebooks) from restarting the server.
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":

    use_reload = "--reload" in sys.argv

    if use_reload:
        print("Starting API in RELOAD mode (dev).")
        print("Watching only serving/, services/, src/ — editing the dashboard")
        print("or simulator will NOT restart this server.")
        uvicorn.run(
            "serving.crosssell_api:app",
            host        = "127.0.0.1",
            port        = 8000,
            reload      = True,
            # Only watch code the API actually uses.
            reload_dirs = ["serving", "services", "src"],
        )
    else:
        print("Starting API (stable mode — no auto-reload).")
        print("Model is loaded once and stays loaded.")
        print("Use 'python scripts/run_api.py --reload' only if editing API source code.")
        uvicorn.run(
            "serving.crosssell_api:app",
            host   = "127.0.0.1",
            port   = 8000,
            reload = False,
        )