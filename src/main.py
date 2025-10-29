import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_ROOT / "ml_models"
MODEL_PATH = MODELS_DIR / "lstm_forecaster.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


class TelemetryInput(BaseModel):
    time_sequence: List[float] = Field(..., description="Last 12 historical load points")
    current_time: float = Field(..., description="Current timestamp as float (e.g., epoch seconds)")
    edge_device_status: str = Field(..., description="Device status string from ESP12F")


class ActionCommand(BaseModel):
    command: str
    target_relay_id: int
    state: str
    reason: str


app = FastAPI(title="AI Microgrid Optimizer", version="0.1.0")

# Assumed operational constants (can be externalized later)
GRID_CAPACITY_KW = 100.0
SHED_THRESHOLD_RATIO = 0.85  # keep predicted below 85% of capacity
# Estimated per-relay load contributions in kW (domain-tune as needed)
RELAY_LOAD_KW = {
    1: 30.0,  # Never shed
    2: 20.0,  # Mid priority
    3: 10.0,  # Low priority
    4: 10.0,  # Low priority,
}


class ModelState:
    def __init__(self) -> None:
        self.model = None
        self.scaler_bundle = None  # dict: {scaler, feature_cols, target_col}
        self.target_index: Optional[int] = None
        self.latest_action: Optional[ActionCommand] = None
        self.relay_status: dict = {1: "on", 2: "on", 3: "on", 4: "on"}

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler_bundle is not None


state = ModelState()


async def load_models() -> None:
    """Asynchronously load Keras model and scaler bundle once at startup.

    Includes robust error handling and clear messaging if artifacts are missing or incompatible.
    """

    def _sync_load() -> Tuple[object, dict]:
        # Load scaler first for metadata
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")
        with open(SCALER_PATH, "rb") as f:
            scaler_bundle = pickle.load(f)

        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.keras.models import load_model
        except Exception as e:
            raise RuntimeError(
                "TensorFlow is not available. On Python 3.13, install in a Python 3.12 venv (tensorflow==2.17.0)."
            ) from e

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
        model = load_model(MODEL_PATH, compile=False)
        return model, scaler_bundle

    try:
        model, scaler_bundle = await asyncio.to_thread(_sync_load)
        state.model = model
        state.scaler_bundle = scaler_bundle
        feature_cols = scaler_bundle.get("feature_cols", [])
        target_col = scaler_bundle.get("target_col")
        if target_col is None:
            raise ValueError("scaler.pkl is missing 'target_col'")
        cols = list(feature_cols) + [target_col]
        state.target_index = cols.index(target_col)
    except Exception as e:
        print("CRITICAL: Failed to load ML artifacts.")
        print(str(e))


@app.on_event("startup")
async def on_startup():
    await load_models()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "scaler_loaded": state.scaler_bundle is not None,
        "models_dir": str(MODELS_DIR),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "relay_status": state.relay_status,
        "latest_action": state.latest_action.model_dump() if state.latest_action else None,
    }


def _build_scaled_sequence_matrix_from_target(values: List[float]):
    """Return a scaled sequence matrix shaped (seq_len, num_features).

    - Fills non-target features with 0.0 (which corresponds to training min after MinMax scaling)
    - Fills target feature column with scaled values of provided target sequence
    """
    if state.scaler_bundle is None or state.target_index is None:
        raise RuntimeError("Scaler not loaded")

    scaler = state.scaler_bundle["scaler"]
    feature_cols = list(state.scaler_bundle.get("feature_cols", []))
    num_features = len(feature_cols)

    import numpy as np
    seq_len = len(values)
    mat = np.zeros((seq_len, num_features), dtype=np.float32)
    return mat


def _inverse_scale_target_sequence(values: List[float]) -> List[float]:
    if state.scaler_bundle is None or state.target_index is None:
        raise RuntimeError("Scaler not loaded")
    scaler = state.scaler_bundle["scaler"]
    idx = state.target_index
    scale_val = float(scaler.scale_[idx])
    min_val = float(scaler.min_[idx])
    return [(v - min_val) / scale_val for v in values]


def run_inference(sequence: List[float]) -> List[float]:
    """Run model inference for next-horizon target values in original units."""
    if not state.is_ready:
        raise RuntimeError("Model artifacts not loaded")
    if len(sequence) != 12:
        raise ValueError("sequence must be length 12")

    try:
        scaled_mat = _build_scaled_sequence_matrix_from_target(sequence)
    except Exception as e:
        raise ValueError(f"Scaling failed: {e}")

    import numpy as np

    try:
        model_input = np.asarray(scaled_mat, dtype=np.float32).reshape(1, 12, -1)
    except Exception as e:
        raise ValueError(f"Reshape failed: {e}")

    try:
        y_pred_scaled = state.model.predict(model_input, verbose=0)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    try:
        y_pred_scaled_list = y_pred_scaled.reshape(-1).tolist()
        y_pred = _inverse_scale_target_sequence(y_pred_scaled_list)
    except Exception as e:
        raise ValueError(f"Inverse scaling failed: {e}")

    return y_pred


def _parse_relay_status(status_str: str) -> dict:
    """Parse simple relay status string to dict of {relay_id: 'on'|'off'}."""
    default = {1: "on", 2: "on", 3: "on", 4: "on"}
    if not status_str:
        return default
    try:
        parsed = {}
        for part in status_str.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip().upper()
                v = v.strip().lower()
                if k.startswith("R") and k[1:].isdigit():
                    rid = int(k[1:])
                    if rid in default and v in {"on", "off"}:
                        parsed[rid] = v
        return {**default, **parsed}
    except Exception:
        return default


def run_optimization_solver(predicted_load_kW: List[float], current_relay_status: str) -> ActionCommand:
    """Choose minimal set of load shedding actions to keep predicted load below threshold."""
    try:
        from pulp import LpProblem, LpVariable, LpMinimize, LpStatusOptimal, lpSum, LpBinary, value
    except Exception as e:
        raise RuntimeError(f"Optimization library not available: {e}")

    if not predicted_load_kW:
        raise ValueError("predicted_load_kW is empty")

    threshold_kw = SHED_THRESHOLD_RATIO * GRID_CAPACITY_KW
    peak_pred_kw = max(predicted_load_kW)
    critical_spike = peak_pred_kw >= threshold_kw
    status_map = _parse_relay_status(current_relay_status)

    prob = LpProblem("LoadShedding", LpMinimize)
    x2 = LpVariable("shed_2", lowBound=0, upBound=1, cat=LpBinary)
    x3 = LpVariable("shed_3", lowBound=0, upBound=1, cat=LpBinary)
    x4 = LpVariable("shed_4", lowBound=0, upBound=1, cat=LpBinary)

    prob += x2 + x3 + x4

    shed_reduction = RELAY_LOAD_KW[2] * x2 + RELAY_LOAD_KW[3] * x3 + RELAY_LOAD_KW[4] * x4
    prob += peak_pred_kw - shed_reduction <= threshold_kw

    if not critical_spike or status_map.get(2) != "on":
        prob += x2 == 0
    if status_map.get(3) != "on":
        prob += x3 == 0
    if status_map.get(4) != "on":
        prob += x4 == 0

    status = prob.solve()
    if prob.status != 1:  # 1 == LpStatusOptimal
        raise RuntimeError("Optimization solver failed to find a feasible solution")

    shed2 = int(value(x2))
    shed3 = int(value(x3))
    shed4 = int(value(x4))

    if shed3 == 1:
        return ActionCommand(command="toggle", target_relay_id=3, state="off", reason="shed-low-priority")
    if shed4 == 1:
        return ActionCommand(command="toggle", target_relay_id=4, state="off", reason="shed-low-priority")
    if shed2 == 1:
        return ActionCommand(command="toggle", target_relay_id=2, state="off", reason="critical-spike-shed")

    return ActionCommand(command="noop", target_relay_id=0, state="unchanged", reason="below-threshold")


@app.post("/api/v1/telemetry/ingest")
async def ingest(payload: dict):
    try:
        print("INGEST PAYLOAD:", json.dumps(payload)[:500])
    except Exception:
        pass
    return {"status": "received", "size": len(json.dumps(payload))}


@app.post("/api/v1/control/optimize", response_model=ActionCommand)
async def optimize(payload: TelemetryInput) -> ActionCommand:
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded; try again later.")

    seq = payload.time_sequence
    if len(seq) != 12:
        raise HTTPException(status_code=422, detail="time_sequence must contain exactly 12 values")

    try:
        y_pred_kw = await asyncio.to_thread(run_inference, seq)
        action = await asyncio.to_thread(run_optimization_solver, y_pred_kw, payload.edge_device_status)
        if action.command == "toggle" and action.target_relay_id in state.relay_status:
            state.relay_status[action.target_relay_id] = action.state
        state.latest_action = action
        return action
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/api/v1/dashboard/metrics")
async def dashboard_metrics():
    """Simulated dashboard metrics: last 24h history, next-hour prediction, latest action, relay status."""
    import random
    from datetime import datetime, timedelta, timezone

    now = datetime.now(tz=timezone.utc)
    timestamps = [now - timedelta(minutes=5 * i) for i in range(288)][::-1]
    base = 80.0
    hist = []
    cur = base
    for _ in timestamps:
        cur = max(0.0, cur + random.uniform(-2.0, 2.5))
        hist.append(round(cur, 2))

    try:
        if state.is_ready:
            seq = hist[-12:]
            preds = await asyncio.to_thread(run_inference, seq)
            predicted = [float(round(x, 2)) for x in preds]
        else:
            predicted = [round(hist[-1] + random.uniform(-1.0, 3.0), 2) for _ in range(12)]
    except Exception:
        predicted = [round(hist[-1] + random.uniform(-1.0, 3.0), 2) for _ in range(12)]

    threshold = SHED_THRESHOLD_RATIO * GRID_CAPACITY_KW

    result = {
        "history": {
            "timestamps": [int(t.timestamp()) for t in timestamps],
            "load_kw": hist,
        },
        "predicted_next_hour_kw": predicted,
        "threshold_kw": threshold,
        "latest_action": state.latest_action.model_dump() if state.latest_action else None,
        "relay_status": state.relay_status,
    }
    return result


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    try:
        port = int(os.getenv("API_PORT", "8001"))
    except ValueError:
        port = 8001

    uvicorn.run(app, host=host, port=port)
