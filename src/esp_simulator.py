import os
import time
import random
import json
from datetime import datetime, timezone

import requests


API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
BASE_URL = f"http://{API_HOST}:{API_PORT}"


def generate_mock_telemetry(seq_len: int = 12):
    # Load sequence (12 points) with mild random walk to emulate net power
    base = random.uniform(80.0, 100.0)
    seq = [base]
    for _ in range(seq_len - 1):
        seq.append(max(0.0, seq[-1] + random.uniform(-2.5, 2.5)))

    # Rich features (weather + simple loads). These are included in ingestion payload
    features = {
        "temperature": round(random.uniform(10.0, 35.0), 2),
        "humidity": round(random.uniform(20.0, 90.0), 1),
        "pressure": round(random.uniform(980.0, 1030.0), 1),
        "windSpeed": round(random.uniform(0.0, 10.0), 2),
        "cloudCover": round(random.uniform(0.0, 1.0), 2),
        "is_cloudy": int(random.random() < 0.5),
        "is_rainy": int(random.random() < 0.2),
        # Simple appliance aggregates
        "appliance_total_kw": round(random.uniform(5.0, 25.0), 2),
        "solar_kw": round(random.uniform(0.0, 5.0), 2),
    }

    status = {
        1: "on",
        2: random.choice(["on", "off"]),
        3: random.choice(["on", "off"]),
        4: random.choice(["on", "off"]),
    }
    status_str = ",".join([f"R{k}={v}" for k, v in status.items()])

    payload_optimize = {
        "time_sequence": [float(x) for x in seq],
        "current_time": float(datetime.now(tz=timezone.utc).timestamp()),
        "edge_device_status": status_str,
    }

    payload_ingest = {
        "telemetry": payload_optimize,
        "features": features,  # richer context for future mapping
    }

    return payload_ingest, payload_optimize


def safe_post(url: str, json_payload: dict, timeout_s: float = 5.0):
    try:
        return requests.post(url, json=json_payload, timeout=timeout_s)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        print("Cloud API Unavailable. Activating local Fail-Safe mode (No action taken).")
        return None


def run_simulation(iterations: int = 5, delay_s: float = 2.0):
    for i in range(iterations):
        ingest_payload, optimize_payload = generate_mock_telemetry()

        # 1) Ingest
        resp_ing = safe_post(f"{BASE_URL}/api/v1/telemetry/ingest", ingest_payload)
        if resp_ing is not None:
            print("Ingest status:", resp_ing.status_code, resp_ing.text[:120])

        # 2) Optimize
        resp_opt = safe_post(f"{BASE_URL}/api/v1/control/optimize", optimize_payload)
        if resp_opt is not None and resp_opt.ok:
            cmd = resp_opt.json()
            # 3) Actuate (simulated)
            print(
                f"Actuator: Setting Relay [{cmd.get('target_relay_id')}] to [{cmd.get('state')}] - reason={cmd.get('reason')}"
            )
        time.sleep(delay_s)


if __name__ == "__main__":
    print(f"ESP Simulator targeting {BASE_URL}")
    run_simulation(iterations=5, delay_s=2.0)


