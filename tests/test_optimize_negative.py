import asyncio
from fastapi.testclient import TestClient
from src.main import app, load_models


def test_optimize_bad_sequence_length_422():
    asyncio.get_event_loop().run_until_complete(load_models())
    client = TestClient(app)
    payload = {
        "time_sequence": [float(i) for i in range(11)],  # should be 12
        "current_time": 1730131200.0,
        "edge_device_status": "R1=on,R2=on,R3=on,R4=on",
    }
    resp = client.post("/api/v1/control/optimize", json=payload)
    assert resp.status_code == 422
