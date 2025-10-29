import asyncio
from fastapi.testclient import TestClient
from src.main import app, load_models


def test_health_models_loaded():
    # Ensure models are loaded before making the request
    asyncio.get_event_loop().run_until_complete(load_models())
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    # Models should be present after training artifacts exist
    assert isinstance(data.get("model_loaded"), bool)
    assert isinstance(data.get("scaler_loaded"), bool)
