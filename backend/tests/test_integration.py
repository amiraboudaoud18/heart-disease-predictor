import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def client():
    """Create test client once for all integration tests."""
    from app.main import app

    with TestClient(app) as c:
        yield c


# ── Test 1: Health endpoint returns correct structure ─────────────────────────
def test_health_endpoint(client):
    """Health endpoint must return status ok with required fields."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "model_stage" in data
    assert "model_name" in data


# ── Test 2: Predict endpoint returns valid prediction ─────────────────────────
def test_predict_endpoint(client):
    """Predict endpoint must return prediction, label and probability."""
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["High Risk", "Low Risk"]
    assert 0.0 <= data["probability"] <= 1.0
