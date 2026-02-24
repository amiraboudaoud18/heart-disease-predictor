import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_full_patient_assessment_flow(client):
    """
    E2E test: simulates a full clinical assessment flow.
    1. Check service is healthy
    2. Submit patient data and get valid prediction
    3. Verify response structure is complete and valid
    """
    # Step 1 — service is healthy
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    # Step 2 — submit a patient
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

    # Step 3 — verify complete valid response structure
    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["High Risk", "Low Risk"]
    assert 0.0 <= data["probability"] <= 1.0
    assert (data["prediction"] == 1) == (data["label"] == "High Risk")
