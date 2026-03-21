from fastapi.testclient import TestClient
from app.main import app

def get_client():
    return TestClient(app)


def test_health():
    with get_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_valid():
    payload = {
        "data": [
            {
                "loan_amnt": 10000,
                "annual_inc": 60000,
                "dti": 15,
                "fico_range_low": 680
            }
        ]
    }

    with get_client() as client:
        response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "request_id" in data
    assert "predictions" in data


def test_predict_empty():
    payload = {"data": []}

    with get_client() as client:
        response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 400


def test_predict_invalid():
    payload = {"data": "invalid"}

    with get_client() as client:
        response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422


def test_metrics():
    with get_client() as client:
        response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "request_count" in data
    assert "error_count" in data
    assert "avg_latency" in data