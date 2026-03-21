from fastapi.testclient import TestClient
from backend.app.main import app

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
                "term": "36 months",
                "emp_length": "10+ years",
                "home_ownership": "RENT",
                "annual_inc": 60000,
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "dti": 15.0,
                "delinq_2yrs": 0,
                "fico_range_low": 680,
                "fico_range_high": 684,
                "open_acc": 10,
                "pub_rec": 0,
                "revol_bal": 12000.0,
                "revol_util": 55.5,
                "total_acc": 25,
                "initial_list_status": "w",
                "application_type": "INDIVIDUAL",
                "mort_acc": 1,
                "pub_rec_bankruptcies": 0,
                "emp_title": "Engineer",
                "title": "Debt consolidation",
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