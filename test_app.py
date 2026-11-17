from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    """Verify that the API lifespan loads the artifacts and health route returns 200."""
    with TestClient(app) as tc:
        response = tc.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    """Verify that the prediction route handles test lap telemetry correctly."""
    with TestClient(app) as tc:
        payload = {
            "position": 15,
            "lap_time_sec": 113.99
        }
        response = tc.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "is_safety_car" in data
        assert "safety_car_probability" in data