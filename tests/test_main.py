
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "running"
    assert data["app_name"] == "AgriSmart API"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_info_endpoint():
    response = client.get("/info")

    assert response.status_code == 200

    data = response.json()
    assert data["version"] == "1.0.0"
    assert len(data["features"]) == 6
