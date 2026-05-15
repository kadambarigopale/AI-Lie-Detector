from fastapi.testclient import TestClient
from app.main import app
import pytest

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_predict_truthful(client):
    response = client.post("/predict", json={"text": "Best purchase ever, highly recommended."})
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "confidence" in json_data
    assert json_data["prediction"] in ["truthful", "deceptive"]

def test_predict_deceptive(client):
    response = client.post("/predict", json={"text": "This is completely garbage. Broke within a day of use. Do not buy."})
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data

def test_empty_text(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
