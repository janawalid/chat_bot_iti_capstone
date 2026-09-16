from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_happy_path():
    fake_results = {
        "documents": [["Overfitting happens when a model learns noise in the training data."]],
        "metadatas": [[{"source": "Overfitting.txt"}]],
    }
    with patch("app.api.routes.query.retrieval.is_ready", return_value=True), \
         patch("app.api.routes.query.retrieval.retrieve", return_value=fake_results), \
         patch(
             "app.api.routes.query.generation.generate_answer",
             return_value=("Overfitting is when a model fits noise instead of signal.", ["Overfitting.txt"]),
         ):
        response = client.post("/query", json={"question": "What is overfitting?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert body["sources"] == ["Overfitting.txt"]


def test_query_invalid_input():
    # Missing required "question" field should trigger a 422 validation error
    response = client.post("/query", json={})
    assert response.status_code == 422
