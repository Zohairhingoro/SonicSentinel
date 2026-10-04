from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check() -> None:
    """Test health check GET /health returns status 200 and ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
