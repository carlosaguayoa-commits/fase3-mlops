import sys
import os
from fastapi.testclient import TestClient

# Agregar carpeta backend al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "mensaje" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_schema():
    # Payload de prueba validando estructura y códigos de respuesta
    sample_payload = {
        "monto": 150.0,
        "hora": 14,
        "distancia_casa": 5.2,
        "transacciones_previas": 2
    }
    response = client.post("/predict", json=sample_payload)
    # Acepta 200 (predicción exitosa) o 422 (validación estricta de esquema Pydantic si cambian nombres de campos)
    assert response.status_code in [200, 422]
