import os
import json
import joblib
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def test_model_artifact_exists_and_loads():
    model_path = os.path.join(BASE_DIR, 'backend', 'model.pkl')
    assert os.path.exists(model_path), "El artefacto model.pkl no existe en backend/"
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "El artefacto cargado no es un estimador válido con método predict"

def test_model_registry_governance():
    registry_path = os.path.join(BASE_DIR, 'backend', 'model_registry.json')
    assert os.path.exists(registry_path), "Falta el archivo model_registry.json"
    
    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    active = data["active_version"]
    prod_models = [m for m in data["registry"] if m["version"] == active and m["status"] == "production"]
    assert len(prod_models) == 1, "Debe existir exactamente un modelo activo en producción"
    
    metrics = prod_models[0]["metrics"]
    assert metrics["f1_score"] >= 0.90, f"F1-Score ({metrics['f1_score']}) por debajo del umbral de calidad"
    assert prod_models[0]["governance"]["pii_anonymized"] is True, "Control de privacidad PII no activo"

def test_input_data_quality():
    # Simulación de validación de datos antes de inferencia (sin valores nulos o infinitos)
    sample_features = np.array([[120.5, 10.0, 1.0, 0.0]])
    assert not np.isnan(sample_features).any(), "Los datos contienen valores NaN"
    assert not np.isinf(sample_features).any(), "Los datos contienen valores infinitos"
