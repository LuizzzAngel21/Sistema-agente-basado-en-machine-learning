"""Pruebas de la interfaz web (FastAPI). Requieren el modelo entrenado."""
import pytest
from fastapi.testclient import TestClient

import config
from webapp.api import app

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="modelo no entrenado; ejecuta 'python main.py train'",
)

client = TestClient(app)


def test_index_ok():
    r = client.get("/")
    assert r.status_code == 200
    assert "CardioAgent" in r.text


def test_meta_ok():
    r = client.get("/api/meta")
    assert r.status_code == 200
    assert "model" in r.json()


def test_predict_high_risk():
    payload = {
        "age_years": 58, "gender": 2, "height": 168, "weight": 92,
        "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 2,
        "smoke": 1, "alco": 0, "active": 0,
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["risk_level"] in config.RISK_LEVELS
    assert 0.0 <= body["prediction"]["probability"] <= 1.0
    assert len(body["shap"]) == len(config.MODEL_FEATURES)
    assert isinstance(body["recommendations"], list)


def test_predict_validation_error():
    r = client.post("/api/predict", json={"age_years": 58})  # faltan campos
    assert r.status_code == 422


def test_report_pdf():
    payload = {
        "age_years": 58, "gender": 2, "height": 168, "weight": 92,
        "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 2,
        "smoke": 1, "alco": 0, "active": 0,
    }
    r = client.post("/api/report.pdf", json=payload)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"


def test_modelo_page():
    r = client.get("/modelo")
    assert r.status_code == 200
    assert "Datos y Modelo" in r.text
