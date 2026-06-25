from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import config
from src.agent.cardio_agent import CardioAgent

BASE = Path(__file__).resolve().parent

app = FastAPI(
    title="CardioAgent",
    description="Agente basado en utilidad para la evaluación y recomendación "
                "clínica del riesgo cardiovascular.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
app.mount("/figures", StaticFiles(directory=str(config.FIGURES_DIR)), name="figures")
templates = Jinja2Templates(directory=str(BASE / "templates"))

_agent: CardioAgent | None = None


def get_agent() -> CardioAgent:
    global _agent
    if _agent is None:
        _agent = CardioAgent()
    return _agent


class PatientIn(BaseModel):
    age_years: float = Field(..., ge=1, le=120, description="Edad en años")
    gender: int = Field(..., ge=1, le=2, description="1=mujer, 2=hombre")
    height: float = Field(..., ge=120, le=220, description="Altura en cm")
    weight: float = Field(..., ge=30, le=300, description="Peso en kg")
    ap_hi: int = Field(..., ge=70, le=300, description="Presión sistólica")
    ap_lo: int = Field(..., ge=40, le=200, description="Presión diastólica")
    cholesterol: int = Field(..., ge=1, le=3, description="1=normal,2=alto,3=muy alto")
    gluc: int = Field(..., ge=1, le=3, description="1=normal,2=alto,3=muy alto")
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_years": 58, "gender": 2, "height": 168, "weight": 92,
                "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 2,
                "smoke": 1, "alco": 0, "active": 0,
            }
        }
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/modelo", response_class=HTMLResponse)
def modelo(request: Request):
    return templates.TemplateResponse(request, "modelo.html")


@app.get("/api/meta")
def meta():
    """Metadata del modelo y umbrales (para mostrar en la UI)."""
    agent = get_agent()
    md = agent.metadata
    return {
        "model": md.get("best_model"),
        "best_params": md.get("best_params"),
        "metrics": md.get("metrics"),
        "metrics_uncalibrated": md.get("metrics_uncalibrated"),
        "all_results": md.get("all_results"),
        "calibrated": md.get("calibrated", False),
        "calibration_method": md.get("calibration_method"),
        "global_shap_importance": md.get("global_shap_importance"),
        "n_train": md.get("n_train"),
        "n_test": md.get("n_test"),
        "thresholds": {"low_max": config.RISK_LOW_MAX, "high_min": config.RISK_HIGH_MIN},
        "feature_labels": config.FEATURE_LABELS,
    }


@app.post("/api/predict")
def predict(patient: PatientIn):
    """Ejecuta el ciclo del agente: predicción → SHAP → recomendaciones."""
    agent = get_agent()
    result = agent.run(patient.model_dump(), save=False, plots=False)

    shap_sorted = sorted(
        result["shap_local"].items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    return {
        "prediction": result["prediction"],
        "shap": [
            {"feature": f, "label": config.FEATURE_LABELS.get(f, f),
             "value": round(float(v), 4)}
            for f, v in shap_sorted
        ],
        "recommendations": [
            {"feature": r.feature, "label": r.label,
             "shap_value": round(float(r.shap_value), 4),
             "patient": r.patient_text, "doctor": r.doctor_text,
             "utility": {
                 "value": r.utility.utility,
                 "share": r.utility.share,
                 "efficacy": r.utility.efficacy,
                 "feasibility": r.utility.feasibility,
                 "cost": r.utility.cost,
             }}
            for r in result["recommendations"]
        ],
    }


@app.post("/api/report.pdf")
def report_pdf(patient: PatientIn):
    """Genera el reporte de doble perfil en PDF descargable."""
    from src.report.pdf import build_pdf

    agent = get_agent()
    result = agent.run(patient.model_dump(), save=False, plots=False)
    pdf_bytes = build_pdf(
        result["prediction"], result["recommendations"], result["shap_local"]
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cardioagent_reporte.pdf"},
    )
