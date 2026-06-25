"""CardioAgent — agente basado en utilidad (Russell & Norvig, 2020).

Implementa el ciclo  percepción → razonamiento → acción  articulando los tres
módulos del sistema bajo un esquema BDI:

    Beliefs    (Creencias)   -> Módulo 1: predicción del riesgo
    Desires    (Deseos)      -> Módulo 2: explicación de los factores (SHAP)
    Intentions (Intenciones) -> Módulo 3: recomendaciones priorizadas (utilidad)

El agente no elige una acción fija: pondera la magnitud y dirección de cada
valor SHAP para seleccionar las recomendaciones de mayor beneficio esperado.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

import config
from src.data.preprocess import add_bmi
from src.explain.shap_explainer import ShapExplainer
from src.recommend.recommender import recommend
from src.report import generator


class CardioAgent:
    def __init__(self):
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                "No hay modelo entrenado. Ejecuta primero: python main.py train"
            )
        self.model = joblib.load(config.MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
        # El calibrador da probabilidades fiables para la estratificación;
        # el SHAP se calcula sobre el modelo base (compatible con TreeExplainer).
        self.calibrator = (
            joblib.load(config.CALIBRATOR_PATH)
            if config.CALIBRATOR_PATH.exists() else None
        )
        self.explainer = ShapExplainer(self.model)

    # --------------------------- PERCEPCIÓN ----------------------------- #
    def perceive(self, patient: dict) -> dict:
        """Normaliza la entrada del paciente a las MODEL_FEATURES.

        Acepta `age_years` directamente o `age` en días. Deriva IMC si solo se
        entregan altura y peso.
        """
        p = dict(patient)
        if "age_years" not in p and "age" in p:
            p["age_years"] = round(p["age"] / 365.25, 1)
        if "bmi" not in p and {"height", "weight"} <= p.keys():
            tmp = add_bmi(pd.DataFrame([p]))
            p["bmi"] = float(tmp["bmi"].iloc[0])
        missing = [f for f in config.MODEL_FEATURES if f not in p]
        if missing:
            raise ValueError(f"Faltan variables del paciente: {missing}")
        return p

    def _scaled_frame(self, perceived: dict) -> pd.DataFrame:
        X = pd.DataFrame([{f: perceived[f] for f in config.MODEL_FEATURES}])
        X[config.CONTINUOUS_FEATURES] = self.scaler.transform(
            X[config.CONTINUOUS_FEATURES]
        )
        return X

    # --------------------------- RAZONAMIENTO --------------------------- #
    def predict(self, X) -> dict:
        """Módulo 1 — Beliefs: probabilidad, etiqueta y nivel de riesgo."""
        source = self.calibrator if self.calibrator is not None else self.model
        proba = float(source.predict_proba(X)[0, 1])
        return {
            "probability": proba,
            "label": int(proba >= 0.5),
            "risk_level": config.risk_level(proba),
            "model": self.metadata.get("best_model"),
        }

    def explain(self, X) -> dict:
        """Módulo 2 — Desires: explicación local con SHAP."""
        return self.explainer.explain_local(X)

    def decide(self, patient: dict, shap_local: dict):
        """Módulo 3 — Intentions: recomendaciones priorizadas por utilidad."""
        return recommend(patient, shap_local)

    # ----------------------------- ACCIÓN ------------------------------- #
    def run(self, patient: dict, save: bool = True, plots: bool = False) -> dict:
        """Ejecuta el ciclo completo y produce los reportes de doble perfil."""
        perceived = self.perceive(patient)
        X = self._scaled_frame(perceived)

        prediction = self.predict(X)
        shap_local = self.explain(X)
        recommendations = self.decide(perceived, shap_local)
        reports = generator.build_reports(prediction, recommendations, shap_local)

        if plots:
            self.explainer.waterfall(X)
        if save:
            generator.save_reports(reports)

        return {
            "prediction": prediction,
            "shap_local": shap_local,
            "recommendations": recommendations,
            "reports": reports,
        }
