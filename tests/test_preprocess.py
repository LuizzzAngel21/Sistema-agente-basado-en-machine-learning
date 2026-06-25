"""Pruebas unitarias del preprocesamiento y las reglas de recomendación.

Ejecutar con:  python -m pytest   (desde la raíz del proyecto)
"""
import pandas as pd

import config
from src.data import preprocess
from src.recommend.recommender import recommend


def _sample_raw():
    return pd.DataFrame(
        {
            "age": [18393, 20228, 50000],          # 3.er registro: edad inválida
            "gender": [2, 1, 1],
            "height": [168, 156, 50],              # 3.er registro: altura inválida
            "weight": [62.0, 85.0, 70.0],
            "ap_hi": [110, 140, 120],
            "ap_lo": [80, 90, 80],
            "cholesterol": [1, 3, 1],
            "gluc": [1, 1, 1],
            "smoke": [0, 0, 0],
            "alco": [0, 0, 0],
            "active": [1, 1, 1],
            "cardio": [0, 1, 0],
        }
    )


def test_age_conversion():
    df = preprocess.add_age_years(_sample_raw())
    assert round(df["age_years"].iloc[0], 0) == 50  # 18393/365.25 ≈ 50.4


def test_bmi_derivation():
    df = preprocess.add_bmi(_sample_raw())
    # 62 / 1.68^2 ≈ 21.97
    assert 21 < df["bmi"].iloc[0] < 23


def test_outliers_removed():
    clean = preprocess.preprocess(_sample_raw())
    # el registro con altura=50 cm debe ser filtrado
    assert len(clean) == 2
    assert set(clean.columns) == set(config.MODEL_FEATURES + [config.TARGET])


def test_recommendation_priority():
    patient = {
        "age_years": 60, "gender": 2, "ap_hi": 150, "ap_lo": 95, "bmi": 32,
        "cholesterol": 3, "gluc": 1, "smoke": 1, "alco": 0, "active": 0,
    }
    shap_local = {
        "ap_hi": 0.40, "bmi": 0.25, "cholesterol": 0.10, "smoke": 0.05,
        "ap_lo": 0.02, "gluc": -0.01, "alco": 0.0, "active": 0.03,
        "age_years": 0.08, "gender": 0.0,
    }
    recs = recommend(patient, shap_local, top_k=3)
    assert recs[0].feature == "ap_hi"           # mayor utilidad esperada
    assert all(r.shap_value > 0 for r in recs)  # solo factores que suben el riesgo
    assert all(r.utility.utility > 0 for r in recs)


def test_utility_downranks_non_modifiable():
    """La edad (no modificable, efficacy baja) debe quedar por debajo de un
    factor modificable aunque su SHAP sea MAYOR — demuestra el agente de utilidad.
    """
    patient = {
        "age_years": 62, "gender": 1, "ap_hi": 145, "ap_lo": 92, "bmi": 24,
        "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 0,
    }
    shap_local = {
        "age_years": 0.50,   # SHAP alto pero NO modificable
        "ap_hi": 0.30,       # SHAP menor pero modificable y eficaz
        "ap_lo": 0.0, "bmi": 0.0, "cholesterol": 0.0, "gluc": 0.0,
        "smoke": 0.0, "alco": 0.0, "active": 0.0, "gender": 0.0,
    }
    recs = recommend(patient, shap_local, top_k=5)
    order = [r.feature for r in recs]
    assert order.index("ap_hi") < order.index("age_years")
