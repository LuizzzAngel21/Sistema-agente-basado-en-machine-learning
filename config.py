"""Configuración central de CardioAgent.

Reúne rutas, constantes del dataset, rangos clínicos de limpieza, columnas
del modelo y umbrales de estratificación de riesgo. Todos los módulos importan
desde aquí para mantener una única fuente de verdad (CRISP-DM, Fase 1).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = RAW_DIR / "cardio_train.csv"          # dataset original de Kaggle
PROCESSED_CSV = PROCESSED_DIR / "cardio_clean.csv"
CSV_SEP = ";"                                     # el dataset usa ';'

# Artefactos del modelo entrenado
MODEL_PATH = MODELS_DIR / "cardio_model.joblib"        # modelo base (para SHAP)
CALIBRATOR_PATH = MODELS_DIR / "calibrator.joblib"     # probabilidades calibradas
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Esquema del dataset
# --------------------------------------------------------------------------- #
TARGET = "cardio"

# Columnas tal como vienen en el CSV crudo (sin 'id').
RAW_FEATURES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
]

# Features finales que consume el modelo (tras preprocesamiento).
# Se reemplazan height/weight por la variable derivada 'bmi' y la edad en años.
MODEL_FEATURES = [
    "age_years", "gender", "ap_hi", "ap_lo", "bmi",
    "cholesterol", "gluc", "smoke", "alco", "active",
]

# Variables continuas que se estandarizan (StandardScaler).
CONTINUOUS_FEATURES = ["age_years", "ap_hi", "ap_lo", "bmi"]

# Etiquetas legibles para reportes y gráficos.
FEATURE_LABELS = {
    "age_years": "Edad",
    "gender": "Género",
    "ap_hi": "Presión sistólica",
    "ap_lo": "Presión diastólica",
    "bmi": "IMC",
    "cholesterol": "Colesterol",
    "gluc": "Glucosa",
    "smoke": "Tabaquismo",
    "alco": "Consumo de alcohol",
    "active": "Actividad física",
}

# --------------------------------------------------------------------------- #
# Rangos clínicos válidos (limpieza de outliers — guías AHA)
# Registros fuera de estos rangos se consideran errores de captura.
# --------------------------------------------------------------------------- #
CLINICAL_RANGES = {
    "ap_hi": (90, 250),    # sistólica mmHg
    "ap_lo": (60, 200),    # diastólica mmHg
    "height": (120, 220),  # cm
    "weight": (30, 200),   # kg
    "bmi": (15, 60),       # kg/m^2
}

# --------------------------------------------------------------------------- #
# Estratificación de riesgo (Módulo 1)
# probabilidad < BAJO -> bajo | [BAJO, ALTO] -> moderado | > ALTO -> alto
# --------------------------------------------------------------------------- #
RISK_LOW_MAX = 0.35
RISK_HIGH_MIN = 0.65
RISK_LEVELS = ("bajo", "moderado", "alto")

# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
GRID_CV_FOLDS = 3            # pliegues para la búsqueda en cuadrícula (más rápido)
PRIMARY_METRIC = "auc_roc"   # criterio de selección del mejor modelo
CALIBRATION_METHOD = "isotonic"

# Rejillas de hiperparámetros para GridSearchCV (búsqueda en cuadrícula).
PARAM_GRIDS = {
    "logistic_regression": {"C": [0.1, 1.0, 10.0]},
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [8, 12],
        "min_samples_leaf": [1, 5],
    },
    "xgboost": {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [300, 500],
    },
}


def risk_level(probability: float) -> str:
    """Mapea una probabilidad continua [0,1] a un nivel de riesgo."""
    if probability < RISK_LOW_MAX:
        return "bajo"
    if probability > RISK_HIGH_MIN:
        return "alto"
    return "moderado"
