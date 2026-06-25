"""Preprocesamiento de datos — CRISP-DM Fase 3 / Análisis Etapa 1.

Tareas:
  * conversión de edad de días a años,
  * eliminación de duplicados,
  * construcción de la variable derivada IMC,
  * filtrado de outliers clínicamente imposibles (rangos AHA),
  * selección de las columnas finales del modelo.

La estandarización (StandardScaler) se aplica en la fase de modelado, no aquí,
para evitar fuga de información entre train y test.
"""
from __future__ import annotations

import pandas as pd

import config


def add_age_years(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_years"] = (df["age"] / 365.25).round(1)
    return df


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """IMC = peso(kg) / altura(m)^2."""
    df = df.copy()
    height_m = df["height"] / 100.0
    df["bmi"] = (df["weight"] / (height_m ** 2)).round(1)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def filter_clinical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina registros fuera de los rangos clínicos válidos (config)."""
    df = df.copy()
    # presión sistólica debe ser mayor que la diastólica
    df = df[df["ap_hi"] > df["ap_lo"]]
    for col, (low, high) in config.CLINICAL_RANGES.items():
        if col in df.columns:
            df = df[(df[col] >= low) & (df[col] <= high)]
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de limpieza. Devuelve solo MODEL_FEATURES + TARGET."""
    df = add_age_years(df)
    df = add_bmi(df)
    df = remove_duplicates(df)
    df = filter_clinical_outliers(df)

    cols = config.MODEL_FEATURES + ([config.TARGET] if config.TARGET in df else [])
    return df[cols].reset_index(drop=True)


def run(save: bool = True) -> pd.DataFrame:
    """Carga el crudo, lo preprocesa y opcionalmente lo guarda."""
    from src.data.load import load_raw

    raw = load_raw()
    clean = preprocess(raw)
    if save:
        clean.to_csv(config.PROCESSED_CSV, index=False)
        print(f"[preprocess] {len(raw)} -> {len(clean)} registros "
              f"({len(raw) - len(clean)} eliminados). "
              f"Guardado en {config.PROCESSED_CSV}")
    return clean
