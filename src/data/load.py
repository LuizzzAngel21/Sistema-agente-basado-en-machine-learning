"""Carga del Cardiovascular Disease Dataset (CRISP-DM Fase 2)."""
from __future__ import annotations

import pandas as pd

import config


def load_raw(path=None) -> pd.DataFrame:
    """Carga el CSV crudo de Kaggle (separador ';').

    Devuelve el DataFrame sin la columna 'id'.
    """
    path = path or config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {path}.\n"
            "Descárgalo de Kaggle (sulianova/cardiovascular-disease-dataset) "
            "y colócalo en data/raw/cardio_train.csv"
        )
    df = pd.read_csv(path, sep=config.CSV_SEP)
    return df.drop(columns=[c for c in ("id",) if c in df.columns])


def load_processed(path=None) -> pd.DataFrame:
    """Carga el dataset ya preprocesado (si existe)."""
    path = path or config.PROCESSED_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero el preprocesamiento "
            "(python main.py train)."
        )
    return pd.read_csv(path)
