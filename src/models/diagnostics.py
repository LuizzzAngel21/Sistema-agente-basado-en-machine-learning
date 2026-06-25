"""Figuras de evaluación del modelo — CRISP-DM Fase 5 (sección Resultados).

Genera y guarda en reports/figures/:
  * matriz de confusión,
  * curva ROC (con AUC),
  * curva de calibración (fiabilidad de las probabilidades),
  * reporte de clasificación en texto.

La calibración es relevante porque CardioAgent estratifica el riesgo por
umbrales de probabilidad (0.35 / 0.65): si las probabilidades no están bien
calibradas, esos cortes no representan el riesgo real.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

import config


def _save(fig, name: str) -> None:
    out = config.FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[diagnostics] figura guardada: {out}")


def plot_confusion(y_true, y_pred, name="05_matriz_confusion.png") -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sin ECV", "Con ECV"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión")
    _save(fig, name)


def plot_roc(y_true, y_proba, name="06_curva_roc.png") -> None:
    fig, ax = plt.subplots(figsize=(5, 4.6))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Modelo")
    ax.plot([0, 1], [0, 1], "--", color="#9aa7b8", label="Azar")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    _save(fig, name)


def plot_calibration(y_true, y_proba, name="07_calibracion.png") -> None:
    fig, ax = plt.subplots(figsize=(5, 4.6))
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", color="#2f6fed", label="Modelo")
    ax.plot([0, 1], [0, 1], "--", color="#9aa7b8", label="Perfectamente calibrado")
    # marcas de los umbrales de estratificación
    for t in (config.RISK_LOW_MAX, config.RISK_HIGH_MIN):
        ax.axvline(t, color="#dc2626", ls=":", lw=1)
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_ylabel("Fracción real de positivos")
    ax.set_title("Curva de calibración")
    ax.legend(loc="upper left")
    _save(fig, name)


def save_classification_report(y_true, y_pred, name="classification_report.txt") -> str:
    text = classification_report(
        y_true, y_pred, target_names=["Sin ECV", "Con ECV"], digits=4
    )
    out = config.REPORTS_DIR / name
    out.write_text(text, encoding="utf-8")
    print(f"[diagnostics] reporte de clasificación guardado: {out}")
    return text


def run_all(y_true, y_pred, y_proba) -> None:
    """Genera todas las figuras y el reporte de clasificación."""
    plot_confusion(y_true, y_pred)
    plot_roc(y_true, y_proba)
    plot_calibration(y_true, y_proba)
    save_classification_report(y_true, y_pred)
