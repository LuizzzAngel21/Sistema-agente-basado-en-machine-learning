"""Métricas de evaluación de clasificadores — CRISP-DM Fase 5 / Etapa 3."""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Calcula accuracy, precision, recall, F1 y AUC-ROC."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_true, y_proba), 4),
    }


def format_table(results: dict) -> str:
    """Devuelve una tabla de texto comparando modelos."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    header = f"{'modelo':<22}" + "".join(f"{m:>11}" for m in metrics)
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(f"{name:<22}" + "".join(f"{m[k]:>11.4f}" for k in metrics))
    return "\n".join(lines)
