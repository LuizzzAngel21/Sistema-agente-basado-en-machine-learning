"""Análisis Exploratorio de Datos (EDA) — CRISP-DM Fase 2 / Etapa 2.

Genera, sobre el dataset ya preprocesado:
  * distribuciones univariadas (histogramas + boxplots),
  * análisis bivariado contra la variable objetivo,
  * matriz de correlaciones.

Todas las figuras se guardan en reports/figures/.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin ventana, apto para ejecución por CLI
import matplotlib.pyplot as plt
import seaborn as sns

import config

sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> None:
    out = config.FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] figura guardada: {out}")


def plot_target_balance(df) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=config.TARGET, data=df, ax=ax)
    ax.set_title("Balance de clases (cardio)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sin ECV (0)", "Con ECV (1)"])
    _save(fig, "01_balance_clases.png")


def plot_distributions(df) -> None:
    cols = config.CONTINUOUS_FEATURES
    fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 8))
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[0, i])
        axes[0, i].set_title(config.FEATURE_LABELS.get(col, col))
        sns.boxplot(x=df[col], ax=axes[1, i])
    fig.suptitle("Distribuciones univariadas")
    _save(fig, "02_distribuciones.png")


def plot_bivariate(df) -> None:
    cols = config.CONTINUOUS_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    for ax, col in zip(axes, cols):
        sns.kdeplot(data=df, x=col, hue=config.TARGET, common_norm=False, ax=ax)
        ax.set_title(config.FEATURE_LABELS.get(col, col))
    fig.suptitle("Análisis bivariado vs. cardio")
    _save(fig, "03_bivariado.png")


def plot_correlation(df) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[config.MODEL_FEATURES + [config.TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlaciones")
    _save(fig, "04_correlaciones.png")


def run() -> None:
    """Ejecuta el EDA completo. Preprocesa si aún no existe el CSV limpio."""
    import config as cfg
    from src.data import preprocess

    if cfg.PROCESSED_CSV.exists():
        from src.data.load import load_processed
        df = load_processed()
    else:
        df = preprocess.run(save=True)

    print(f"\n[eda] Registros: {len(df)} | Variables: {df.shape[1]}")
    print(df.describe().round(2).to_string())

    plot_target_balance(df)
    plot_distributions(df)
    plot_bivariate(df)
    plot_correlation(df)
    print("[eda] EDA completo.")
