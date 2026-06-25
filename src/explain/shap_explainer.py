"""Módulo 2 — Explicabilidad mediante SHAP (CRISP-DM Fase 5 / Etapa 4).

Provee:
  * explicación GLOBAL: importancia media de cada variable sobre el conjunto
    de prueba (gráfico de barras / summary plot),
  * explicación LOCAL: contribución de cada variable para un paciente concreto
    (waterfall plot) — es la que alimenta el módulo de recomendación.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


class ShapExplainer:
    """Envoltorio sobre shap.Explainer que normaliza la salida a un dict
    {feature: valor_shap} para el paciente, usable por el recomendador.
    """

    def __init__(self, model, background=None):
        import shap

        self.model = model
        # TreeExplainer para RF/XGBoost; fallback genérico para el resto.
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            self.explainer = shap.Explainer(model, background)

    def _shap_values(self, X):
        sv = self.explainer.shap_values(X)
        # algunos explainers devuelven una lista por clase; tomamos la positiva
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        return np.asarray(sv)

    # ----------------------------- LOCAL -------------------------------- #
    def explain_local(self, x_row) -> dict:
        """Devuelve {feature: shap_value} para una única instancia (escalada)."""
        sv = self._shap_values(x_row)
        if sv.ndim == 2:
            sv = sv[0]
        return dict(zip(config.MODEL_FEATURES, sv.tolist()))

    def waterfall(self, x_row, out_name="shap_local.png") -> None:
        import shap

        exp = self.explainer(x_row)
        if hasattr(exp, "values") and exp.values.ndim == 3:
            exp = exp[:, :, 1]
        fig = plt.figure()
        shap.plots.waterfall(exp[0], show=False)
        out = config.FIGURES_DIR / out_name
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[shap] waterfall local guardado: {out}")

    # ----------------------------- GLOBAL ------------------------------- #
    def explain_global(self, X, out_name="shap_global.png") -> dict:
        """Importancia media |SHAP| por variable + summary plot."""
        import shap

        sv = self._shap_values(X)
        importance = dict(
            zip(config.MODEL_FEATURES, np.abs(sv).mean(axis=0).tolist())
        )
        fig = plt.figure()
        shap.summary_plot(sv, X, feature_names=config.MODEL_FEATURES,
                          plot_type="bar", show=False)
        out = config.FIGURES_DIR / out_name
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[shap] summary global guardado: {out}")
        return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
