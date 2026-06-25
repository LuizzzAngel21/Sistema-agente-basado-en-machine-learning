"""Módulo 3 — Recomendación personalizada (CRISP-DM Fase 5 / Etapa 4).

El agente basado en utilidad traduce los valores SHAP locales en acciones y
selecciona las de mayor UTILIDAD ESPERADA (ver src/recommend/utility.py).

Proceso para cada variable del paciente:
  1. la regla clínica debe activarse (valor de riesgo),
  2. el SHAP debe empujar el riesgo HACIA ARRIBA (shap > 0),
  3. se calcula su utilidad esperada y se descartan las de utilidad ≤ 0,
  4. se ordenan por utilidad descendente.

Así el agente pondera no solo la influencia (SHAP) sino la eficacia clínica, la
factibilidad y el costo de cada acción — el comportamiento propio de un agente
de utilidad, no de un simple clasificador.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.recommend.rules import RULES
from src.recommend.utility import UtilityBreakdown, action_utility, positive_shap_total


@dataclass
class Recommendation:
    feature: str
    label: str
    shap_value: float
    patient_text: str
    doctor_text: str
    utility: UtilityBreakdown

    @property
    def priority(self) -> float:
        """El agente prioriza por utilidad esperada (no por |SHAP| crudo)."""
        return self.utility.utility


def recommend(patient: dict, shap_local: dict, top_k: int = 5) -> list[Recommendation]:
    """Genera recomendaciones ordenadas por utilidad esperada.

    Args:
        patient: valores clínicos ORIGINALES (sin escalar) del paciente.
        shap_local: {feature: valor_shap} de la explicación local.
        top_k: número máximo de recomendaciones a devolver.
    """
    shap_total = positive_shap_total(shap_local)
    recs: list[Recommendation] = []

    for feature, rule in RULES.items():
        value = patient.get(feature)
        shap_val = shap_local.get(feature, 0.0)
        if value is None or not rule.triggered(value) or shap_val <= 0:
            continue

        util = action_utility(shap_val, rule, shap_total)
        if util.utility <= 0:
            continue  # el agente descarta acciones sin beneficio esperado neto

        recs.append(
            Recommendation(
                feature=feature,
                label=rule.label,
                shap_value=shap_val,
                patient_text=rule.patient,
                doctor_text=rule.doctor,
                utility=util,
            )
        )

    recs.sort(key=lambda r: r.priority, reverse=True)
    return recs[:top_k]
