"""Función de utilidad del agente (Russell & Norvig, 2020, cap. 16).

Un agente basado en utilidad no ejecuta una acción fija ante un estado: evalúa
las acciones candidatas y selecciona las que maximizan su utilidad esperada.

Aquí cada acción candidata es "intervenir sobre la variable X". Su utilidad se
define como:

    U(acción) = share_SHAP · efficacy · feasibility · (1 − cost)

donde:
  * share_SHAP  = contribución relativa de la variable a EMPUJAR el riesgo hacia
                  arriba (valor SHAP positivo normalizado sobre el total de
                  presión de riesgo del paciente). Es la "ganancia potencial".
  * efficacy    = evidencia clínica de que modificar la variable reduce el riesgo.
  * feasibility = probabilidad de adherencia / accionabilidad para el paciente.
  * cost        = carga/esfuerzo de la intervención, como descuento en [0,1].

Las variables con SHAP ≤ 0 (que NO empujan el riesgo) obtienen utilidad nula y
el agente las descarta. Las no modificables (efficacy baja, p. ej. la edad)
quedan al fondo del ranking aunque su SHAP sea alto.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.recommend.rules import ClinicalRule


@dataclass
class UtilityBreakdown:
    """Desglose transparente del cálculo de utilidad de una acción."""
    share: float        # contribución SHAP normalizada (ganancia potencial)
    efficacy: float
    feasibility: float
    cost: float
    expected_reduction: float  # share · efficacy
    utility: float             # valor final que el agente maximiza


def positive_shap_total(shap_local: dict) -> float:
    """Suma de las contribuciones SHAP que empujan el riesgo hacia arriba."""
    total = sum(v for v in shap_local.values() if v > 0)
    return total if total > 0 else 1e-9


def action_utility(
    shap_value: float, rule: ClinicalRule, shap_total: float
) -> UtilityBreakdown:
    """Calcula la utilidad esperada de intervenir sobre una variable."""
    share = max(shap_value, 0.0) / shap_total
    expected_reduction = share * rule.efficacy
    utility = expected_reduction * rule.feasibility * (1.0 - rule.cost)
    return UtilityBreakdown(
        share=round(share, 4),
        efficacy=rule.efficacy,
        feasibility=rule.feasibility,
        cost=rule.cost,
        expected_reduction=round(expected_reduction, 4),
        utility=round(utility, 4),
    )
