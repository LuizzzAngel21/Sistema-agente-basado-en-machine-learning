"""Reglas clínicas basadas en evidencia (guías AHA / estándares CDC 2024).

Cada variable del dataset tiene una regla que define:
  * `triggered(value)`: si el valor clínico del paciente es de riesgo,
  * `patient`: recomendación en lenguaje accesible (para el paciente),
  * `doctor`: recomendación clínica (para el médico tratante).

El recomendador (recommender.py) combina estas reglas con la dirección y
magnitud del valor SHAP local para priorizar las acciones del agente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ClinicalRule:
    """Regla clínica + parámetros de la función de utilidad del agente.

    Los tres coeficientes (0-1) modelan, para la acción asociada a la variable:
      * efficacy:    evidencia de que modificarla reduce el riesgo CV (AHA/CDC).
      * feasibility: cuán accionable es para el paciente (adherencia esperada).
      * cost:        carga/esfuerzo de la intervención (penalización).

    Las variables NO modificables (p. ej. edad) llevan efficacy baja, por lo que
    el agente las descarta de forma natural aunque su SHAP sea alto.
    """
    feature: str
    label: str
    triggered: Callable[[float], bool]
    patient: str
    doctor: str
    efficacy: float = 0.5
    feasibility: float = 0.5
    cost: float = 0.1


# Umbrales clínicos de referencia (AHA / CDC):
#   sistólica >=130, diastólica >=80 -> hipertensión etapa 1
#   IMC >=25 sobrepeso, >=30 obesidad
#   colesterol/glucosa: 2 = alto, 3 = muy alto
RULES = {
    "ap_hi": ClinicalRule(
        feature="ap_hi", label="Presión sistólica",
        triggered=lambda v: v >= 130,
        patient="Controla el estrés y reduce el consumo de sal en tus comidas.",
        doctor="Evaluar terapia antihipertensiva y monitoreo de presión arterial.",
        efficacy=0.90, feasibility=0.60, cost=0.10,
    ),
    "ap_lo": ClinicalRule(
        feature="ap_lo", label="Presión diastólica",
        triggered=lambda v: v >= 80,
        patient="Mantén hábitos que cuiden tu presión: menos sal y más descanso.",
        doctor="Confirmar diagnóstico de hipertensión y considerar manejo farmacológico.",
        efficacy=0.75, feasibility=0.60, cost=0.10,
    ),
    "bmi": ClinicalRule(
        feature="bmi", label="IMC elevado",
        triggered=lambda v: v >= 25,
        patient="Incorpora actividad física regular y ajusta tu dieta.",
        doctor="Referir a nutricionista y plan de control de peso.",
        efficacy=0.70, feasibility=0.40, cost=0.15,
    ),
    "cholesterol": ClinicalRule(
        feature="cholesterol", label="Colesterol alto",
        triggered=lambda v: v >= 2,
        patient="Reduce el consumo de grasas saturadas y frituras.",
        doctor="Solicitar perfil lipídico y evaluar tratamiento con estatinas.",
        efficacy=0.80, feasibility=0.60, cost=0.10,
    ),
    "gluc": ClinicalRule(
        feature="gluc", label="Glucosa elevada",
        triggered=lambda v: v >= 2,
        patient="Limita azúcares y carbohidratos refinados en tu alimentación.",
        doctor="Solicitar glucemia en ayunas / HbA1c y descartar diabetes.",
        efficacy=0.60, feasibility=0.50, cost=0.10,
    ),
    "smoke": ClinicalRule(
        feature="smoke", label="Tabaquismo",
        triggered=lambda v: v == 1,
        patient="Dejar de fumar es el cambio más beneficioso para tu corazón.",
        doctor="Ofrecer programa de cesación tabáquica y apoyo farmacológico.",
        efficacy=0.95, feasibility=0.45, cost=0.10,
    ),
    "alco": ClinicalRule(
        feature="alco", label="Consumo de alcohol",
        triggered=lambda v: v == 1,
        patient="Modera o evita el consumo de alcohol.",
        doctor="Aconsejar reducción de consumo de alcohol según guías AHA.",
        efficacy=0.50, feasibility=0.60, cost=0.05,
    ),
    "active": ClinicalRule(
        feature="active", label="Sedentarismo",
        triggered=lambda v: v == 0,   # active=0 -> sedentario
        patient="Realiza ejercicio aeróbico regular (al menos 150 min/semana).",
        doctor="Reforzar adherencia al ejercicio físico y actividad cardiovascular.",
        efficacy=0.70, feasibility=0.70, cost=0.05,
    ),
    "age_years": ClinicalRule(
        feature="age_years", label="Edad",
        triggered=lambda v: v >= 55,
        patient="A tu edad, los controles cardiovasculares periódicos son clave.",
        doctor="Intensificar tamizaje cardiovascular acorde al grupo etario.",
        # Factor NO modificable: efficacy baja -> el agente lo descarta aunque
        # su SHAP sea alto. Solo queda la acción de tamizaje (feasibility alta).
        efficacy=0.15, feasibility=0.70, cost=0.05,
    ),
}
