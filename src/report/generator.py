"""Generación del reporte de salida con DOBLE PERFIL de usuario.

  * Reporte PACIENTE: lenguaje accesible, no técnico.
  * Reporte MÉDICO:   probabilidad, valores SHAP y recomendaciones clínicas
                      alineadas a guías AHA / estándares CDC (2024).
"""
from __future__ import annotations

import config


def _risk_phrase(level: str) -> str:
    return {
        "bajo": "Tu riesgo cardiovascular es BAJO. ¡Buen trabajo! Mantén tus hábitos.",
        "moderado": "Tu riesgo cardiovascular es MODERADO. Conviene tomar medidas preventivas.",
        "alto": "Tu riesgo cardiovascular es ALTO. Es importante actuar pronto y consultar a tu médico.",
    }[level]


def patient_report(prediction: dict, recommendations) -> str:
    lines = [
        "=" * 60,
        "  REPORTE PARA EL PACIENTE — CardioAgent",
        "=" * 60,
        "",
        _risk_phrase(prediction["risk_level"]),
        "",
        "Recomendaciones personalizadas para ti:",
    ]
    if recommendations:
        for i, r in enumerate(recommendations, 1):
            lines.append(f"  {i}. [{r.label}] {r.patient_text}")
    else:
        lines.append("  Sin factores de riesgo modificables prioritarios. ¡Sigue así!")
    lines += [
        "",
        "Este reporte es orientativo y no reemplaza la consulta médica.",
        "=" * 60,
    ]
    return "\n".join(lines)


def doctor_report(prediction: dict, recommendations, shap_local: dict) -> str:
    lines = [
        "=" * 60,
        "  REPORTE CLÍNICO — CardioAgent",
        "=" * 60,
        "",
        f"Probabilidad de ECV : {prediction['probability']:.3f}",
        f"Clasificación        : {'RIESGO' if prediction['label'] == 1 else 'SIN RIESGO'}",
        f"Nivel de riesgo      : {prediction['risk_level'].upper()}",
        f"Modelo               : {prediction.get('model', 'n/d')}",
        "",
        "Factores determinantes (SHAP local, orden de influencia):",
    ]
    ordered = sorted(shap_local.items(), key=lambda kv: abs(kv[1]), reverse=True)
    for feat, val in ordered:
        label = config.FEATURE_LABELS.get(feat, feat)
        direction = "↑ riesgo" if val > 0 else "↓ riesgo"
        lines.append(f"  - {label:<20} SHAP={val:+.4f}  ({direction})")

    lines += ["", "Recomendaciones priorizadas por utilidad esperada (AHA / CDC 2024):"]
    if recommendations:
        for i, r in enumerate(recommendations, 1):
            u = r.utility
            lines.append(f"  {i}. [{r.label}] {r.doctor_text}")
            lines.append(
                f"       U={u.utility:.3f}  "
                f"(contrib={u.share:.2f} · eficacia={u.efficacy:.2f} · "
                f"factibilidad={u.feasibility:.2f} · costo={u.cost:.2f})"
            )
    else:
        lines.append("  Sin indicaciones farmacológicas/derivaciones prioritarias.")
    lines += ["=" * 60]
    return "\n".join(lines)


def build_reports(prediction: dict, recommendations, shap_local: dict) -> dict:
    """Devuelve ambos reportes en un dict."""
    return {
        "patient": patient_report(prediction, recommendations),
        "doctor": doctor_report(prediction, recommendations, shap_local),
    }


def save_reports(reports: dict, prefix: str = "reporte") -> None:
    for profile, text in reports.items():
        out = config.REPORTS_DIR / f"{prefix}_{profile}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"[report] guardado: {out}")
