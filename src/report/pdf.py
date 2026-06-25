"""Reporte en PDF de doble perfil (paciente + médico).

Usa fpdf2 con fuentes core (Helvetica), por lo que el texto se sanea a Latin-1
(se sustituyen flechas/símbolos por equivalentes ASCII).
"""
from __future__ import annotations

from fpdf import FPDF

import config

NAVY = (13, 39, 71)
BLUE = (47, 111, 237)
GRAY = (107, 122, 144)
RISK_COLORS = {"bajo": (21, 163, 74), "moderado": (224, 160, 23), "alto": (220, 38, 38)}


def _s(text: str) -> str:
    """Reemplaza caracteres no Latin-1 para las fuentes core de fpdf2."""
    repl = {"↑": " (sube)", "↓": " (baja)", "→": "->",
            "•": "-", "–": "-", "—": "-", "…": "..."}
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, self.w, 22, "F")
        self.set_xy(12, 6)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, "CardioAgent")
        self.set_xy(12, 6)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 10, "Reporte de evaluacion del riesgo cardiovascular", align="R")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*GRAY)
        self.multi_cell(
            0, 4,
            _s("Herramienta de apoyo a la decision clinica. No reemplaza el "
               "diagnostico ni la consulta medica profesional."),
            align="C",
        )

    def section_title(self, text: str):
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_text_color(*BLUE)
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(self.epw, 7, _s(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(40, 40, 40)

    def body(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(self.epw, 5.5, _s(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(prediction: dict, recommendations, shap_local: dict) -> bytes:
    pdf = _Report()
    pdf.set_left_margin(12)
    pdf.set_right_margin(12)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # --- Resultado ---
    level = prediction["risk_level"]
    pdf.section_title("Resultado de la prediccion (Modulo 1)")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*RISK_COLORS.get(level, NAVY))
    pdf.multi_cell(pdf.epw, 10, _s(f"Riesgo {level.upper()}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(40, 40, 40)
    pdf.body(
        f"Probabilidad de enfermedad cardiovascular: {prediction['probability']*100:.1f}%\n"
        f"Modelo: {prediction.get('model', 'n/d')} (probabilidades calibradas)"
    )

    # --- SHAP ---
    pdf.section_title("Factores determinantes (Modulo 2 - SHAP)")
    ordered = sorted(shap_local.items(), key=lambda kv: abs(kv[1]), reverse=True)
    for feat, val in ordered:
        label = config.FEATURE_LABELS.get(feat, feat)
        arrow = "sube el riesgo" if val > 0 else "baja el riesgo"
        pdf.body(f"- {label}: SHAP={val:+.4f} ({arrow})")

    # --- Recomendaciones paciente ---
    pdf.section_title("Recomendaciones para el paciente (Modulo 3)")
    if recommendations:
        for i, r in enumerate(recommendations, 1):
            pdf.body(f"{i}. [{r.label}] {r.patient_text}")
    else:
        pdf.body("Sin factores de riesgo modificables prioritarios.")

    # --- Recomendaciones medico ---
    pdf.section_title("Recomendaciones clinicas - priorizadas por utilidad (AHA/CDC 2024)")
    if recommendations:
        for i, r in enumerate(recommendations, 1):
            u = r.utility
            pdf.body(
                f"{i}. [{r.label}] {r.doctor_text}\n"
                f"   Utilidad esperada U={u.utility:.3f} "
                f"(contrib={u.share:.2f}, eficacia={u.efficacy:.2f}, "
                f"factibilidad={u.feasibility:.2f}, costo={u.cost:.2f})"
            )
    else:
        pdf.body("Sin indicaciones farmacologicas o derivaciones prioritarias.")

    out = pdf.output()
    return bytes(out)
