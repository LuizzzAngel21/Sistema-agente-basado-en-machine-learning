"""Casos de estudio representativos — validación del módulo de recomendación.

Ejecuta CardioAgent sobre tres perfiles clínicos contrastantes (riesgo bajo,
moderado y alto) y guarda los reportes (paciente + médico) y un PDF por caso en
reports/casos/. Cumple la validación "con casos de uso representativos" del
informe.

Uso:  python -m examples.casos_estudio
"""
from __future__ import annotations

import config
from src.agent.cardio_agent import CardioAgent
from src.report import generator
from src.report.pdf import build_pdf

CASOS = {
    "01_riesgo_bajo": {
        "descripcion": "Adulto joven, sano, activo, sin factores de riesgo.",
        "paciente": {
            "age_years": 32, "gender": 1, "height": 165, "weight": 58,
            "ap_hi": 110, "ap_lo": 70, "cholesterol": 1, "gluc": 1,
            "smoke": 0, "alco": 0, "active": 1,
        },
    },
    "02_riesgo_moderado": {
        "descripcion": "Adulto de mediana edad con sobrepeso leve y presión limítrofe.",
        "paciente": {
            "age_years": 50, "gender": 2, "height": 175, "weight": 86,
            "ap_hi": 130, "ap_lo": 85, "cholesterol": 1, "gluc": 1,
            "smoke": 0, "alco": 0, "active": 0,
        },
    },
    "03_riesgo_alto": {
        "descripcion": "Adulto mayor, fumador, hipertenso, colesterol muy alto, sedentario.",
        "paciente": {
            "age_years": 58, "gender": 2, "height": 168, "weight": 92,
            "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 2,
            "smoke": 1, "alco": 0, "active": 0,
        },
    },
}


def run() -> None:
    out_dir = config.REPORTS_DIR / "casos"
    out_dir.mkdir(parents=True, exist_ok=True)
    agent = CardioAgent()

    for nombre, caso in CASOS.items():
        result = agent.run(caso["paciente"], save=False, plots=False)
        pred = result["prediction"]
        print(f"\n### {nombre} — {caso['descripcion']}")
        print(f"    Nivel: {pred['risk_level'].upper()} "
              f"(prob={pred['probability']:.3f}) | "
              f"recomendaciones={len(result['recommendations'])}")

        reports = generator.build_reports(
            pred, result["recommendations"], result["shap_local"]
        )
        (out_dir / f"{nombre}_paciente.txt").write_text(reports["patient"], encoding="utf-8")
        (out_dir / f"{nombre}_medico.txt").write_text(reports["doctor"], encoding="utf-8")
        pdf = build_pdf(pred, result["recommendations"], result["shap_local"])
        (out_dir / f"{nombre}.pdf").write_bytes(pdf)

    print(f"\nReportes y PDFs guardados en {out_dir}")


if __name__ == "__main__":
    run()
