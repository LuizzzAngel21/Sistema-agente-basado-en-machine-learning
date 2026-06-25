"""CardioAgent — punto de entrada por línea de comandos.

Uso:
    python main.py eda                          # análisis exploratorio
    python main.py train                        # entrena y selecciona el modelo
    python main.py predict --input paciente.json
"""
from __future__ import annotations

import argparse
import json
import sys

# La consola de Windows usa cp1252 por defecto y no puede imprimir algunos
# caracteres del reporte (flechas ↑/↓). Forzamos UTF-8 en la salida.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def cmd_eda(_args) -> None:
    from src.eda import exploratory
    exploratory.run()


def cmd_train(_args) -> None:
    from src.models import train
    train.run()


def cmd_predict(args) -> None:
    from src.agent.cardio_agent import CardioAgent

    with open(args.input, encoding="utf-8") as fh:
        patient = json.load(fh)

    agent = CardioAgent()
    result = agent.run(patient, save=not args.no_save, plots=args.plots)

    print("\n" + result["reports"]["patient"])
    print("\n" + result["reports"]["doctor"])


def cmd_serve(args) -> None:
    import uvicorn
    print(f"CardioAgent disponible en http://{args.host}:{args.port}")
    uvicorn.run("webapp.api:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cardioagent",
        description="Agente basado en utilidad para riesgo cardiovascular.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("eda", help="Genera el análisis exploratorio de datos")
    sub.add_parser("train", help="Entrena y compara los modelos")

    p_pred = sub.add_parser("predict", help="Evalúa un paciente y emite reporte")
    p_pred.add_argument("--input", "-i", required=True, help="JSON del paciente")
    p_pred.add_argument("--plots", action="store_true", help="Genera waterfall SHAP")
    p_pred.add_argument("--no-save", action="store_true", help="No guardar reportes")

    p_serve = sub.add_parser("serve", help="Lanza la interfaz web (FastAPI)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Recarga en caliente")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"eda": cmd_eda, "train": cmd_train,
                "predict": cmd_predict, "serve": cmd_serve}
    handlers[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
