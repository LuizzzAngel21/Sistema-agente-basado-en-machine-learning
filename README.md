CardioAgent
Proyecto de Ingeniería de Sistemas de Información 

Integrantes:
Huanuco Flores, Luis Angel
Sanabria Farfán, Andrés Paolo

Acerca del proyecto
CardioAgent es un agente inteligente basado en utilidad (Russell & Norvig) diseñado para evaluar el riesgo cardiovascular. A diferencia de un clasificador tradicional, el sistema no solo predice, sino que usa Machine Learning (XGBoost + SHAP) para ponderar qué factores del paciente son accionables y recomendar cambios de hábitos que maximicen el beneficio esperado.

Cuenta con una función de utilidad propia: U = contribución_SHAP * eficacia * factibilidad * (1 - costo), lo que le permite descartar variables no modificables (como la edad) y emitir un reporte doble (uno técnico para el médico y uno accesible para el paciente).

Dataset
Usamos el Cardiovascular Disease Dataset de Kaggle (70k registros, clases balanceadas).
Para que el código funcione, debes descargar el archivo cardio_train.csv desde https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data y guardarlo en la carpeta data/raw/ (esta carpeta está ignorada en git por el peso).

Cómo ejecutarlo
Requiere Python 3.10 o superior.

1. Preparar el entorno:

Bash
python -m venv .venv
.venv\Scripts\activate  # En Mac/Linux usa: source .venv/bin/activate
pip install -r requirements.txt
2. Comandos principales (CLI):
El orquestador de todo el proyecto es main.py.

Bash
# Correr el análisis exploratorio (guarda los gráficos en reports/figures/)
python main.py eda

# Entrenar los modelos y guardar el mejor por AUC-ROC
python main.py train

# Evaluar un paciente de prueba
python main.py predict --input ejemplo_paciente.json

# Levantar la interfaz web en localhost:8000
python main.py serve
Arquitectura del repositorio
La metodología sigue las fases de CRISP-DM adaptadas a la estructura de carpetas:

data/: Datos crudos y procesados.

src/: Lógica principal dividida por módulos (data, eda, models, explain, recommend, agent, report).

webapp/: Frontend y API hecha con FastAPI.

models/: Archivos .joblib generados tras el entrenamiento.

reports/: Donde el agente escupe los PDFs finales y los gráficos.

examples/: Casos de estudio y pruebas rápidas.
