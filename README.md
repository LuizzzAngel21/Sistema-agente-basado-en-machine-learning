# CardioAgent

Sistema **agente basado en utilidad** apoyado de *Machine Learning* para la
**evaluación y recomendación clínica del riesgo cardiovascular**.

Integrantes: Huanuco Flores, Luis Angel · Sanabria Farfán, Andrés Paolo


## 1. Descripción

CardioAgent es un agente inteligente que, a partir de los datos clínicos de un
paciente, ejecuta de forma automática un único flujo de tres etapas:

1. **Predice** el riesgo cardiovascular (clasificación supervisada).
2. **Explica** los factores determinantes de esa predicción (XAI con SHAP).
3. **Recomienda** acciones personalizadas y priorizadas, con un reporte
   diferenciado para el **paciente** (lenguaje accesible) y para el **médico**
   (lenguaje clínico, alineado a guías AHA / estándares CDC).

Se concibe como un **agente basado en utilidad** (Russell & Norvig, 2020): no
elige una acción fija, sino que pondera la magnitud y dirección de cada valor
SHAP para seleccionar las recomendaciones de mayor beneficio esperado.

```
ENTRADA            MÓDULO 1          MÓDULO 2           MÓDULO 3            SALIDA
Datos clínicos  →  Predicción   →   Explicación   →   Recomendación   →   Reporte
del paciente       del riesgo       con SHAP          personalizada       paciente + médico
                  (Beliefs)        (Desires)         (Intentions)
```

## 2. Dataset

**Cardiovascular Disease Dataset** (Kaggle · Svetlana Ulianova) — 70 000
registros, 11 variables de entrada + variable objetivo `cardio` (binaria,
clases balanceadas ~50/50). Archivo `cardio_train.csv`, separado por `;`.

Descarga: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Coloca el archivo en `data/raw/cardio_train.csv`.

| Variable      | Tipo       | Descripción                                   |
|---------------|------------|-----------------------------------------------|
| `age`         | objetiva   | edad en **días** (se convierte a años)        |
| `gender`      | objetiva   | 1 = mujer, 2 = hombre                          |
| `height`      | objetiva   | altura (cm) → se usa para derivar IMC          |
| `weight`      | objetiva   | peso (kg) → se usa para derivar IMC            |
| `ap_hi`       | objetiva   | presión arterial sistólica                    |
| `ap_lo`       | objetiva   | presión arterial diastólica                   |
| `cholesterol` | examen     | 1 = normal, 2 = alto, 3 = muy alto            |
| `gluc`        | examen     | 1 = normal, 2 = alto, 3 = muy alto            |
| `smoke`       | subjetiva  | 0/1 consumo de tabaco                         |
| `alco`        | subjetiva  | 0/1 consumo de alcohol                        |
| `active`      | subjetiva  | 0/1 actividad física                          |
| `cardio`      | objetivo   | 0 = sin ECV, 1 = con ECV                       |

## 3. Estructura del proyecto

```
CardioAgent/
├── README.md
├── requirements.txt
├── config.py                 # rutas, umbrales y constantes
├── main.py                   # CLI: train | predict | eda
├── data/
│   ├── raw/                  # cardio_train.csv (no versionado)
│   └── processed/            # datos limpios
├── models/                   # modelos entrenados (.joblib)
├── reports/
│   └── figures/              # gráficos EDA y SHAP
├── src/
│   ├── data/                 # carga y preprocesamiento (CRISP-DM E1)
│   ├── eda/                  # análisis exploratorio (CRISP-DM E2)
│   ├── models/               # entrenamiento y evaluación (E3)
│   ├── explain/              # explicabilidad SHAP (Módulo 2)
│   ├── recommend/            # reglas clínicas y recomendador (Módulo 3)
│   ├── agent/                # orquestador del agente (utilidad / BDI)
│   └── report/               # generación del reporte doble perfil
├── webapp/                   # interfaz web (FastAPI + HTML/CSS/JS)
│   ├── api.py                # API REST y servidor
│   ├── templates/            # index.html, modelo.html
│   └── static/               # styles.css, app.js, modelo.js
├── examples/                 # casos de estudio representativos
├── docs/                     # MODEL_CARD.md (limitaciones)
└── tests/
```

En `src/recommend/` el agente implementa una **función de utilidad explícita**
(`utility.py`): `U = contribución_SHAP · eficacia · factibilidad · (1 − costo)`.
El agente selecciona las acciones que la maximizan, por lo que descarta factores
no modificables (p. ej. la edad) aunque su SHAP sea alto — comportamiento propio
de un agente de utilidad, no de un clasificador.

## 4. Instalación

Requiere **Python 3.10+**.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## 5. Uso (CLI)

```bash
# 1) Análisis exploratorio (genera figuras en reports/figures/)
python main.py eda

# 2) Entrenar y comparar los 3 modelos; selecciona el mejor por AUC-ROC
python main.py train

# 3) Evaluar un paciente y generar su reporte (paciente + médico)
python main.py predict --input ejemplo_paciente.json

# 4) Interfaz web (entregable): http://127.0.0.1:8000
python main.py serve

# 5) Casos de estudio (genera reportes + PDF en reports/casos/)
python -m examples.casos_estudio
```

## 5.1 Interfaz web

`python main.py serve` levanta la aplicación web.

- **Evaluar** (`/`): formulario de datos clínicos → nivel de riesgo (medidor con
  los cortes 0.35 / 0.65), pestañas **Paciente** / **Médico** (con el desglose
  de utilidad esperada), explicación SHAP del caso y **descarga del reporte en PDF**.
- **Datos y Modelo** (`/modelo`): métricas, comparación de clasificadores,
  SHAP global y galería de figuras (EDA + Resultados) — vista de transparencia.

| Ruta              | Método | Descripción                              |
|-------------------|--------|------------------------------------------|
| `/`               | GET    | Interfaz web — evaluación                 |
| `/modelo`         | GET    | Vista de datos y modelo                   |
| `/api/predict`    | POST   | Ejecuta el ciclo del agente (JSON)       |
| `/api/report.pdf` | POST   | Reporte doble perfil en PDF              |
| `/api/meta`       | GET    | Modelo, métricas y SHAP global           |
| `/docs`           | GET    | Swagger UI (documentación de la API)     |

Ejemplo de `ejemplo_paciente.json`:

```json
{
  "age_years": 58, "gender": 2, "height": 168, "weight": 92,
  "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 2,
  "smoke": 1, "alco": 0, "active": 0
}
```

## 6. Metodología — CRISP-DM extendido

| Fase | CRISP-DM                         | Módulo del repo        |
|------|----------------------------------|------------------------|
| 1    | Comprensión del dominio          | `README`, `config`     |
| 2    | Comprensión de los datos (EDA)   | `src/eda`              |
| 3    | Preparación de datos             | `src/data/preprocess`  |
| 4    | Modelado ML                      | `src/models/train`     |
| 5    | XAI + Recomendación              | `src/explain`, `src/recommend` |
| 6    | Evaluación y entrega             | `src/models/evaluate`, `src/report` |

## 7. Stack técnico

Python 3.10+ · scikit-learn · XGBoost · SHAP · pandas · NumPy · Matplotlib · Seaborn · joblib
