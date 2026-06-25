# Model Card — CardioAgent

Documento de transparencia del modelo predictivo que sustenta a CardioAgent,
siguiendo la práctica de *model cards* (Mitchell et al., 2019). Su objetivo es
declarar uso previsto, desempeño y **limitaciones** de forma honesta.

## 1. Detalles del modelo

- **Tarea:** clasificación binaria del riesgo cardiovascular (`cardio` 0/1).
- **Algoritmo seleccionado:** XGBoost (gradient boosting), elegido por AUC-ROC
  tras búsqueda en cuadrícula (GridSearchCV) y validación cruzada estratificada,
  comparado contra Regresión Logística y Random Forest.
- **Probabilidades calibradas** (isotónica), necesarias porque el agente
  estratifica el riesgo por umbrales de probabilidad (0.35 / 0.65).
- **Variables de entrada (10):** edad, género, presión sistólica y diastólica,
  IMC (derivado), colesterol, glucosa, tabaquismo, alcohol y actividad física.
- **Explicabilidad:** SHAP (global y local), que alimenta la función de utilidad.

## 2. Uso previsto

- **Para:** apoyo educativo y de cribado orientativo, dirigido a paciente y
  médico tratante, en el contexto de un trabajo académico.
- **NO es** un dispositivo médico ni una herramienta diagnóstica. No debe usarse
  para decisiones clínicas reales sin validación prospectiva y aprobación
  regulatoria.

## 3. Datos

- **Cardiovascular Disease Dataset** (Kaggle · Sulianova), 70 000 registros de
  chequeos; tras limpieza clínica quedan ~68 374. Clases balanceadas (~50/50).
- **Partición:** 80% entrenamiento / 20% prueba, estratificada.

## 4. Desempeño (conjunto de prueba)

| Modelo               | Accuracy | Precision | Recall | F1     | AUC-ROC |
|----------------------|----------|-----------|--------|--------|---------|
| Regresión Logística  | 0.7280   | 0.7537    | 0.6699 | 0.7093 | 0.7945  |
| Random Forest        | 0.7354   | 0.7561    | 0.6874 | 0.7201 | 0.8048  |
| **XGBoost** (elegido)| 0.7358   | 0.7538    | 0.6929 | 0.7221 | 0.8057  |

Figuras en `reports/figures/` (matriz de confusión, ROC, calibración, SHAP).

## 5. Limitaciones (importante)

1. **Techo de desempeño (~0.80 AUC).** Es coherente con la literatura sobre este
   dataset; las variables disponibles no capturan todo el riesgo cardiovascular
   (faltan antecedentes familiares, dieta detallada, biomarcadores, etc.).
2. **Datos parcialmente autorreportados.** Tabaco, alcohol y actividad física
   son subjetivos y propensos a sesgo de reporte.
3. **Calidad de la presión arterial.** El dataset original contiene valores
   imposibles (negativos, >250 mmHg); se filtran por rangos clínicos, pero
   subsiste ruido de medición.
4. **Población no peruana.** El dataset no representa necesariamente a la
   población local; la generalización a Perú no está validada.
5. **Coeficientes de utilidad heurísticos.** Los valores de eficacia,
   factibilidad y costo se basan en guías generales (AHA/CDC), no en un metaanálisis
   cuantitativo; son razonables pero perfectibles.
6. **Correlación, no causalidad.** SHAP explica la contribución al *modelo*, no
   un efecto causal clínico.

## 6. Consideraciones éticas

- Toda salida incluye un **descargo de responsabilidad** visible.
- El doble perfil evita entregar lenguaje alarmista al paciente.
- No se procesan datos personales identificables; el dataset está anonimizado.

## 7. Mantenimiento

Reentrenable con `python main.py train`. Versionado en GitHub. Revisar
calibración y métricas ante cualquier cambio de datos o de preprocesamiento.

---
*Referencia:* Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAT*.
