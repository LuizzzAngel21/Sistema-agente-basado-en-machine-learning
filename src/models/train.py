"""Modelado y selección — CRISP-DM Fase 4 / Etapa 3.

Entrena y compara tres clasificadores con OPTIMIZACIÓN DE HIPERPARÁMETROS
mediante búsqueda en cuadrícula (GridSearchCV) y validación cruzada estratificada:
  * Regresión Logística  (línea base)
  * Random Forest        (ensemble por bagging)
  * XGBoost              (ensemble por boosting)

Tras seleccionar el mejor por AUC-ROC:
  * se CALIBRAN las probabilidades (isotónica) — clave porque el agente
    estratifica el riesgo por umbrales de probabilidad (0.35 / 0.65),
  * se generan las figuras de evaluación (matriz de confusión, ROC, calibración)
    y el SHAP global para la sección de Resultados,
  * se persisten modelo base (para SHAP), calibrador, scaler y metadata.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import config
from src.models.evaluate import compute_metrics, format_table


def _build_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_jobs=-1, random_state=config.RANDOM_STATE
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=config.RANDOM_STATE,
        )
    except ImportError:
        print("[train] xgboost no instalado; se omite ese modelo.")
    return models


def _scale(X_train, X_test):
    """Estandariza solo las variables continuas; devuelve scaler ajustado."""
    scaler = StandardScaler()
    X_train, X_test = X_train.copy(), X_test.copy()
    cont = config.CONTINUOUS_FEATURES
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])
    return X_train, X_test, scaler


def run() -> dict:
    from src.data import preprocess

    df = preprocess.run(save=True)
    X = df[config.MODEL_FEATURES]
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )
    X_train_s, X_test_s, scaler = _scale(X_train, X_test)

    grid_cv = StratifiedKFold(
        n_splits=config.GRID_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    results, fitted, best_params = {}, {}, {}
    for name, model in _build_models().items():
        print(f"\n[train] GridSearchCV: {name}...")
        gs = GridSearchCV(
            model, config.PARAM_GRIDS[name], scoring="roc_auc",
            cv=grid_cv, n_jobs=-1, refit=True,
        )
        gs.fit(X_train_s, y_train)
        est = gs.best_estimator_

        y_pred = est.predict(X_test_s)
        y_proba = est.predict_proba(X_test_s)[:, 1]
        m = compute_metrics(y_test, y_pred, y_proba)
        m["cv_auc_mean"] = round(float(gs.best_score_), 4)

        results[name] = m
        fitted[name] = est
        best_params[name] = gs.best_params_
        print(f"[train] {name}: AUC-ROC={m['auc_roc']} "
              f"(CV {m['cv_auc_mean']}) | best={gs.best_params_}")

    print("\n=== Comparación de modelos (hiperparámetros óptimos) ===")
    print(format_table(results))

    best_name = max(results, key=lambda n: results[n][config.PRIMARY_METRIC])
    best_model = fitted[best_name]
    print(f"\n[train] Modelo seleccionado: {best_name} "
          f"(AUC-ROC={results[best_name]['auc_roc']})")

    # ---- Calibración de probabilidades ----
    print(f"[train] Calibrando probabilidades ({config.CALIBRATION_METHOD})...")
    calibrator = CalibratedClassifierCV(
        clone(best_model), method=config.CALIBRATION_METHOD, cv=config.CV_FOLDS
    )
    calibrator.fit(X_train_s, y_train)
    y_proba_cal = calibrator.predict_proba(X_test_s)[:, 1]
    y_pred_cal = (y_proba_cal >= 0.5).astype(int)
    cal_metrics = compute_metrics(y_test, y_pred_cal, y_proba_cal)
    print(f"[train] Métricas calibradas: {cal_metrics}")

    # ---- Figuras de evaluación (Resultados) ----
    from src.models import diagnostics
    diagnostics.run_all(y_test, y_pred_cal, y_proba_cal)

    # ---- SHAP global sobre una muestra del test ----
    try:
        from src.explain.shap_explainer import ShapExplainer
        sample = X_test_s.sample(min(2000, len(X_test_s)), random_state=config.RANDOM_STATE)
        global_importance = ShapExplainer(best_model).explain_global(sample)
        print(f"[train] SHAP global (top 3): "
              f"{list(global_importance.items())[:3]}")
    except Exception as exc:  # pragma: no cover
        global_importance = {}
        print(f"[train] aviso: no se pudo generar SHAP global ({exc})")

    # ---- Persistencia ----
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(calibrator, config.CALIBRATOR_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    metadata = {
        "best_model": best_name,
        "best_params": best_params[best_name],
        "features": config.MODEL_FEATURES,
        "continuous_features": config.CONTINUOUS_FEATURES,
        "calibrated": True,
        "calibration_method": config.CALIBRATION_METHOD,
        "metrics": cal_metrics,
        "metrics_uncalibrated": results[best_name],
        "all_results": results,
        "all_best_params": best_params,
        "global_shap_importance": global_importance,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[train] Guardado: {config.MODEL_PATH.name}, {config.CALIBRATOR_PATH.name}, "
          f"{config.SCALER_PATH.name}, {config.METADATA_PATH.name}")
    return metadata
