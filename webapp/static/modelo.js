// CardioAgent — página "Datos y Modelo"
"use strict";

const FIGURES = [
  { file: "05_matriz_confusion.png", cap: "Matriz de confusión (test)" },
  { file: "06_curva_roc.png", cap: "Curva ROC" },
  { file: "07_calibracion.png", cap: "Curva de calibración" },
  { file: "shap_global.png", cap: "SHAP global (importancia de variables)" },
  { file: "01_balance_clases.png", cap: "Balance de clases" },
  { file: "02_distribuciones.png", cap: "Distribuciones univariadas" },
  { file: "03_bivariado.png", cap: "Análisis bivariado vs. cardio" },
  { file: "04_correlaciones.png", cap: "Matriz de correlaciones" },
];

const METRIC_LABELS = {
  accuracy: "Accuracy", precision: "Precision", recall: "Recall",
  f1: "F1", auc_roc: "AUC-ROC",
};

fetch("/api/meta")
  .then((r) => r.json())
  .then(render)
  .catch((e) => {
    document.getElementById("model-sub").textContent =
      "No se pudo cargar la metadata. ¿Entrenaste el modelo? (python main.py train)";
  });

function render(meta) {
  // Encabezado
  const cal = meta.calibrated ? ` · probabilidades calibradas (${meta.calibration_method})` : "";
  document.getElementById("model-sub").textContent =
    `${meta.model?.toUpperCase()}${cal} · entrenado con ${fmt(meta.n_train)} registros, ` +
    `evaluado con ${fmt(meta.n_test)}.`;

  // KPIs
  const grid = document.getElementById("kpi-grid");
  grid.innerHTML = "";
  for (const k of ["auc_roc", "accuracy", "precision", "recall", "f1"]) {
    if (meta.metrics?.[k] == null) continue;
    grid.insertAdjacentHTML("beforeend",
      `<div class="kpi"><span class="kpi-value">${meta.metrics[k].toFixed(4)}</span>` +
      `<span class="kpi-label">${METRIC_LABELS[k]}</span></div>`);
  }

  if (meta.best_params) {
    document.getElementById("model-params").textContent =
      "Hiperparámetros óptimos: " +
      Object.entries(meta.best_params).map(([k, v]) => `${k}=${v}`).join(" · ");
  }

  renderComparison(meta.all_results);
  renderGlobalShap(meta.global_shap_importance, meta.feature_labels);
  renderGallery();
}

function renderComparison(results) {
  if (!results) return;
  const cols = ["accuracy", "precision", "recall", "f1", "auc_roc"];
  const thead = document.querySelector("#cmp-table thead");
  const tbody = document.querySelector("#cmp-table tbody");
  thead.innerHTML =
    "<tr><th>Modelo</th>" + cols.map((c) => `<th>${METRIC_LABELS[c]}</th>`).join("") + "</tr>";

  const best = Object.keys(results).reduce((a, b) =>
    results[a].auc_roc >= results[b].auc_roc ? a : b);
  tbody.innerHTML = "";
  for (const [name, m] of Object.entries(results)) {
    const cls = name === best ? ' class="row-best"' : "";
    tbody.insertAdjacentHTML("beforeend",
      `<tr${cls}><td>${name}</td>` +
      cols.map((c) => `<td>${m[c].toFixed(4)}</td>`).join("") + "</tr>");
  }
}

function renderGlobalShap(importance, labels) {
  const chart = document.getElementById("global-shap");
  if (!importance || !Object.keys(importance).length) {
    chart.innerHTML = '<p class="rec-empty">SHAP global no disponible.</p>';
    return;
  }
  const entries = Object.entries(importance).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map((e) => e[1]), 1e-9);
  chart.innerHTML = "";
  for (const [feat, val] of entries) {
    const w = (val / max) * 100;
    const label = (labels && labels[feat]) || feat;
    chart.insertAdjacentHTML("beforeend",
      `<div class="shap-row">` +
      `<span class="shap-label">${label}</span>` +
      `<div class="shap-track"><span class="shap-bar-global" style="width:${w}%"></span></div>` +
      `<span class="shap-value">${val.toFixed(3)}</span></div>`);
  }
}

function renderGallery() {
  const gal = document.getElementById("gallery");
  gal.innerHTML = "";
  for (const fig of FIGURES) {
    gal.insertAdjacentHTML("beforeend",
      `<figure class="fig"><img loading="lazy" src="/figures/${fig.file}" alt="${fig.cap}" ` +
      `onerror="this.closest('.fig').style.display='none'"/>` +
      `<figcaption>${fig.cap}</figcaption></figure>`);
  }
}

function fmt(n) {
  return n == null ? "—" : n.toLocaleString("es-PE");
}
