// CardioAgent — lógica de la interfaz
"use strict";

const form = document.getElementById("patient-form");
const submitBtn = document.getElementById("submit-btn");
const emptyState = document.getElementById("empty-state");
const resultsContent = document.getElementById("results-content");

let lastPayload = null; // último paciente evaluado (para la descarga PDF)

// --- Cargar metadata del modelo (nombre + métricas) ---
fetch("/api/meta")
  .then((r) => r.json())
  .then((m) => {
    const el = document.getElementById("risk-model");
    if (m.metrics) {
      el.dataset.text =
        `Modelo: ${m.model} · AUC-ROC ${m.metrics.auc_roc} · ` +
        `Accuracy ${m.metrics.accuracy}`;
    }
  })
  .catch(() => {});

// --- Envío del formulario ---
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = readForm();
  lastPayload = data;
  submitBtn.disabled = true;
  submitBtn.textContent = "Evaluando…";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Error " + res.status);
    render(await res.json());
  } catch (err) {
    alert("No se pudo evaluar: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Evaluar riesgo";
  }
});

function readForm() {
  const fd = new FormData(form);
  return {
    age_years: Number(fd.get("age_years")),
    gender: Number(fd.get("gender")),
    height: Number(fd.get("height")),
    weight: Number(fd.get("weight")),
    ap_hi: Number(fd.get("ap_hi")),
    ap_lo: Number(fd.get("ap_lo")),
    cholesterol: Number(fd.get("cholesterol")),
    gluc: Number(fd.get("gluc")),
    smoke: form.smoke.checked ? 1 : 0,
    alco: form.alco.checked ? 1 : 0,
    active: form.active.checked ? 1 : 0,
  };
}

function render(result) {
  emptyState.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  renderRisk(result.prediction);
  renderRecs(result.recommendations);
  renderShap(result.shap);
}

function renderRisk(pred) {
  const level = pred.risk_level; // bajo | moderado | alto
  const pct = Math.round(pred.probability * 100);

  const levelEl = document.getElementById("risk-level");
  levelEl.textContent = level;
  levelEl.className = "risk-level " + level;

  document.getElementById("risk-prob-value").textContent = pct + "%";
  document.getElementById("meter-marker").style.left = pct + "%";

  const modelEl = document.getElementById("risk-model");
  modelEl.textContent = modelEl.dataset.text || `Modelo: ${pred.model}`;
}

function recItem(i, rec, profile) {
  const li = document.createElement("li");
  li.className = "rec-item";
  let body =
    `<strong>${rec.label}</strong>` +
    `<span>${profile === "patient" ? rec.patient : rec.doctor}</span>`;
  // En el perfil médico mostramos el desglose de utilidad esperada del agente.
  if (profile === "doctor" && rec.utility) {
    const u = rec.utility;
    body +=
      `<span class="rec-utility">Utilidad esperada U=${u.value.toFixed(3)} ` +
      `· contrib ${u.share.toFixed(2)} · eficacia ${u.efficacy.toFixed(2)} ` +
      `· factibilidad ${u.feasibility.toFixed(2)} · costo ${u.cost.toFixed(2)}</span>`;
  }
  li.innerHTML = `<span class="rec-num">${i}</span><div class="rec-body">${body}</div>`;
  return li;
}

function renderRecs(recs) {
  for (const profile of ["patient", "doctor"]) {
    const ul = document.getElementById("rec-" + profile);
    ul.innerHTML = "";
    if (!recs.length) {
      ul.innerHTML =
        '<li class="rec-empty">Sin factores de riesgo modificables prioritarios.</li>';
      continue;
    }
    recs.forEach((r, idx) => ul.appendChild(recItem(idx + 1, r, profile)));
  }
}

function renderShap(shap) {
  const chart = document.getElementById("shap-chart");
  chart.innerHTML = "";
  const max = Math.max(...shap.map((s) => Math.abs(s.value)), 1e-6);

  for (const s of shap) {
    const up = s.value > 0;
    const width = (Math.abs(s.value) / max) * 50; // % sobre media pista
    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML =
      `<span class="shap-label">${s.label}</span>` +
      `<div class="shap-track">` +
      `<span class="shap-bar ${up ? "up" : "down"}" style="width:${width}%"></span>` +
      `</div>` +
      `<span class="shap-value">${s.value > 0 ? "+" : ""}${s.value.toFixed(3)}</span>`;
    chart.appendChild(row);
  }
}

// --- Descarga del reporte en PDF ---
document.getElementById("download-pdf").addEventListener("click", async (e) => {
  if (!lastPayload) return;
  const btn = e.currentTarget;
  btn.disabled = true;
  const prev = btn.textContent;
  btn.textContent = "Generando…";
  try {
    const res = await fetch("/api/report.pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastPayload),
    });
    if (!res.ok) throw new Error("Error " + res.status);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cardioagent_reporte.pdf";
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("No se pudo generar el PDF: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = prev;
  }
});

// --- Pestañas Paciente / Médico ---
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.tab;
    document.getElementById("tab-patient").classList.toggle("hidden", target !== "patient");
    document.getElementById("tab-doctor").classList.toggle("hidden", target !== "doctor");
  });
});
