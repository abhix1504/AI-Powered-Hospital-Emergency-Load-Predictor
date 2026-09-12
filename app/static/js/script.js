/**
 * script.js
 * ---------
 * Same contract as before: builds a raw JSON payload from the form and
 * POSTs it to /predict. No preprocessing happens here — the saved
 * scikit-learn Pipeline handles scaling and encoding server-side.
 *
 * New in this version: slider-driven inputs (with live readouts + track
 * fill), segmented controls for day/weekend/holiday, a live /health status
 * pill, an animated arc gauge for the result, and a client-side session
 * history list. None of this changes what gets sent to the backend.
 */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictionForm");
  const predictBtn = document.getElementById("predictBtn");
  const predictBtnText = document.getElementById("predictBtnText");
  const predictBtnSpinner = document.getElementById("predictBtnSpinner");
  const formError = document.getElementById("formError");
  const resetBtn = document.getElementById("resetBtn");

  const resultValue = document.getElementById("resultValue");
  const resultLoadLevel = document.getElementById("resultLoadLevel");
  const resultTimestamp = document.getElementById("resultTimestamp");
  const gaugeFill = document.getElementById("gaugeFill");

  const historySection = document.getElementById("historySection");
  const historyList = document.getElementById("historyList");

  const NUMERIC_FIELDS = [
    "hour", "month", "is_weekend", "is_holiday",
    "lag_1", "lag_2", "lag_3", "lag_24",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_24",
  ];

  const MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"];
  const WEEKEND_DAYS = ["Saturday", "Sunday"];

  // Gauge geometry matches the SVG arc path length (πr, r = 100).
  const GAUGE_ARC_LENGTH = 314;
  const GAUGE_MAX = 40; // headroom above the CRITICAL threshold (31+)

  const sessionHistory = [];

  // ------------------------------------------------------------------
  // Sliders: live readout + track fill
  // ------------------------------------------------------------------
  function formatSliderValue(input) {
    const val = Number(input.value);
    if (input.id === "hour") return `${String(val).padStart(2, "0")}:00`;
    if (input.id === "month") return MONTH_NAMES[val - 1];
    if (input.step === "0.1") return val.toFixed(1);
    return String(val);
  }

  function syncSlider(input) {
    const min = Number(input.min);
    const max = Number(input.max);
    const pct = ((Number(input.value) - min) / (max - min)) * 100;
    input.style.setProperty("--fill", `${pct}%`);

    const readout = document.getElementById(`${input.id}Value`);
    if (readout) readout.textContent = formatSliderValue(input);
  }

  form.querySelectorAll('input[type="range"]').forEach((input) => {
    syncSlider(input);
    input.addEventListener("input", () => syncSlider(input));
  });

  // ------------------------------------------------------------------
  // Segmented controls (day of week, weekend, holiday)
  // ------------------------------------------------------------------
  function wireSegmented(containerId, hiddenInputId, onChange) {
    const container = document.getElementById(containerId);
    const hiddenInput = document.getElementById(hiddenInputId);

    container.querySelectorAll(".segmented__btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        container.querySelectorAll(".segmented__btn").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        hiddenInput.value = btn.dataset.value;
        if (onChange) onChange(btn.dataset.value);
      });
    });
  }

  const weekendContainer = document.getElementById("weekendControl");
  const weekendHidden = document.getElementById("is_weekend");

  function setWeekend(value) {
    weekendHidden.value = value;
    weekendContainer.querySelectorAll(".segmented__btn").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.value === String(value));
    });
  }

  wireSegmented("dayNameControl", "day_name", (dayValue) => {
    // Convenience: derive weekend flag from the chosen day. Still
    // overridable afterwards via the weekend control itself.
    setWeekend(WEEKEND_DAYS.includes(dayValue) ? 1 : 0);
  });
  wireSegmented("weekendControl", "is_weekend");
  wireSegmented("holidayControl", "is_holiday");

  // ------------------------------------------------------------------
  // Live model status (/health)
  // ------------------------------------------------------------------
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");

  async function checkModelStatus() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      const online = Boolean(data.model_loaded);
      statusDot.className = `status-dot ${online ? "is-online" : "is-offline"}`;
      statusText.textContent = online ? "Model online" : "Model unavailable";
    } catch {
      statusDot.className = "status-dot is-offline";
      statusText.textContent = "Server unreachable";
    }
  }
  checkModelStatus();
  setInterval(checkModelStatus, 30000);

  // ------------------------------------------------------------------
  // Form helpers
  // ------------------------------------------------------------------
  function setLoading(isLoading) {
    predictBtn.disabled = isLoading;
    predictBtnSpinner.hidden = !isLoading;
    predictBtnText.textContent = isLoading ? "Running forecast…" : "Run forecast";
  }

  function showFormError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }
  function clearFormError() {
    formError.hidden = true;
    formError.textContent = "";
  }

  function collectFormData() {
    const formData = new FormData(form);
    const payload = {};
    for (const [key, rawValue] of formData.entries()) {
      if (NUMERIC_FIELDS.includes(key)) {
        const num = Number(rawValue);
        if (rawValue === "" || Number.isNaN(num)) {
          throw new Error(`"${key.replace(/_/g, " ")}" must be a valid number.`);
        }
        payload[key] = num;
      } else {
        if (!rawValue) {
          throw new Error(`Please select a value for "${key.replace(/_/g, " ")}".`);
        }
        payload[key] = rawValue;
      }
    }
    return payload;
  }

  async function toFriendlyErrorMessage(response) {
    let body = null;
    try { body = await response.json(); } catch { /* not JSON */ }

    if (response.status === 422) {
      if (body && Array.isArray(body.detail) && body.detail.length) {
        const first = body.detail[0];
        const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : "input";
        return `Invalid value for "${field}": ${first.msg}`;
      }
      return "Some of the values entered are invalid. Please check the form and try again.";
    }
    if (response.status === 400) return (body && body.detail) || "The server could not understand this request.";
    if (response.status === 500) return (body && body.detail) || "The prediction service ran into a problem. Please try again shortly.";
    return `Unexpected error (HTTP ${response.status}). Please try again.`;
  }

  // ------------------------------------------------------------------
  // Gauge + history rendering
  // ------------------------------------------------------------------
  function renderGauge(prediction, loadLevel) {
    const fraction = Math.min(Math.max(prediction / GAUGE_MAX, 0), 1);
    const offset = GAUGE_ARC_LENGTH * (1 - fraction);
    gaugeFill.style.strokeDashoffset = String(offset);
    gaugeFill.style.stroke = `var(--${loadLevel.toLowerCase()})`;
  }

  function renderHistory() {
    historySection.hidden = sessionHistory.length === 0;
    historyList.innerHTML = "";
    sessionHistory.slice(0, 5).forEach((entry) => {
      const li = document.createElement("li");
      li.className = "history__item";

      const pct = Math.min((entry.value / GAUGE_MAX) * 100, 100);
      li.innerHTML = `
        <span class="history__time">${entry.time}</span>
        <span class="history__bar-track"><span class="history__bar-fill" style="width:${pct}%; background: var(--${entry.level.toLowerCase()})"></span></span>
        <span class="history__val">${entry.value}</span>
      `;
      historyList.appendChild(li);
    });
  }

  function renderResult(data) {
    resultValue.textContent = data.prediction;
    resultLoadLevel.textContent = data.load_level;
    resultLoadLevel.className = `gauge-card__badge load--${data.load_level}`;
    resultTimestamp.textContent = new Date(data.timestamp).toLocaleString();

    renderGauge(data.prediction, data.load_level);

    sessionHistory.unshift({
      time: new Date(data.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      value: data.prediction,
      level: data.load_level,
    });
    renderHistory();
  }

  // ------------------------------------------------------------------
  // Submit / reset
  // ------------------------------------------------------------------
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearFormError();

    let payload;
    try {
      payload = collectFormData();
    } catch (validationErr) {
      showFormError(validationErr.message);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        showFormError(await toFriendlyErrorMessage(response));
        return;
      }

      renderResult(await response.json());
    } catch {
      showFormError("Could not reach the prediction service. Check your connection and try again.");
    } finally {
      setLoading(false);
    }
  });

  resetBtn.addEventListener("click", () => {
    form.reset();
    clearFormError();
    form.querySelectorAll('input[type="range"]').forEach(syncSlider);
  });
});