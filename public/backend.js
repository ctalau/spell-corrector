(function (global) {
  const STORAGE_KEY = "spell-corrector-backend";
  const BACKENDS = {
    local: {
      url: "/api/correct",
      label: "Local (Vercel GGUF)",
      running:
        "Running (first request after a cold start can take several seconds while the model loads)...",
    },
    runpod: {
      url: "/api/correct-runpod",
      label: "RunPod GPU",
      running:
        "Calling RunPod GPU (if the worker was idle this can take several minutes; if it times out, wait and retry)...",
    },
  };

  function getBackend() {
    const checked = document.querySelector('input[name="backend"]:checked');
    return checked && checked.value === "runpod" ? "runpod" : "local";
  }

  function persist(value) {
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch {
      // ignore private-mode / blocked storage
    }
  }

  function setBackend(value) {
    const next = value === "runpod" ? "runpod" : "local";
    const radio = document.querySelector(`input[name="backend"][value="${next}"]`);
    if (radio) radio.checked = true;
    persist(next);
    return next;
  }

  function initBackendSelector() {
    let saved = "local";
    try {
      saved = localStorage.getItem(STORAGE_KEY) || "local";
    } catch {
      saved = "local";
    }
    setBackend(saved);
    document.querySelectorAll('input[name="backend"]').forEach((el) => {
      el.addEventListener("change", () => setBackend(getBackend()));
    });
  }

  function runningStatus() {
    return BACKENDS[getBackend()].running;
  }

  async function postCorrection(sentence) {
    const backend = getBackend();
    const res = await fetch(BACKENDS[backend].url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  global.SpellBackend = {
    getBackend,
    setBackend,
    initBackendSelector,
    runningStatus,
    postCorrection,
  };
})(window);
