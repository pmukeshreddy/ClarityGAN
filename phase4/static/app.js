// ── DOM nodes ──────────────────────────────────────────────
const fileInput   = document.getElementById("file-input");
const beforeImg   = document.getElementById("before-img");
const afterImg    = document.getElementById("after-img");
const latencySpan = document.getElementById("latency");
const downloadBtn = document.getElementById("download-btn");

const slider    = document.getElementById("ba-slider");
const divider   = slider.querySelector(".divider");
const handle    = slider.querySelector(".handle");
const afterWrap = slider.querySelector(".after-wrapper");

let dragging = false;

// ── helper to position the slider ──────────────────────────
function setSlider(pct) {
  divider.style.left    = pct + "%";
  handle.style.left     = pct + "%";
  afterWrap.style.width = pct + "%";
}

// ── upload & call /deblur ──────────────────────────────────
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // show original
  const blurURL = URL.createObjectURL(file);
  beforeImg.src = blurURL;

  // prepare multipart form
  const form = new FormData();
  form.append("file", file);

  const t0  = performance.now();
  const res = await fetch("/deblur", { method: "POST", body: form });
  if (!res.ok) {
    alert("Server error: " + res.statusText);
    return;
  }
  const blob = await res.blob();
  const t1   = performance.now();

  // latency: prefer header if server provided it
  const hdr = res.headers.get("X-Processing-Time");
  latencySpan.textContent = hdr ? `${hdr}` : `${((t1 - t0) / 1000).toFixed(3)} s`;

  // show result
  const sharpURL = URL.createObjectURL(blob);
  afterImg.src = sharpURL;

  // enable slider & download button
  slider.classList.remove("hidden");
  downloadBtn.classList.remove("hidden");
  downloadBtn.href = sharpURL;
  downloadBtn.download = `deblurred_${file.name}`;

  // start divider mid‑way
  setSlider(50);
});

// ── simple drag handler for divider ────────────────────────
handle.addEventListener("pointerdown", () => (dragging = true));
window.addEventListener("pointerup",   () => (dragging = false));
window.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const rect = slider.getBoundingClientRect();
  let x = e.clientX - rect.left;
  x = Math.max(0, Math.min(x, rect.width));
  setSlider((x / rect.width) * 100);
});
