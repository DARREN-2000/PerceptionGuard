/* PerceptionGuard playground.
 *
 * This is a VIEWER, not a simulator. Every frame, score, signal value and
 * diagnosis shown here was produced by the real Python pipeline in
 * scripts/build_site_data.py and verified frame-by-frame against
 * outputs/experiments.csv (0 mismatches over all 1776 frames).
 * No perception logic is reimplemented in JavaScript.
 */

const DATA_URL = "../data/playground.json";
const FRAMES = "../data/frames/";
const EPS = 1e-6;

const el = (id) => document.getElementById(id);
const pad3 = (s) => String(Math.round(s * 100)).padStart(3, "0");
const clamp01 = (v) => Math.max(0, Math.min(1, v));

let META = null;
let COND = null;
let state = { scene: null, deg: "none", sev: 0, frame: 0, playing: false };
let timer = null;

const PRESETS = [
  {
    t: "Confident but wrong",
    d: "Calibration drift: the image looks perfect, the detector stays confident, only geometry objects.",
    s: { deg: "calibration_error", sev: 1.0, frame: 12 },
  },
  {
    t: "Occlusion splits a track",
    d: "A pedestrian cuts the vehicle in two. The 3rd association pass keeps one ID.",
    s: { deg: "patch_occlusion", sev: 0.75, frame: 8 },
  },
  {
    t: "Dropped frames",
    d: "Strongest early warning: reliability 0.802 vs 0.951 clean at the LOWEST severity.",
    s: { deg: "frame_delay", sev: 0.25, frame: 14 },
  },
  {
    t: "Blur the monitor under-rates",
    d: "A known weakness: hue matching is blur-tolerant, so sharpness is anti-correlated with failure.",
    s: { deg: "motion_blur", sev: 1.0, frame: 10 },
  },
  {
    t: "Clean baseline",
    d: "No degradation. Note it still is not always TRUSTED - false-alarm rate is 0.646.",
    s: { deg: "none", sev: 0, frame: 6 },
  },
];

const key = () => `${state.scene}|${state.deg}|${pad3(state.sev)}`;
const frames = () => (COND[key()] || { frames: [] }).frames;

/* ------------------------------------------------------------------ */

async function boot() {
  let payload;
  try {
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    payload = await res.json();
  } catch (err) {
    el("app").innerHTML =
      `<div class="loading">Could not load playground data (${err.message}).<br>` +
      `Run <code>python scripts/build_site_data.py</code>, or serve this folder over HTTP ` +
      `(<code>python -m http.server</code>) - <code>file://</code> blocks fetch.</div>`;
    return;
  }
  META = payload.meta;
  COND = payload.conditions;
  state.scene = META.scenes[0];

  buildControls();
  buildPresets();
  el("app").hidden = false;
  el("boot").remove();
  render();
}

function buildControls() {
  const scene = el("scene");
  scene.innerHTML = META.scenes
    .map((s) => {
      const role = s === META.train_scene ? "train" : s === META.test_scene ? "held out" : "";
      return `<option value="${s}">${s}${role ? ` (${role})` : ""}</option>`;
    })
    .join("");
  scene.value = state.scene;

  el("deg").innerHTML = META.degradations
    .map((d) => `<option value="${d.name}">${d.name === "none" ? "none (clean)" : d.name} &middot; ${d.channel}</option>`)
    .join("");
  el("deg").value = state.deg;

  el("frame").max = META.n_frames - 1;

  scene.onchange = () => { state.scene = scene.value; render(); };
  el("deg").onchange = () => {
    state.deg = el("deg").value;
    const sevs = META.degradations.find((d) => d.name === state.deg).severities;
    state.sev = sevs.includes(state.sev) ? state.sev : sevs[sevs.length - 1];
    render();
  };
  el("sev").onchange = () => { state.sev = parseFloat(el("sev").value); render(); };
  el("frame").oninput = () => { state.frame = parseInt(el("frame").value, 10); render(); };
  el("play").onclick = togglePlay;

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
    if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
    if (e.key === " ") { togglePlay(); e.preventDefault(); }
  });
}

function buildPresets() {
  el("presets").innerHTML = PRESETS.map(
    (p, i) => `<button class="mini" data-i="${i}"><b>${p.t}</b><span>${p.d}</span></button>`
  ).join("");
  el("presets").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      Object.assign(state, PRESETS[+b.dataset.i].s);
      el("deg").value = state.deg;
      render();
    };
  });
}

function step(d) {
  state.frame = (state.frame + d + META.n_frames) % META.n_frames;
  render();
}

function togglePlay() {
  state.playing = !state.playing;
  el("play").textContent = state.playing ? "Pause" : "Play";
  clearInterval(timer);
  if (state.playing) timer = setInterval(() => step(1), 220);
}

/* ------------------------------------------------------------------ */

function render() {
  const sevs = META.degradations.find((d) => d.name === state.deg).severities;
  el("sev").disabled = state.deg === "none";
  el("sev").innerHTML = sevs.map((s) => `<option value="${s}">${s.toFixed(2)}</option>`).join("");
  el("sev").value = state.sev;

  const seq = frames();
  if (!seq.length) return;
  state.frame = Math.min(state.frame, seq.length - 1);
  const f = seq[state.frame];

  el("frame").value = state.frame;
  el("frameLabel").textContent = `${state.frame + 1} / ${META.n_frames}`;

  const url = `${FRAMES}${f.img}`;
  el("panelRgb").style.backgroundImage = `url(${url})`;
  el("panelDepth").style.backgroundImage = `url(${url})`;
  seq.forEach((g) => { new Image().src = `${FRAMES}${g.img}`; });

  renderStatus(f);
  renderVerdict(f);
  renderSignals(f);
  renderTimeline(seq, state.frame);
  renderStats(f);
}

function renderStatus(f) {
  const bar = el("statusbar");
  bar.className = `statusbar b-${f.st}`;
  bar.innerHTML =
    `<span class="word s-${f.st}">${f.st}</span>` +
    `<span class="score s-${f.st}">${f.s.toFixed(3)}</span>` +
    `<span class="meta">TRUSTED &ge; ${META.threshold_trusted} &middot; ` +
    `CAUTION &ge; ${META.threshold_caution} &middot; thresholds calibrated on the ` +
    `<code>${META.train_scene}</code> scene</span>`;
}

function renderVerdict(f) {
  const failed = f.f === 1;
  const flagged = f.st !== "TRUSTED";
  const conf = f.sg[META.signals.indexOf("confidence")];
  let verdict, cls;
  if (failed && flagged) { verdict = "Caught it."; cls = "ok"; }
  else if (!failed && !flagged) { verdict = "Correctly trusted."; cls = "ok"; }
  else if (failed && !flagged) { verdict = "MISSED - perception failed while the monitor said TRUSTED."; cls = "miss"; }
  else { verdict = "False alarm - perception was actually fine."; cls = "miss"; }

  el("verdict").className = `verdict ${cls}`;
  el("verdict").innerHTML =
    `<b>${verdict}</b> &nbsp; Ground truth this frame: ` +
    `${failed ? "<b>perception FAILED</b>" : "perception was correct"} ` +
    `(${f.tp} matched, ${f.fp} false positive${f.fp === 1 ? "" : "s"}, ${f.fn} missed of ${f.ng} visible). ` +
    `Detector confidence signal <b>${conf.toFixed(3)}</b> vs fused reliability <b>${f.s.toFixed(3)}</b>.`;
}

function renderSignals(f) {
  const rows = META.signals
    .map((n, i) => {
      const v = f.sg[i];
      const w = META.weights[n] || 0;
      return { n, v, w, c: w * -Math.log(Math.max(v, EPS)) };
    })
    .sort((a, b) => b.c - a.c || a.n.localeCompare(b.n));

  el("signals").innerHTML = rows
    .map((r) => {
      const col = r.v > 0.85 ? "var(--good)" : r.v > 0.6 ? "var(--warn)" : "var(--bad)";
      return (
        `<div class="sig${r.w === 0 ? " off" : ""}">` +
        `<div class="top"><span class="nm">${r.n}<em>w ${r.w.toFixed(3)}${r.w === 0 ? " (clamped)" : ""}</em></span>` +
        `<span class="vl">${r.v.toFixed(3)}${r.c > 0.0005 ? ` &nbsp;-${r.c.toFixed(3)}` : ""}</span></div>` +
        `<div class="bar"><i style="width:${clamp01(r.v) * 100}%;background:${col}"></i></div></div>`
      );
    })
    .join("");

  el("diagnosis").innerHTML = f.d.length
    ? f.d.map((d) => `<span class="chip">${d}</span>`).join("")
    : `<span class="chip none">no fault diagnosed</span>`;
}

function renderTimeline(seq, cur) {
  const W = 560, H = 132, P = { l: 38, r: 10, t: 12, b: 20 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const scores = seq.map((d) => d.s);
  const lo = Math.max(0, Math.min(...scores, META.threshold_caution) - 0.04);
  const hi = 1.0;
  const X = (i) => P.l + (iw * i) / Math.max(seq.length - 1, 1);
  const Y = (v) => P.t + ih * (1 - (v - lo) / (hi - lo));

  const band = (a, b, c) =>
    `<rect x="${P.l}" y="${Y(b)}" width="${iw}" height="${Math.max(Y(a) - Y(b), 0)}" fill="${c}"/>`;

  const line = seq.map((d, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(d.s).toFixed(1)}`).join("");
  const fails = seq
    .map((d, i) => (d.f ? `<rect x="${X(i) - 2}" y="${H - P.b + 4}" width="4" height="6" fill="var(--bad)"/>` : ""))
    .join("");
  const dots = seq
    .map((d, i) => `<circle cx="${X(i).toFixed(1)}" cy="${Y(d.s).toFixed(1)}" r="${i === cur ? 4.5 : 2}" fill="${i === cur ? "var(--accent)" : "#7c8798"}"/>`)
    .join("");

  el("timeline").innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none">` +
    band(META.threshold_trusted, hi, "rgba(74,222,128,.10)") +
    band(META.threshold_caution, META.threshold_trusted, "rgba(251,191,36,.10)") +
    band(lo, META.threshold_caution, "rgba(248,113,113,.10)") +
    `<line x1="${P.l}" y1="${Y(META.threshold_trusted)}" x2="${W - P.r}" y2="${Y(META.threshold_trusted)}" stroke="rgba(74,222,128,.5)" stroke-dasharray="3 3"/>` +
    `<line x1="${P.l}" y1="${Y(META.threshold_caution)}" x2="${W - P.r}" y2="${Y(META.threshold_caution)}" stroke="rgba(251,191,36,.5)" stroke-dasharray="3 3"/>` +
    `<path d="${line}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>` +
    `<line x1="${X(cur)}" y1="${P.t}" x2="${X(cur)}" y2="${H - P.b}" stroke="var(--accent)" stroke-width="1" opacity=".45"/>` +
    dots + fails +
    `<text x="4" y="${Y(hi) + 4}" fill="#98a3b3" font-size="10">${hi.toFixed(2)}</text>` +
    `<text x="4" y="${Y(lo) + 4}" fill="#98a3b3" font-size="10">${lo.toFixed(2)}</text>` +
    `<text x="${P.l}" y="${H - 4}" fill="#98a3b3" font-size="10">frame 1</text>` +
    `<text x="${W - P.r - 46}" y="${H - 4}" fill="#98a3b3" font-size="10">frame ${seq.length}</text>` +
    `</svg>`;

  el("timeline").onclick = (ev) => {
    const r = el("timeline").getBoundingClientRect();
    const rel = ((ev.clientX - r.left) / r.width) * W;
    const i = Math.round(((rel - P.l) / iw) * (seq.length - 1));
    state.frame = Math.max(0, Math.min(seq.length - 1, i));
    render();
  };
}

function renderStats(f) {
  const cells = [
    ["tracks", f.nt],
    ["visible GT", f.ng],
    ["recall", f.rec.toFixed(3)],
    ["depth err", f.de === null ? "n/a" : `${f.de.toFixed(2)} m`],
    ["frame latency", `${f.lat.toFixed(1)} ms`],
    ["monitor cost", `${f.mon.toFixed(2)} ms`],
  ];
  el("stats").innerHTML = cells
    .map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");
}

boot();
