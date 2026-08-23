<div align="center">

# PerceptionGuard

**Reliable 3D perception and runtime reliability monitoring for autonomous systems.**

[![CI](https://github.com/USER/perceptionguard/actions/workflows/ci.yml/badge.svg)](../../actions)
[![Tests](https://img.shields.io/badge/tests-53%20passing-4ade80)](tests/)
[![Held-out AUROC](https://img.shields.io/badge/held--out%20AUROC-0.984-4ade80)](#1-does-fusing-signals-beat-raw-detector-confidence)
[![Latency](https://img.shields.io/badge/44.7%20ms%20%C2%B7%2022.3%20FPS-2%20vCPU%2C%20no%20GPU-5eead4)](#what-does-it-cost)
[![License](https://img.shields.io/badge/license-MIT-98a3b3)](LICENSE)

### [→ Live site &amp; interactive playground](https://USER.github.io/perceptionguard/)

</div>

---

The question this project answers is not "can I detect objects?" It is:

> **Can an autonomous system know, at runtime, whether its own perception can currently be trusted?**

A detector that is wrong is survivable. A detector that is wrong *and confident* is not.
PerceptionGuard runs a full RGB-D perception pipeline and, in parallel, scores its own
trustworthiness from 13 independent signals — then says **why** it distrusts itself.

```
# calibration drift, severity 1.0 — the image looks perfect
PERCEPTION STATUS: DEGRADED
  reliability      0.31   TRUSTED >= 0.964 | CAUTION >= 0.940 | DEGRADED below
  detector conf    0.94   <- high! the detector is confident and wrong
  geometry         0.22   <- reprojection disagrees with the class-height prior
  motion           0.34   <- 3D positions jumping between frames
  likely cause     Geometric inconsistency (calibration or depth scale error)
```

That example is the point of the whole repository: **detector confidence was 0.94 while
perception was broken.** Confidence alone cannot see calibration drift, because the
detector never looks at the calibration.

## Contents

[Headline results](#headline-results) · [Honest findings](#honest-findings-including-the-ones-that-look-bad) ·
[What it cannot detect](#what-this-system-cannot-detect) · [Architecture](#architecture) ·
[Quickstart](#quickstart) · [Playground](#the-playground) · [Why synthetic data](#why-synthetic-data-and-where-that-costs-me) ·
[Repository layout](#repository-layout) · [Status](#status)

---

## Headline results

Swept **1,776 frames** = 2 scenes × 24 frames × (**9 degradations × 4 severities** + clean).
Measured perception failure rate **0.350**. Every number below is measured; none is estimated.

### 1. Does fusing signals beat raw detector confidence?

Fit on the `multi` scene, evaluated on the **held-out** `approach` scene. Splitting by scene
rather than by frame matters: consecutive frames are near-duplicates, and a random split
would leak.

| Failure predictor | AUROC (held-out) |
|---|---|
| Raw detector confidence — the baseline to beat | 0.804 |
| Hand-set a-priori weights | 0.851 |
| **Fitted non-negative weights (deployed)** | **0.984** |
| Unconstrained fit | 0.971 — **rejected**, see below |

**Yes — +0.179 AUROC over raw confidence.**

### 2. Do the reliability buckets mean anything?

| Bucket | Frames | Share | Measured failure rate |
|---|---|---|---|
| TRUSTED (≥ 0.964) | 103 | 11.6% | **0.000** |
| CAUTION (0.940–0.964) | 171 | 19.3% | **0.000** |
| DEGRADED (< 0.940) | 614 | 69.1% | 0.389 |

Monotonic, and nothing inside TRUSTED actually failed on the held-out scene.

### 3. Can it warn *early*?

At the **lowest** severity of every degradation, against a clean baseline of 0.951 — all nine
score below the TRUSTED cut:

| Degradation | Score @ sev 0.25 | vs clean | Actual failure rate |
|---|---|---|---|
| frame_delay | 0.802 | −0.149 | 0.54 |
| patch_occlusion | 0.873 | −0.077 | 0.27 |
| distribution_shift | 0.934 | −0.017 | 0.10 |
| calibration_error | 0.940 | −0.011 | 0.10 |
| noise | 0.941 | −0.010 | 0.10 |
| motion_blur | 0.946 | −0.005 | 0.12 |
| glare / low_light / depth_degradation | 0.947 / 0.948 / 0.950 | −0.003 / −0.003 / −0.001 | 0.10 |

### 4. Does it identify the right cause?

Fraction of frames (severity ≥ 0.5) where the true cause is in the top-3 diagnosis:

| | | | |
|---|---|---|---|
| distribution_shift **1.00** | frame_delay **1.00** | patch_occlusion **0.99** | calibration_error **0.98** |
| motion_blur **0.98** | depth_degradation 0.85 | noise 0.55 | low_light 0.53 |
| glare **0.49** — coin flip, open issue | | | |

Glare and low light are entangled: both move exposure and clipping in overlapping ways, so the
monitor detects *that* something is wrong far more reliably than *which* of the two it is.

### What does it cost?

| | mean | p95 |
|---|---|---|
| End-to-end frame | **44.7 ms** (22.3 FPS) | 50.5 ms |
| Reliability monitor alone | **6.65 ms** (14.9% of budget) | — |

On 2 vCPU with **no GPU**. The monitor originally cost 24.67 ms (41.6% of the budget);
computing its image statistics at 320 px instead of full resolution cut it **~3.7×** with no
loss of held-out AUROC. That is safe because `fit_reference()` runs through the *same*
function — absolute Laplacian variance is scale-dependent, but the **ratio** the monitor
actually consumes is not.

---

## Honest findings, including the ones that look bad

This section exists because a portfolio project that reports only its wins is not engineering,
it's marketing. Each of these was measured, and each survived into the final repo rather than
being tuned away.

**1. The false-alarm rate on clean frames is 0.646.** Almost two thirds of clean frames score
below TRUSTED. The monitor errs toward distrust — the correct direction for an
emergency-braking consumer, but high. **Open issue**, not a solved problem.

**2. Two signals are anti-correlated with failure.** `sharpness` scored **AUROC 0.452** and
`noise` **0.472** — both worse than a coin flip. Cause: the reference detector matches **hue**,
and hue survives blur, so blur lowers the sharpness signal without actually breaking
perception. Both are **clamped to zero weight** and reported. *Specific to the colour-model
backend; must be re-measured with a CNN.*

**3. Pooling the scenes reverses the conclusion (Simpson's paradox).** Pooled across both
scenes, fused reliability (0.706) looks *worse* than raw confidence (0.739). Per-scene and held
out, it is 0.984 vs 0.804. The scenes have different base failure rates (0.431 vs 0.269), so
the pooled comparison is the wrong one. This was nearly reported as a negative result.

**4. The unconstrained fit scored well and was rejected anyway.** It reached 0.971 by giving
*negative* weights to chroma, depth_consistency, exposure, noise, sharpness and temporal —
learning "which scene am I in" from differing base rates rather than "is perception failing".
Non-negativity makes every weight mean *more of this signal ⇒ less trust*, the only form safe
to deploy.

**5. The best standalone signal has a deployed weight of zero.** `temporal` has the highest
individual AUROC (0.822) but is collinear with `motion` (0.799), which the fit preferred. High
individual informativeness does not imply marginal contribution.

**6. Building the playground exposed a reproducibility bug.** Cross-checking the site data
against `outputs/experiments.csv` revealed **187 disagreeing frames**. Root cause: degradation
seeding used Python's builtin `hash()` on a string, which is randomized per process — so
`noise`, `glare` and `patch_occlusion` produced *different* corruption on every run, and the
docstring's reproducibility claim was false across processes. Fixed with a stable BLAKE2b seed,
verified identical across three processes with randomized hash seeds, and the entire evaluation
chain was re-run. The playground now matches the CSV on **all 1,776 frames**.

### What this system cannot detect

- **Consistent-but-wrong perception.** If the detector confidently mislabels a static object
  every frame, temporal consistency, depth and geometry all agree. Nothing fires.
- **Correlated feature drift.** The OOD reference uses a *diagonal* covariance — with tens of
  clean frames a full 7×7 covariance is rank-deficient and its inverse is meaningless.
- **Sim-to-real distribution shift.** A hue rotation is not a new city, new weather, or a new sensor.
- **Its own calibration going stale.** Baselines are fitted once on clean frames; on a real
  robot they must be re-fitted per deployment.

---

## Architecture

```
RGB + depth ──► detect ──► track ──► lift to 3D ──► geometric check
                  │          │            │               │
                  └──────────┴────────────┴───────────────┴──► reliability monitor
                                                                 │
                                              13 signals ──► 4 channels ──► score + diagnosis
```

The monitor consumes **four independent channels** so that no single failure can silently take
out every signal at once:

| Channel | Signals | Sees |
|---|---|---|
| Appearance | sharpness, exposure, clipping, noise, chroma, ood | Image quality, distribution shift |
| Detection | confidence | Model self-report |
| Temporal | temporal, motion, timing | Frame-to-frame agreement, dropped frames |
| Geometric | geometry, depth_valid, depth_consistency | Physical consistency, **calibration** |

**Fusion is a weighted geometric mean**, not a sum. A geometric mean lets any single signal
veto the frame — with a sum, twelve healthy signals can outvote one that has collapsed to zero,
exactly the failure a safety monitor must not have.

Because `weighted geometric mean = exp(−Σ w·deficit)`, fitting a linear model on log-signals
yields the geometric-mean weights directly. Calibration changes the weights **without changing
the fusion form**.

### Two weight tables, and why they disagree

These are easy to confuse, so both ship. The standardized coefficients say which signal carries
most information *per standard deviation*; the deployed weights are those divided by each
signal's spread and normalized — what actually runs.

| Signal | Standardized coefficient | Deployed weight |
|---|---|---|
| geometry | 1.855 | 0.135 |
| clipping | 1.304 | **0.478** |
| confidence | 1.154 | 0.089 |
| motion | 0.800 | 0.059 |
| timing | 0.316 | 0.232 |
| depth_valid | 0.097 | 0.007 |
| chroma, depth_consistency, exposure, noise, ood, sharpness, temporal | 0.000 | 0.000 |

The ordering flips because `clipping` sits at 1.000 in almost every condition: a small spread
means a large per-unit weight, so when it finally moves, it moves the score hard. Publishing
only one of these tables would misrepresent the system.

---

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python scripts/smoke_m1.py            # geometry + detection + tracking gates (~30 s)
python -m unittest discover -s tests  # 53 tests
python scripts/run_experiments.py     # 1,776-frame sweep -> outputs/experiments.csv
python scripts/calibrate_monitor.py   # fit weights + thresholds -> configs/monitor_weights.json
python scripts/make_plots.py          # figures -> outputs/figures/
python scripts/build_site_data.py     # regenerate the playground from the real pipeline
```

Or reproduce the verification in one command, with no dataset to download:

```bash
docker build -t perceptionguard . && docker run --rm perceptionguard
```

### Verification gates

| Gate | Result |
|---|---|
| PnP rotation / translation / reprojection | **0.0000° / 0.0000 m / 0.0000 px** |
| Depth lies on object surfaces (interior) | max 0.0094 m = **0.370 px-equivalent** |
| Clean detection P / R / F1 | **0.961 / 0.961 / 0.961** |
| Track identity stability | **1 ID per object** (3/3) |
| Test suite | **53 passed** |
| Playground vs measured CSV | **0 mismatches / 1,776 frames** |

The depth gate is measured in **pixel-footprint units**, not metres. A 1 px error at 30 m is
60 mm; in metres the same sensor error looks 60× worse far away than near. Metres is the wrong
unit for a projective sensor.

---

## The playground

The GitHub Pages site ships an interactive playground: pick a scene, inject any of the 9
degradations at 4 severities, scrub through 24 frames, and watch the reliability score, the 13
signal contributions, the diagnosis and the ground-truth verdict update per frame.

**It is a viewer, not a simulator.** A static site cannot run OpenCV or the tracker, and there
were only three options:

1. Reimplement the pipeline in JavaScript — a *different code path*, so whatever it shows is not
   what the Python system does. **Rejected.**
2. Ship Pyodide and run the real code in-browser — `opencv-python` has no working wasm wheel.
   **Not feasible.**
3. **Precompute every frame with the real pipeline and ship the outputs.** ← chosen.

`scripts/build_site_data.py` runs the identical code path as the sweep — same renderer,
detector, tracker, monitor, degradations, bias correction and failure definition — then
verifies its fused score against an independent recomputation of the weighted geometric mean
(max drift `1.1e-16`) and aborts the build on mismatch. The output was checked frame-by-frame
against `outputs/experiments.csv`: **0 detection mismatches, 0 failure-label mismatches**.

To publish: **Settings → Pages → Source: `main` / `/docs`**. To run it locally:

```bash
cd docs && python -m http.server   # file:// blocks fetch(), so serve over HTTP
```

---

## Why synthetic data (and where that costs me)

The development environment had **no GPU, no network, 2 vCPU and 4 GB RAM** — KITTI, NYUv2 and
every pretrained checkpoint were unreachable. Rather than fake it, the repo renders its own
scenes analytically, which buys three things a downloaded dataset could not:

1. **Exact ground truth for the geometry** — PnP recovers pose to `0.0000° / 0.0000 m / 0.0000 px`.
   That is a correctness proof of the geometric core, not a benchmark score.
2. **Ground truth for the *degradation itself*** — so the monitor can be scored on whether it
   identified the *correct cause*, which no public dataset labels.
3. **Perfectly reproducible failure injection** — every degradation is seeded by
   `(name, severity, frame_index)` through a stable BLAKE2b hash, verified identical across
   processes.

**The cost, stated plainly:** the reference detector is a deterministic colour model, not a
neural network. So "fused signals beat raw confidence" is proven *for this backend* and must be
re-run with a CNN before it is generalized. `TorchDetector`
(`src/perceptionguard/inference/torch_detector.py`) exists as that swap seam and implements the
identical `Detector` protocol — nothing downstream changes.

---

## Repository layout

```
src/perceptionguard/
  geometry/camera.py       intrinsics, distortion, SE3, projection, PnP, reprojection error
  data/synthetic.py        analytic renderer with exact depth + instance ground truth
  data/scenes.py           four scenes: approach, crossing, occlusion, multi
  data/degradations.py     9 reproducible, severity-parameterized degradations
  perception/detector.py   colour-model reference detector (hue matching)
  perception/pipeline.py   detect -> track -> lift -> monitor, per-stage timing
  tracking/tracker.py      constant-velocity Kalman + 3-pass association
  reliability/monitor.py   13 signals, 4 channels, geometric-mean fusion, diagnosis
  evaluation/metrics.py    AUROC, PRF, matching, correlations (no sklearn/scipy)
  inference/               ONNX export seam + PyTorch backend (see Status)
scripts/                   smoke gates, sweep, calibration, figures, site data, diagnostics
docs/                      GitHub Pages site + playground (data generated by the pipeline)
tests/                     53 stdlib-unittest tests
```

### Engineering decisions worth defending in an interview

- **Association is 3-pass**: IoU → centre-distance recovery → *union-of-fragments*. When an
  occluder splits one object into two blobs, a single frame genuinely cannot distinguish that
  from two adjacent objects; only the **track prior** can. So the fix lives in the tracker, not
  in a detector heuristic. An earlier detector-side merge was implemented, proven unable to
  work, and fully reverted. (Regression test: `tests/test_tracking.py`.)
- **Group split by scene, not random split** — consecutive frames are near-duplicates.
- **AUROC returns NaN, not 0.5, when a class is absent** — a fake 0.5 silently pollutes averages.
- **Metrics are hand-implemented** (scipy/sklearn unavailable) and unit-tested against
  hand-computed values.
- **Thresholds are anchored to a *feasible* target.** Demanding ≤5% failures inside TRUSTED is
  unachievable when clean data itself fails at 20.8%; the first attempt pushed the cut to 0.996
  and left TRUSTED empty. The target is now clean-failure-rate + 5%.

---

## Status

| Milestone | State |
|---|---|
| 1–3 · Perception, 3D geometry, tracking | Complete, gated |
| 4 · Reliability monitor | Complete |
| 5 · Degradation suite (9) | Complete, reproducible across processes |
| 6 · Evaluation + calibration | Complete, held-out validated |
| 7 · Inference optimization | **Code only — not measured.** No GPU/PyTorch/TensorRT here. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). |
| 8 · Tests, Docker, CI, figures, docs, Pages + playground | Complete. Streamlit app written but **never executed** (not installed). |

**No FP16/INT8/TensorRT speedup is quoted anywhere in this repository**, because none was
measured. All latency numbers are real CPU measurements.

See [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) for the decision log, including every bug found
and how it was diagnosed.

## License

MIT
