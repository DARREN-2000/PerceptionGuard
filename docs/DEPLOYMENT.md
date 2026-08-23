# Deployment and edge inference

## Status: this document describes an intended path, not a measured result

The development environment had **no GPU, no PyTorch, no ONNX, no TensorRT and no network
access**. Therefore:

- **No FP32 / FP16 / INT8 comparison is quoted anywhere in this repository.**
- **No TensorRT speedup number appears in the README.**
- The code below is written and type-checked but has **never been executed**.

Quoting an unmeasured 3x TensorRT speedup would be the single fastest way to fail a
technical interview, so it is not quoted. What follows is the plan, the seams that make it
a small change, and the numbers that *were* measured.

## What was actually measured (CPU, 2 vCPU Xeon @ 2.90 GHz, no GPU)

| Stage | mean ms/frame | share |
|---|---|---|
| Detection | 22.4 | 53.8% |
| Reliability monitor | 6.34 | 15.2% |
| Tracking + 3D lift | 5.9 | 14.2% |
| Render / IO | 6.9 | 16.6% |
| **End-to-end** | **41.6** (24.0 FPS) | p95 47.0 ms |

One optimization *was* measured, because it could be: computing the monitor's image
statistics on a 320 px-wide downsample rather than at full resolution.

| | monitor ms | frame ms | FPS |
|---|---|---|---|
| Full resolution | 24.67 | 59.2 | 16.9 |
| **320 px statistics** | **6.34** | **41.6** | **24.0** |

**3.9x on the monitor, +42% end-to-end**, with no change in held-out AUROC. This is safe
because `fit_reference()` runs through the *same* function: the absolute Laplacian variance
changes with scale, but the **ratio to the reference** does not — and the monitor only ever
consumes the ratio.

## The swap seam

Everything downstream of detection depends only on this protocol:

```python
class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]: ...
```

`Detection` is `(bbox, label, score)`. The tracker, 3D lift, geometry checks and all 13
reliability signals are backend-agnostic. Swapping the colour-model reference detector for
a CNN is a one-line change at the construction site:

```python
# from:
detector = ColorModelDetector(palette)
# to:
detector = TorchDetector(
    "fasterrcnn_mobilenet_v3_large_fpn", device="cuda", score_threshold=0.5
)
```

`src/perceptionguard/inference/torch_detector.py` implements this and imports torch lazily,
so the package still imports cleanly on a machine without torch installed — which is how
the 53 tests run here.

## Intended path: PyTorch -> ONNX -> TensorRT

```bash
pip install -r requirements-gpu.txt

# 1. export
python -m perceptionguard.inference.onnx_export \
    --model fasterrcnn_mobilenet_v3_large_fpn \
    --output outputs/detector.onnx --opset 17 --half

# 2. build a TensorRT engine
trtexec --onnx=outputs/detector.onnx --saveEngine=outputs/detector_fp16.plan \
        --fp16 --workspace=4096

# 3. re-run the identical sweep against the optimized backend
python scripts/run_experiments.py --backend tensorrt --engine outputs/detector_fp16.plan
```

Step 3 matters most: the benchmark harness is the *same* `run_experiments.py`, so an
optimized backend is scored on the same 1,776 frames with the same failure definition.

## What must be re-validated after quantizing — not just latency

This is the part usually skipped. FP16/INT8 changes the *scores*, and the monitor consumes
scores. So the following must be re-measured, not assumed:

1. **Detection quality**: precision/recall/F1 vs the FP32 baseline.
2. **Reliability calibration**: held-out AUROC. If quantization compresses the confidence
   distribution, `s_confidence` shifts and the fitted weights are stale.
3. **Thresholds**: `configs/monitor_weights.json` is fitted to a specific backend.
   **Re-run `scripts/calibrate_monitor.py` after any backend change.** Reusing FP32
   thresholds under INT8 is a silent, dangerous failure.
4. **The reference statistics**: `fit_reference()` must be re-fitted on clean frames from
   the *target sensor*, not the development sensor.

A useful property of this design: the monitor's own cost (6.34 ms) is independent of the
detector backend, so as detection is accelerated the monitor becomes a *larger* fraction of
the budget. At a 5 ms TensorRT detector, the monitor would dominate — the next optimization
target would be the monitor, not the network.

## Running on an edge robot

| Concern | Approach in this repo |
|---|---|
| Fixed frame budget | `PipelineOutput.timings_ms` reports per-stage cost every frame |
| Dropped frames | `dt` is passed explicitly per frame; the `timing` signal degrades when `dt` exceeds nominal |
| Sensor recalibration | Intrinsics are an explicit argument, never a global |
| Monitor cost scaling | `enabled_signals=` disables channels to fit a tighter budget |
| Thermal/clock variance | Report p95 (47.0 ms), not just the mean |

The pipeline is single-threaded and allocation-light by design; the natural deployment shape
is one process per camera with the monitor inline, because a reliability score that arrives
a frame late describes a frame the robot has already acted on.
