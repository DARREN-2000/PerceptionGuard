"""Generate the GitHub Pages playground data from the REAL pipeline.

Why this script exists
----------------------
A static site cannot run PyTorch, OpenCV or the tracker. There are three ways to
put a "playground" on GitHub Pages and only one of them is honest:

  1. Reimplement the pipeline in JavaScript  -> a different code path. Whatever
     it shows is NOT what the Python system does. Rejected.
  2. Ship Pyodide and run the real code in the browser -> opencv-python has no
     working wasm wheel, and the tracker/monitor need it. Not feasible here.
  3. PRECOMPUTE every frame with the real pipeline and ship the outputs.
     The browser is then a viewer for measured results, not a simulator.

This script is option 3. Every number and every pixel the playground displays
was produced by the same code path as ``scripts/run_experiments.py``:
same renderer, same detector, same tracker, same monitor, same degradations,
same bias correction, and the SAME failure definition.

Deliberate consistency requirements (any drift here would make the site lie):
  * The monitor is constructed with the DEPLOYED fitted weights from
    ``configs/monitor_weights.json``, not the a-priori defaults that
    ``run_experiments.py`` uses to produce the raw CSV.
  * Status is assigned with the CALIBRATED thresholds from that same file,
    because the library's built-in status cuts are pre-calibration defaults.
  * The fused score is cross-checked against an independent recomputation of
    the weighted geometric mean; a mismatch aborts the build.
  * ``failure`` is recomputed exactly as in the sweep: missed visible GT, or any
    false positive, or bias-corrected depth error > 1 m.

Usage:
    python scripts/build_site_data.py [--scenes multi,approach] [--frames 24]
                                      [--width 288] [--quality 66]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perceptionguard.data.degradations import (  # noqa: E402
    DEGRADATION_CHANNEL,
    SEVERITIES,
    apply_degradation,
    degradation_names,
)
from perceptionguard.data.scenes import (  # noqa: E402
    appearances,
    build_scene,
    camera_pose,
    default_intrinsics,
)
from perceptionguard.data.synthetic import render_frame  # noqa: E402
from perceptionguard.evaluation.metrics import match_boxes, prf  # noqa: E402
from perceptionguard.perception.detector import ColorModelDetector  # noqa: E402
from perceptionguard.perception.pipeline import PerceptionPipeline  # noqa: E402
from perceptionguard.reliability.monitor import ReliabilityMonitor  # noqa: E402
from perceptionguard.tracking.tracker import Tracker  # noqa: E402

# Mirrored from scripts/run_experiments.py -- must stay identical.
BASE_DT = 0.05
MIN_VISIBLE = 0.30
DEPTH_FAIL_M = 1.0
CLASS_HEIGHTS: dict[str, float] = {"vehicle": 1.6, "pedestrian": 1.75, "cyclist": 1.7}
EPS = 1e-6

# BGR overlay colours.
C_MATCH = (150, 255, 90)  # correctly detected
C_MISS = (60, 60, 255)  # missed ground truth (false negative)
C_FP = (40, 170, 255)  # false positive
C_GT = (170, 170, 170)  # ground-truth reference


def sev_tag(sev: float) -> str:
    return f"{int(round(sev * 100)):03d}"


def depth_panel(depth: np.ndarray) -> np.ndarray:
    """Colourize depth. NaN (background / no return) renders black."""
    valid = np.isfinite(depth)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        d = depth[valid]
        lo, hi = float(d.min()), float(d.max())
        norm = (depth - lo) / max(hi - lo, 1e-6)
        gray[valid] = np.clip(255.0 * (1.0 - norm[valid]), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def draw_overlay(image, gt, est, matched_gt, matched_pred) -> np.ndarray:
    """Draw ground truth and predictions so failures are visible, not asserted."""
    canvas = image.copy()
    for gi, g in enumerate(gt):
        x1, y1, x2, y2 = (int(v) for v in g.bbox)
        hit = gi in matched_gt
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C_GT if hit else C_MISS, 1)
        if not hit:
            cv2.putText(
                canvas,
                "MISSED",
                (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                C_MISS,
                1,
                cv2.LINE_AA,
            )
    for pj, e in enumerate(est):
        x1, y1, x2, y2 = (int(v) for v in e.bbox)
        ok = pj in matched_pred
        col = C_MATCH if ok else C_FP
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        depth = f" {e.depth:.1f}m" if np.isfinite(e.depth) else " no-depth"
        tag = f"#{e.track_id}{depth}" if ok else f"FP #{e.track_id}"
        cv2.putText(
            canvas,
            tag,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            col,
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="multi,approach")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--width", type=int, default=288, help="width of each panel")
    ap.add_argument("--quality", type=int, default=66)
    ap.add_argument("--out", default=str(ROOT / "docs" / "data"))
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    out = Path(args.out)
    frames_dir = out / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = ROOT / "configs" / "monitor_weights.json"
    cfg = json.loads(cfg_path.read_text())
    weights: dict[str, float] = cfg["weights"]
    t_trusted = float(cfg["threshold_trusted"])
    t_caution = float(cfg["threshold_caution"])
    sig_names = sorted(weights)
    w_vec = np.array([weights[n] for n in sig_names], dtype=float)
    print(
        f"deployed weights loaded from {cfg_path.name}; "
        f"TRUSTED>={t_trusted:.3f} CAUTION>={t_caution:.3f}"
    )

    detector = ColorModelDetector(appearances())

    # --- 1. render clean sequences and fit the monitor reference ----------
    seqs = {}
    clean_images = []
    for s in scenes:
        intr = default_intrinsics()
        seq = []
        for t in range(args.frames):
            boxes = build_scene(s, t, args.frames)
            pose = camera_pose(t, args.frames)
            seq.append(render_frame(boxes, intr, pose, index=t, timestamp=t * BASE_DT))
        seqs[s] = (intr, seq)
        clean_images.extend(f.image for f in seq)

    # Deployed weights, not the a-priori defaults: the site must show what ships.
    monitor = ReliabilityMonitor(CLASS_HEIGHTS, weights=weights)
    ref = monitor.fit_reference(clean_images, dt=BASE_DT)
    print(
        f"reference fitted on {len(clean_images)} clean frames: "
        f"lap_var={ref.lap_var:.1f} lum={ref.mean_lum:.1f} "
        f"noise={ref.noise:.3f} chroma={ref.chroma:.2f}"
    )

    # --- 2. sweep every condition, saving frames + records ----------------
    records: list[dict] = []
    max_score_drift = 0.0
    t0 = time.time()

    for scene in scenes:
        intr, seq = seqs[scene]
        for deg in degradation_names():
            sevs = (0.0,) if deg == "none" else SEVERITIES[1:]
            for sev in sevs:
                cond_dir = frames_dir / scene / f"{deg}_{sev_tag(sev)}"
                cond_dir.mkdir(parents=True, exist_ok=True)
                pipe = PerceptionPipeline(detector, Tracker(), monitor)
                prev = None

                for t, frame in enumerate(seq):
                    di = apply_degradation(
                        deg, sev, frame.image, frame.depth, intr, frame_index=t
                    )
                    # Dropped frame: the pipeline re-processes stale sensor data
                    # while ground truth has moved on. Show what it actually saw.
                    use = prev if (di.stale and prev is not None) else di
                    prev = di
                    o = pipe.process(
                        image=use.image,
                        depth=use.depth,
                        intrinsics=use.intrinsics,
                        frame_index=t,
                        dt=BASE_DT * di.dt_scale,
                    )

                    gt = [
                        g
                        for g in frame.instances
                        if g.label in CLASS_HEIGHTS and g.visible_ratio >= MIN_VISIBLE
                    ]
                    est = o.estimates
                    m = match_boxes(
                        [g.bbox for g in gt],
                        [e.bbox for e in est],
                        gt_labels=[g.label for g in gt],
                        pred_labels=[e.label for e in est],
                        iou_threshold=0.5,
                    )
                    _, recall, _ = prf(m.tp, m.fp, m.fn)

                    matched_gt = {gi for gi, _, _ in m.matches}
                    matched_pred = {pj for _, pj, _ in m.matches}
                    derrs = [
                        est[pj].depth - gt[gi].depth
                        for gi, pj, _ in m.matches
                        if np.isfinite(est[pj].depth)
                    ]

                    rep = o.report
                    sig = np.array([float(rep.signals.get(n, 1.0)) for n in sig_names])
                    # Independent recomputation of the weighted geometric mean.
                    manual = float(np.exp(-(-np.log(np.clip(sig, EPS, 1.0)) @ w_vec)))
                    max_score_drift = max(max_score_drift, abs(manual - rep.score))

                    rel = manual
                    status = (
                        "TRUSTED"
                        if rel >= t_trusted
                        else "CAUTION"
                        if rel >= t_caution
                        else "DEGRADED"
                    )

                    rgb = draw_overlay(use.image, gt, est, matched_gt, matched_pred)
                    panel = (
                        args.width,
                        int(round(args.width * rgb.shape[0] / rgb.shape[1])),
                    )
                    comp = np.hstack(
                        [
                            cv2.resize(rgb, panel, interpolation=cv2.INTER_AREA),
                            cv2.resize(
                                depth_panel(use.depth),
                                panel,
                                interpolation=cv2.INTER_AREA,
                            ),
                        ]
                    )
                    rel_path = f"{scene}/{deg}_{sev_tag(sev)}/f{t:02d}.jpg"
                    cv2.imwrite(
                        str(frames_dir / rel_path),
                        comp,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.quality],
                    )

                    records.append(
                        {
                            "scene": scene,
                            "deg": deg,
                            "sev": float(sev),
                            "i": t,
                            "s": round(rel, 4),
                            "st": status,
                            "sg": [round(float(v), 4) for v in sig],
                            "d": list(rep.diagnosis),
                            "tp": int(m.tp),
                            "fp": int(m.fp),
                            "fn": int(m.fn),
                            "ng": len(gt),
                            "nt": int(rep.n_tracks),
                            "rec": round(float(recall), 4),
                            "dsig": (
                                round(float(np.mean(derrs)), 4) if derrs else None
                            ),
                            "lat": round(float(o.timings_ms["total"]), 2),
                            "mon": round(float(o.timings_ms["monitor"]), 2),
                            "img": rel_path,
                        }
                    )
        print(
            f"  scene {scene}: {len(records)} frames rendered "
            f"({time.time() - t0:.0f}s elapsed)"
        )

    if max_score_drift > 1e-6:
        print(
            f"ERROR: fused score disagrees with the recomputed geometric mean "
            f"by {max_score_drift:.2e}. The site would misreport the monitor."
        )
        return 1
    print(f"score cross-check OK (max drift {max_score_drift:.2e})")

    # --- 3. bias correction + failure labels, exactly as in the sweep -----
    clean_d = [
        r["dsig"] for r in records if r["deg"] == "none" and r["dsig"] is not None
    ]
    bias = float(np.median(clean_d)) if clean_d else 0.0
    for r in records:
        dc = None if r["dsig"] is None else r["dsig"] - bias
        r["de"] = None if dc is None else round(dc, 4)
        r["f"] = int(
            (r["rec"] < 1.0)
            or (r["fp"] > 0)
            or (dc is not None and abs(dc) > DEPTH_FAIL_M)
        )
        del r["dsig"]
    fail_rate = float(np.mean([r["f"] for r in records]))
    print(
        f"depth bias (clean median) = {bias:+.3f} m -> failure rate {fail_rate:.3f} "
        f"over {len(records)} frames"
    )

    # --- 4. pack -----------------------------------------------------------
    conditions: dict[str, dict] = {}
    for r in records:
        key = f"{r['scene']}|{r['deg']}|{sev_tag(r['sev'])}"
        conditions.setdefault(key, {"frames": []})["frames"].append(
            {k: v for k, v in r.items() if k not in ("scene", "deg", "sev")}
        )

    lat = np.array([r["lat"] for r in records])
    mon = np.array([r["mon"] for r in records])
    degs = [
        {
            "name": d,
            "channel": DEGRADATION_CHANNEL[d],
            "severities": [0.0] if d == "none" else [float(s) for s in SEVERITIES[1:]],
        }
        for d in degradation_names()
    ]

    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "scenes": scenes,
            "degradations": degs,
            "n_frames": args.frames,
            "signals": sig_names,
            "weights": {n: round(float(weights[n]), 4) for n in sig_names},
            "threshold_trusted": t_trusted,
            "threshold_caution": t_caution,
            "train_scene": cfg["train_scene"],
            "test_scene": cfg["test_scene"],
            "test_auroc_fitted": cfg["test_auroc_fitted"],
            "test_auroc_confidence": cfg["test_auroc_confidence_baseline"],
            "total_frames": len(records),
            "failure_rate": round(fail_rate, 4),
            "depth_bias_m": round(bias, 4),
            "latency_mean_ms": round(float(lat.mean()), 2),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "monitor_mean_ms": round(float(mon.mean()), 2),
            "panel_width": args.width,
        },
        "conditions": conditions,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "playground.json").write_text(json.dumps(payload, separators=(",", ":")))

    # --- 5. copy the measured figures into the site ------------------------
    figsrc = ROOT / "outputs" / "figures"
    figdst = ROOT / "docs" / "assets" / "figures"
    figdst.mkdir(parents=True, exist_ok=True)
    n_fig = 0
    for png in sorted(figsrc.glob("*.png")):
        shutil.copy2(png, figdst / png.name)
        n_fig += 1

    jsz = (out / "playground.json").stat().st_size / 1e6
    isz = sum(p.stat().st_size for p in frames_dir.rglob("*.jpg")) / 1e6
    n_img = sum(1 for _ in frames_dir.rglob("*.jpg"))
    print(f"\nwrote {out / 'playground.json'}  ({jsz:.2f} MB)")
    print(
        f"wrote {n_img} frames ({isz:.1f} MB, {1000 * isz / max(n_img, 1):.1f} KB each)"
    )
    print(f"copied {n_fig} figures -> docs/assets/figures/")
    print(f"TOTAL SITE PAYLOAD: {jsz + isz:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
