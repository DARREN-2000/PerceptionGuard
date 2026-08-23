"""Controlled degradation sweep + reliability evaluation (Milestones 5-6).

Protocol
--------
1. Render clean sequences and FIT the monitor's reference statistics on them.
   Nothing about the reference is hardcoded.
2. Run the clean condition through the pipeline to measure (a) baseline accuracy
   and (b) the systematic depth bias of surface-centre lifting, which is then
   subtracted before labelling failures. Not correcting it would label almost
   every frame a failure and destroy the experiment.
3. Sweep every degradation over 5 severities, recording per-frame accuracy,
   3D error, every reliability signal, the fused score, and the diagnosis.
4. Answer the five required questions with measured numbers.

Failure label (stated explicitly because every downstream number depends on it):
a frame is a PERCEPTION FAILURE if any visible ground-truth object is missed,
or a false positive is emitted, or the bias-corrected depth error of any matched
object exceeds 1.0 m. This is a statement about the *output being wrong*, and is
computed entirely from ground truth -- the monitor never sees it.

Usage:  python scripts/run_experiments.py [--scenes multi,approach] [--frames 24]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
from perceptionguard.evaluation.metrics import (  # noqa: E402
    auroc,
    match_boxes,
    prf,
    spearman,
)
from perceptionguard.perception.detector import ColorModelDetector  # noqa: E402
from perceptionguard.perception.pipeline import PerceptionPipeline  # noqa: E402
from perceptionguard.reliability.monitor import ReliabilityMonitor  # noqa: E402
from perceptionguard.tracking.tracker import Tracker  # noqa: E402

BASE_DT = 0.05
MIN_VISIBLE = 0.30  # GT objects less visible than this are not required detections
DEPTH_FAIL_M = 1.0

# Height priors (m) matching the rendered box extents. Passed in explicitly so
# the monitor never imports the scene definitions.
CLASS_HEIGHTS: dict[str, float] = {"vehicle": 1.6, "pedestrian": 1.75, "cyclist": 1.7}

# Which diagnosis string SHOULD appear for each degradation. Sets, because some
# corruptions legitimately manifest through more than one signal.
EXPECTED_CAUSE: dict[str, set[str]] = {
    "motion_blur": {"Motion blur or defocus"},
    "low_light": {"Poor illumination (under-exposed)", "Colour information lost"},
    "noise": {"Sensor noise"},
    "glare": {"Glare / sensor saturation", "Poor illumination (under-exposed)"},
    "patch_occlusion": {
        "Missing depth returns",
        "Unstable tracking / temporal disagreement",
        "Low detector confidence",
    },
    "depth_degradation": {"Missing depth returns", "Inconsistent depth within objects"},
    "calibration_error": {"Geometric inconsistency (calibration or depth scale error)"},
    "frame_delay": {
        "Frame delay / dropped frames",
        "Unstable tracking / temporal disagreement",
    },
    "distribution_shift": {
        "Distribution shift (input unlike calibration set)",
        "Colour information lost",
        "Low detector confidence",
    },
}


def render_sequence(scene: str, n_frames: int):
    intr = default_intrinsics()
    frames = []
    for t in range(n_frames):
        boxes = build_scene(scene, t, n_frames)
        pose = camera_pose(t, n_frames)
        frames.append(render_frame(boxes, intr, pose, index=t, timestamp=t * BASE_DT))
    return intr, frames


def run_condition(scene, frames, intr, deg, sev, detector, monitor) -> list[dict]:
    pipe = PerceptionPipeline(detector, Tracker(), monitor)
    prev = None
    rows: list[dict] = []
    for t, frame in enumerate(frames):
        di = apply_degradation(deg, sev, frame.image, frame.depth, intr, frame_index=t)
        # Dropped frame: the pipeline re-processes stale sensor data while the
        # world (ground truth) has moved on.
        use = prev if (di.stale and prev is not None) else di
        prev = di
        out = pipe.process(
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
        est = out.estimates
        m = match_boxes(
            [g.bbox for g in gt],
            [e.bbox for e in est],
            gt_labels=[g.label for g in gt],
            pred_labels=[e.label for e in est],
            iou_threshold=0.5,
        )
        p, r, f1 = prf(m.tp, m.fp, m.fn)

        depth_errs, errs3d, ious = [], [], []
        for gi, pj, iou in m.matches:
            ious.append(iou)
            e = est[pj]
            if np.isfinite(e.depth):
                depth_errs.append(e.depth - gt[gi].depth)
            if e.center_3d is not None:
                errs3d.append(float(np.linalg.norm(e.center_3d - gt[gi].center_cam)))

        rep = out.report
        row = {
            "scene": scene,
            "degradation": deg,
            "channel": DEGRADATION_CHANNEL[deg],
            "severity": sev,
            "frame": t,
            "n_gt": len(gt),
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
            "precision": p,
            "recall": r,
            "f1": f1,
            "mean_iou": float(np.mean(ious)) if ious else np.nan,
            "depth_err_signed": float(np.mean(depth_errs)) if depth_errs else np.nan,
            "depth_err_max": float(np.max(np.abs(depth_errs)))
            if depth_errs
            else np.nan,
            "err3d": float(np.mean(errs3d)) if errs3d else np.nan,
            "reliability": rep.score,
            "status": rep.status,
            "diagnosis1": rep.diagnosis[0] if rep.diagnosis else "",
            "diagnosis_all": " | ".join(rep.diagnosis),
            "latency_ms": out.timings_ms["total"],
            "monitor_ms": out.timings_ms["monitor"],
            "n_tracks": rep.n_tracks,
        }
        for k, v in rep.signals.items():
            row[f"s_{k}"] = v
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="multi,approach")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    detector = ColorModelDetector(appearances())

    # --- 1. render + fit reference on clean data --------------------------
    seqs = {}
    clean_images = []
    for s in scenes:
        intr, frames = render_sequence(s, args.frames)
        seqs[s] = (intr, frames)
        clean_images.extend(f.image for f in frames)
    monitor = ReliabilityMonitor(CLASS_HEIGHTS)
    ref = monitor.fit_reference(clean_images, dt=BASE_DT)
    print(
        f"reference fitted on {len(clean_images)} clean frames: "
        f"lap_var={ref.lap_var:.1f} lum={ref.mean_lum:.1f} "
        f"noise={ref.noise:.3f} chroma={ref.chroma:.2f}"
    )

    # --- 2. sweep --------------------------------------------------------
    rows: list[dict] = []
    for s in scenes:
        intr, frames = seqs[s]
        for deg in degradation_names():
            sevs = (0.0,) if deg == "none" else SEVERITIES[1:]
            for sev in sevs:
                rows.extend(run_condition(s, frames, intr, deg, sev, detector, monitor))
        print(f"  scene {s}: {len(rows)} rows so far")
    df = pd.DataFrame(rows)

    # --- 3. bias correction + failure labels -----------------------------
    clean = df[df.degradation == "none"]
    bias = float(np.nanmedian(clean["depth_err_signed"]))
    df["depth_err_corr"] = df["depth_err_signed"] - bias
    df["failure"] = (
        (df["recall"] < 1.0)
        | (df["fp"] > 0)
        | (df["depth_err_corr"].abs() > DEPTH_FAIL_M)
    ).astype(int)
    df.to_csv(outdir / "experiments.csv", index=False)

    signal_cols = [c for c in df.columns if c.startswith("s_")]

    def hdr(t):
        print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    print(
        f"\nsurface-centre depth bias (clean median) = {bias:+.3f} m "
        f"-- subtracted before failure labelling"
    )
    print(f"total frames={len(df)}  failure rate={df.failure.mean():.3f}")

    hdr("Q1. Does reliability fall as degradation severity rises?")
    t1 = (
        df[df.degradation != "none"]
        .groupby(["degradation", "severity"])
        .agg(
            reliability=("reliability", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            depth_err=("depth_err_corr", lambda x: np.nanmean(np.abs(x))),
            err3d=("err3d", "mean"),
            failure=("failure", "mean"),
        )
        .round(3)
    )
    print(t1.to_string())
    clean_labeled = df[df.degradation == "none"]
    print(
        "\nclean baseline: "
        + repr(
            clean_labeled[["reliability", "recall", "f1", "failure"]]
            .mean()
            .round(3)
            .to_dict()
        )
    )

    hdr(
        "Q1b. Severity->reliability rank correlation (Spearman, want strongly negative)"
    )
    for deg in sorted(df.degradation.unique()):
        if deg == "none":
            continue
        sub = df[df.degradation == deg]
        rho = spearman(sub["severity"], sub["reliability"])
        rho_err = spearman(sub["severity"], sub["failure"])
        print(
            f"  {deg:<20s} rho(sev,reliability)={rho:+.3f}   rho(sev,failure)={rho_err:+.3f}"
        )

    hdr("Q4. Does fusing signals beat raw detector confidence? (AUROC for failure)")
    y = df["failure"].to_numpy()
    a_fused = auroc(1.0 - df["reliability"], y)
    a_conf = auroc(1.0 - df["s_confidence"], y)
    print(f"  fused reliability      AUROC = {a_fused:.3f}")
    print(f"  raw detector confidence AUROC = {a_conf:.3f}")
    print(f"  delta = {a_fused - a_conf:+.3f}")

    hdr("Q3. Which individual signals are most informative? (AUROC, all conditions)")
    per_sig = sorted(
        ((c, auroc(1.0 - df[c], y)) for c in signal_cols),
        key=lambda kv: -kv[1] if np.isfinite(kv[1]) else 0,
    )
    for c, a in per_sig:
        print(f"  {c:<22s} AUROC = {a:.3f}")

    hdr("Q3b. Per-degradation: which signal actually reacts? (mean normalized value)")
    t3 = df.groupby("degradation")[signal_cols].mean().round(3)
    print(t3.to_string())

    hdr("Q5. Diagnosis accuracy: is the true cause in the top-3 diagnosis?")
    for deg in sorted(df.degradation.unique()):
        if deg == "none":
            continue
        sub = df[(df.degradation == deg) & (df.severity >= 0.5)]
        exp = EXPECTED_CAUSE.get(deg, set())
        hit = sub["diagnosis_all"].apply(
            lambda s: any(e in s for e in exp) if isinstance(s, str) else False
        )
        print(
            f"  {deg:<20s} ({DEGRADATION_CHANNEL[deg]:<12s}) top3 contains cause: {hit.mean():.2f}"
        )

    hdr("Runtime cost (2 vCPU, no GPU)")
    print(
        f"  total pipeline  mean={df.latency_ms.mean():.1f} ms  "
        f"p95={np.percentile(df.latency_ms, 95):.1f} ms  -> {1000 / df.latency_ms.mean():.1f} FPS"
    )
    print(
        f"  monitor only    mean={df.monitor_ms.mean():.2f} ms  "
        f"({100 * df.monitor_ms.mean() / df.latency_ms.mean():.1f}% of frame budget)"
    )

    print(f"\nwrote {outdir / 'experiments.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
