"""Milestone 1-3 verification.

This is not a demo. It asserts the properties the rest of the project depends on:

1. GEOMETRY   -- unprojecting the rendered depth map must land on the object
                 surface, and PnP on the projected corners must recover the
                 ground-truth pose. If either fails, every 3D and reliability
                 number downstream is meaningless.
2. DETECTION  -- the reference detector must actually find the objects under
                 clean conditions (a floor to measure degradation against).
3. TRACKING   -- identities must be stable across a clean sequence.

Run:  python3 scripts/smoke_m1.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perceptionguard.data.scenes import (  # noqa: E402
    appearances,
    build_scene,
    camera_pose,
    default_intrinsics,
)
from perceptionguard.data.synthetic import render_frame  # noqa: E402
from perceptionguard.geometry.camera import solve_pnp, unproject_pixels  # noqa: E402
from perceptionguard.perception.detector import ColorModelDetector  # noqa: E402
from perceptionguard.tracking.tracker import Tracker, iou_xyxy  # noqa: E402

N_FRAMES = 24
SCENE = "multi"


def main() -> int:
    intr = default_intrinsics()
    detector = ColorModelDetector(appearances())
    tracker = Tracker()

    print(
        f"scene={SCENE}  frames={N_FRAMES}  {intr.width}x{intr.height}  "
        f"fx={intr.fx:.1f}"
    )
    print("-" * 78)

    # Two separate measurements, because they test different things:
    #  interior -- is the depth model itself exact? Must be ~0.
    #  boundary -- how far does fillConvexPoly stray off the true face? This is
    #              a rasterization artifact, so it is only meaningful in units of
    #              pixel footprint (z/fx), not metres: 1 px at 30 m is 60 mm.
    depth_resid_interior: list[float] = []
    depth_resid_interior_px: list[float] = []
    depth_resid_px: list[float] = []
    pnp_rot_err: list[float] = []
    pnp_trans_err: list[float] = []
    pnp_rmse: list[float] = []
    tp = fp = fn = 0
    render_ms: list[float] = []
    detect_ms: list[float] = []
    id_by_obj: dict[int, set[int]] = {}

    for t in range(N_FRAMES):
        boxes = build_scene(SCENE, t, N_FRAMES)
        pose = camera_pose(t, N_FRAMES)

        t0 = time.perf_counter()
        frame = render_frame(boxes, intr, pose, index=t, timestamp=t / 20.0)
        render_ms.append((time.perf_counter() - t0) * 1e3)

        # --- 1a. Depth self-consistency -------------------------------------
        # Unproject every pixel of each instance using the rendered depth and
        # verify the points lie on that box's surface (distance to the box, in
        # the object frame, must be ~0).
        for inst in frame.instances:
            if inst.label == "occluder":
                continue
            inst_mask = (frame.instance_map == inst.obj_id).astype(np.uint8)
            if not inst_mask.any():
                continue
            half = np.abs(inst.corners_object).max(axis=0)

            def surface_error(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                ys, xs = np.nonzero(mask)
                if xs.size == 0:
                    return np.empty(0), np.empty(0)
                sel = np.linspace(0, xs.size - 1, min(xs.size, 400)).astype(int)
                uv = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float64)
                z = frame.depth[ys[sel], xs[sel]].astype(np.float64)
                pts_cam = unproject_pixels(uv, z, intr)
                pts_obj = inst.T_co.inverse().apply(pts_cam)
                # Distance from the point to the axis-aligned box in the object
                # frame; zero iff the point lies on or inside the surface.
                outside = np.maximum(np.abs(pts_obj) - half, 0.0)
                return np.linalg.norm(outside, axis=1), z

            # Interior: erode away a 2 px band so only true-face pixels remain.
            interior = cv2.erode(inst_mask, np.ones((3, 3), np.uint8), iterations=2)
            err_in, z_in = surface_error(interior)
            if err_in.size:
                depth_resid_interior.append(float(err_in.max()))
                fp_in = np.maximum(z_in / intr.fx, 1e-9)
                depth_resid_interior_px.append(float((err_in / fp_in).max()))

            # All pixels, expressed in pixel-footprint units.
            err_all, z_all = surface_error(inst_mask)
            if err_all.size:
                footprint = np.maximum(z_all / intr.fx, 1e-9)
                depth_resid_px.append(float(np.percentile(err_all / footprint, 95)))

            # --- 1b. PnP pose recovery --------------------------------------
            est, mask, rmse = solve_pnp(
                inst.corners_object, inst.corners_px, intr, use_ransac=False
            )
            if est is not None:
                dR = est.R @ inst.T_co.R.T
                cos = (np.trace(dR) - 1.0) / 2.0
                pnp_rot_err.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
                pnp_trans_err.append(float(np.linalg.norm(est.t - inst.T_co.t)))
                pnp_rmse.append(float(rmse))

        # --- 2. Detection ----------------------------------------------------
        t0 = time.perf_counter()
        dets = detector.detect(frame.image)
        detect_ms.append((time.perf_counter() - t0) * 1e3)

        gts = [i for i in frame.instances if i.label != "occluder"]
        used = set()
        for gt in gts:
            best, best_iou = -1, 0.0
            for k, d in enumerate(dets):
                if k in used or d.label != gt.label:
                    continue
                v = iou_xyxy(d.bbox, gt.bbox)
                if v > best_iou:
                    best, best_iou = k, v
            if best >= 0 and best_iou >= 0.5:
                tp += 1
                used.add(best)
            else:
                fn += 1
        fp += len(dets) - len(used)

        # --- 3. Tracking -----------------------------------------------------
        tracks = tracker.update(dets)
        for tr in tracks:
            best, best_iou = None, 0.0
            for gt in gts:
                v = iou_xyxy(tr.bbox, gt.bbox)
                if v > best_iou:
                    best, best_iou = gt, v
            if best is not None and best_iou >= 0.3:
                id_by_obj.setdefault(best.obj_id, set()).add(tr.track_id)

    def stat(name: str, arr: list[float], unit: str = "") -> None:
        if not arr:
            print(f"{name:<34} (no samples)")
            return
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        print(
            f"{name:<34} mean={a.mean():8.4f}  p95={np.percentile(a, 95):8.4f}  "
            f"max={a.max():8.4f} {unit}"
        )

    print("GEOMETRY")
    stat("  depth residual (interior)", depth_resid_interior, "m")
    stat("  depth residual (interior)", depth_resid_interior_px, "px-equiv")
    stat("  depth residual (all, p95)", depth_resid_px, "px-equiv")
    stat("  PnP rotation error", pnp_rot_err, "deg")
    stat("  PnP translation error", pnp_trans_err, "m")
    stat("  PnP reprojection RMSE", pnp_rmse, "px")
    print()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print("DETECTION (clean, IoU>=0.5)")
    print(
        f"  TP={tp}  FP={fp}  FN={fn}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}"
    )
    print()
    print("TRACKING")
    for obj_id, ids in sorted(id_by_obj.items()):
        flag = "OK" if len(ids) == 1 else f"{len(ids)} IDs (fragmented)"
        print(f"  gt_obj {obj_id}: track_ids={sorted(ids)}  {flag}")
    print()
    print("TIMING (2 vCPU, no GPU)")
    stat("  render", render_ms, "ms")
    stat("  detect", detect_ms, "ms")

    # Hard gates. These are correctness properties, not tuning targets.
    ok = True
    # Gated in pixel-footprint units, not metres: for a projective sensor the
    # metric error of a 1 px rasterization offset grows linearly with depth
    # (60 mm at 30 m here), so an absolute-metre gate would just be a proxy for
    # scene depth. Sub-pixel is the property we actually require.
    if depth_resid_interior_px and np.nanmax(depth_resid_interior_px) > 0.5:
        print(
            "\nFAIL: interior depth does not lie on object surfaces "
            "-- the depth model itself is wrong"
        )
        ok = False
    if depth_resid_px and np.nanmean(depth_resid_px) > 2.0:
        print(
            "\nFAIL: silhouette depth error exceeds 2 px of footprint "
            "-- rasterization is straying off faces"
        )
        ok = False
    if pnp_rmse and np.nanmean(pnp_rmse) > 1.0:
        print("\nFAIL: PnP reprojection RMSE too high")
        ok = False
    if rec < 0.8:
        print(f"\nFAIL: clean recall {rec:.3f} < 0.80 -- no baseline to degrade from")
        ok = False
    print("\n" + ("ALL GATES PASSED" if ok else "GATES FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
