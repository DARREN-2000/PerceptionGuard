"""Diagnostic: trace the detector's mask -> component -> detection funnel."""

from __future__ import annotations

import sys
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
from perceptionguard.perception.detector import ColorModelDetector  # noqa: E402
from perceptionguard.tracking.tracker import iou_xyxy  # noqa: E402

N = 24


def main() -> None:
    intr = default_intrinsics()
    det = ColorModelDetector(appearances())
    print(
        f"detector params: hue_thresh={det.hue_threshold_deg} min_chroma={det.min_chroma} "
        f"min_area={det.min_area} min_L={det.min_lightness}"
    )
    print(f"ref hues: { {k: round(v, 1) for k, v in det._ref_hue.items()} }")

    for t in (0, 12):
        frame = render_frame(
            build_scene("multi", t, N), intr, camera_pose(t, N), index=t
        )
        lab = cv2.cvtColor(frame.image, cv2.COLOR_BGR2LAB).astype(np.float64)
        ab = lab[:, :, 1:] - 128.0
        chroma = np.linalg.norm(ab, axis=2)
        hue = np.degrees(np.arctan2(ab[:, :, 1], ab[:, :, 0]))

        print(f"\n=== FRAME {t} ===")
        for app in det.appearances:
            rh = det._ref_hue[app.label]
            d_hue = np.abs((hue - rh + 180.0) % 360.0 - 180.0)
            m_hue = d_hue < det.hue_threshold_deg
            m_chr = chroma >= det.min_chroma
            m_lum = lab[:, :, 0] >= det.min_lightness
            mask = (m_hue & m_chr & m_lum).astype(np.uint8)
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, det._kernel)
            n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
            areas = sorted((int(stats[k][4]) for k in range(1, n_lbl)), reverse=True)[
                :5
            ]
            print(
                f"  {app.label:<11} hue_px={m_hue.sum():7d} chr_px={m_chr.sum():7d} "
                f"lum_px={m_lum.sum():7d} AND={mask.sum():7d} opened={opened.sum():7d} "
                f"comps={n_lbl - 1} top_areas={areas}"
            )

        dets = det.detect(frame.image)
        print(f"  -> detect() returned {len(dets)} detections")
        for d in dets:
            print(
                f"     {d.label:<11} score={d.score:.3f} bbox={[round(v) for v in d.bbox]} area={d.mask_area}"
            )
        print("  GT:")
        for inst in frame.instances:
            if inst.label == "occluder":
                continue
            best = max(
                (iou_xyxy(d.bbox, inst.bbox) for d in dets if d.label == inst.label),
                default=0.0,
            )
            print(
                f"     {inst.label:<11} bbox={[round(v) for v in inst.bbox]} "
                f"n_px={inst.num_pixels} bestIoU={best:.3f}"
            )


if __name__ == "__main__":
    main()
