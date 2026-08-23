"""Trace tracker association frame by frame to locate ID fragmentation."""

from __future__ import annotations

import sys
from pathlib import Path

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
from perceptionguard.tracking.tracker import Tracker, iou_xyxy  # noqa: E402

N, SCENE = 24, "multi"


def main() -> int:
    intr = default_intrinsics()
    det = ColorModelDetector(appearances())
    trk = Tracker()
    for t in range(N):
        frame = render_frame(
            build_scene(SCENE, t, N),
            intr,
            camera_pose(t, N),
            index=t,
            timestamp=t * 0.05,
        )
        gt1 = [g for g in frame.instances if g.obj_id == 1]
        before = {tr.track_id: tr.bbox for tr in trk.tracks}
        dets = det.detect(frame.image)
        active = trk.update(dets)

        g = gt1[0] if gt1 else None
        best = None
        if g is not None:
            for d in dets:
                i = iou_xyxy(g.bbox, d.bbox)
                if best is None or i > best[0]:
                    best = (i, d)
        ids = [tr.track_id for tr in active if g and iou_xyxy(g.bbox, tr.bbox) > 0.3]
        print(f"--- frame {t:2d} ---")
        if g is None:
            print("  gt_obj1: not visible")
        else:
            print(
                f"  gt_obj1 bbox=({g.bbox[0]:.0f},{g.bbox[1]:.0f},{g.bbox[2]:.0f},{g.bbox[3]:.0f})"
                f" vis={g.visible_ratio:.2f} z={g.depth:.1f}"
            )
            if best:
                d = best[1]
                print(
                    f"  best det: label={d.label} iou={best[0]:.2f} score={d.score:.2f}"
                    f" bbox=({d.bbox[0]:.0f},{d.bbox[1]:.0f},{d.bbox[2]:.0f},{d.bbox[3]:.0f})"
                )
            else:
                print("  best det: NONE")
        print(f"  tracks before: {sorted(before)}")
        for tr in trk.tracks:
            mark = "*" if g and iou_xyxy(g.bbox, tr.bbox) > 0.3 else " "
            print(
                f"   {mark}id={tr.track_id} label={tr.label} hits={tr.hits}"
                f" tsu={tr.time_since_update} conf={tr.confirmed}"
                f" bbox=({tr.bbox[0]:.0f},{tr.bbox[1]:.0f},{tr.bbox[2]:.0f},{tr.bbox[3]:.0f})"
            )
        print(f"  active ids overlapping gt1: {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
