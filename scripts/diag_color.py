"""Diagnostic: what does the renderer actually put on screen, in Lab terms?

Run before touching detector thresholds, so they are chosen from measured data
rather than guessed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perceptionguard.data.scenes import (  # noqa: E402
    OBJECT_PALETTE,
    build_scene,
    camera_pose,
    default_intrinsics,
)
from perceptionguard.data.synthetic import render_frame  # noqa: E402

N = 24


def ab_of_bgr(bgr) -> np.ndarray:
    return cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[
        0, 0
    ].astype(float)


def main() -> None:
    intr = default_intrinsics()
    print("REFERENCE palette (unshaded):")
    for lbl, bgr in OBJECT_PALETTE.items():
        L, a, b = ab_of_bgr(bgr)
        ab = np.array([a - 128.0, b - 128.0])
        print(
            f"  {lbl:<11} BGR={bgr}  L={L:6.1f} a={a - 128:+7.1f} b={b - 128:+7.1f} "
            f"chroma={np.linalg.norm(ab):6.1f} hue={np.degrees(np.arctan2(ab[1], ab[0])):+7.1f}deg"
        )

    for t in (0, N // 2, N - 1):
        frame = render_frame(
            build_scene("multi", t, N), intr, camera_pose(t, N), index=t
        )
        lab = cv2.cvtColor(frame.image, cv2.COLOR_BGR2LAB).astype(float)
        print(f"\nFRAME {t}: rendered instance pixels")
        for inst in frame.instances:
            m = frame.instance_map == inst.obj_id
            if not m.any():
                continue
            px = lab[m]
            L = px[:, 0]
            ab = px[:, 1:] - 128.0
            chroma = np.linalg.norm(ab, axis=1)
            hue = np.degrees(np.arctan2(ab[:, 1], ab[:, 0]))
            ref = None
            if inst.label in OBJECT_PALETTE:
                rL, ra, rb = ab_of_bgr(OBJECT_PALETTE[inst.label])
                rab = np.array([ra - 128.0, rb - 128.0])
                ref = rab
                d = np.linalg.norm(ab - rab, axis=1)
                rhue = np.degrees(np.arctan2(rab[1], rab[0]))
                dhue = np.abs((hue - rhue + 180) % 360 - 180)
            print(
                f"  {inst.label:<11} n={inst.num_pixels:6d} z={inst.depth:5.1f}m "
                f"vis={inst.visible_ratio:.2f}"
            )
            print(
                f"      L      : min={L.min():6.1f} mean={L.mean():6.1f} max={L.max():6.1f}"
            )
            print(
                f"      chroma : min={chroma.min():6.1f} mean={chroma.mean():6.1f} max={chroma.max():6.1f}"
            )
            if ref is not None:
                print(
                    f"      |ab-ref|: min={d.min():6.1f} mean={d.mean():6.1f} max={d.max():6.1f}"
                    f"   (current thresh=14)"
                )
                print(
                    f"      d_hue  : min={dhue.min():6.1f} mean={dhue.mean():6.1f} max={dhue.max():6.1f} deg"
                )
                print(f"      frac passing |ab-ref|<14 : {(d < 14).mean():.3f}")
                print(f"      frac passing d_hue<20    : {(dhue < 20).mean():.3f}")


if __name__ == "__main__":
    main()
