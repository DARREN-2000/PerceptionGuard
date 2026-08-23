"""End-to-end integration tests: render -> detect -> track -> lift -> monitor.

These are the tests that would catch a wiring regression that all the unit
tests would happily pass.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perceptionguard.data.degradations import apply_degradation  # noqa: E402
from perceptionguard.data.scenes import (  # noqa: E402
    appearances,
    build_scene,
    camera_pose,
    default_intrinsics,
)
from perceptionguard.data.synthetic import render_frame  # noqa: E402
from perceptionguard.perception.detector import ColorModelDetector  # noqa: E402
from perceptionguard.perception.pipeline import PerceptionPipeline  # noqa: E402
from perceptionguard.reliability.monitor import ReliabilityMonitor  # noqa: E402
from perceptionguard.tracking.tracker import Tracker  # noqa: E402

CLASS_HEIGHTS = {"vehicle": 1.6, "pedestrian": 1.75, "cyclist": 1.7}
N = 8


def _frames(scene: str = "multi", n: int = N):
    intr = default_intrinsics()
    return [
        render_frame(build_scene(scene, t, n), intr, camera_pose(t, n), index=t)
        for t in range(n)
    ], intr


def _pipeline(frames):
    monitor = ReliabilityMonitor(CLASS_HEIGHTS)
    monitor.reference = monitor.fit_reference([f.image for f in frames])
    return PerceptionPipeline(ColorModelDetector(appearances()), Tracker(), monitor)


class TestMonitorGuards(unittest.TestCase):
    def test_update_before_fit_reference_raises(self) -> None:
        # Scoring against an unfitted baseline would silently produce numbers
        # that look valid, which is worse than crashing.
        frames, intr = _frames(n=2)
        pipe = PerceptionPipeline(
            ColorModelDetector(appearances()),
            Tracker(),
            ReliabilityMonitor(CLASS_HEIGHTS),
        )
        with self.assertRaises(RuntimeError):
            pipe.process(image=frames[0].image, depth=frames[0].depth, intrinsics=intr)

    def test_class_heights_are_required(self) -> None:
        with self.assertRaises(ValueError):
            ReliabilityMonitor({})

    def test_fit_reference_needs_frames(self) -> None:
        with self.assertRaises(ValueError):
            ReliabilityMonitor(CLASS_HEIGHTS).fit_reference([])


class TestCleanRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frames, cls.intr = _frames()
        cls.pipe = _pipeline(cls.frames)
        cls.outs = [
            cls.pipe.process(
                image=f.image, depth=f.depth, intrinsics=cls.intr, frame_index=f.index
            )
            for f in cls.frames
        ]

    def test_reliability_is_a_probability(self) -> None:
        for o in self.outs:
            self.assertGreaterEqual(o.report.score, 0.0)
            self.assertLessEqual(o.report.score, 1.0)

    def test_status_is_one_of_three_levels(self) -> None:
        for o in self.outs:
            self.assertIn(o.report.status, {"TRUSTED", "CAUTION", "DEGRADED"})

    def test_objects_are_detected_and_tracked(self) -> None:
        self.assertTrue(any(len(o.estimates) > 0 for o in self.outs))

    def test_3d_positions_are_in_front_of_the_camera(self) -> None:
        # +Z forward in the OpenCV camera frame. A negative Z means a sign
        # error somewhere in the lift, which is exactly the class of bug that
        # silently produces plausible-looking 2D output.
        seen = 0
        for o in self.outs:
            for e in o.estimates:
                if e.center_3d is not None and np.isfinite(e.center_3d).all():
                    self.assertGreater(float(e.center_3d[2]), 0.0)
                    seen += 1
        self.assertGreater(seen, 0, "no 3D estimates produced")

    def test_every_stage_is_timed(self) -> None:
        for key in ("detect", "track", "lift3d", "monitor", "total"):
            self.assertIn(key, self.outs[0].timings_ms)

    def test_signals_are_bounded(self) -> None:
        for o in self.outs:
            for name, value in o.report.signals.items():
                self.assertGreaterEqual(value, 0.0, name)
                self.assertLessEqual(value, 1.0, name)


class TestDegradationLowersReliability(unittest.TestCase):
    """The single claim the whole project rests on.

    Uses patch_occlusion, which the sweep measured as one of the degradations
    the monitor responds to most strongly (reliability 0.942 clean -> 0.767 at
    the lowest severity). Blur is deliberately NOT used here: the colour-model
    reference detector is largely blur-tolerant, so sharpness is measurably
    anti-correlated with failure for this backend. Asserting otherwise would
    encode a claim the data does not support.
    """

    def test_occlusion_reduces_mean_reliability(self) -> None:
        frames, intr = _frames()
        pipe = _pipeline(frames)

        clean = []
        for f in frames:
            clean.append(
                pipe.process(
                    image=f.image, depth=f.depth, intrinsics=intr, frame_index=f.index
                ).report.score
            )

        pipe.reset()
        degraded = []
        for f in frames:
            d = apply_degradation(
                "patch_occlusion", 1.0, f.image, f.depth, intr, frame_index=f.index
            )
            degraded.append(
                pipe.process(
                    image=d.image,
                    depth=d.depth,
                    intrinsics=d.intrinsics,
                    frame_index=f.index,
                ).report.score
            )

        self.assertLess(
            float(np.mean(degraded)),
            float(np.mean(clean)),
            "monitor did not respond to heavy occlusion",
        )

    def test_a_diagnosis_is_offered_when_degraded(self) -> None:
        frames, intr = _frames()
        pipe = _pipeline(frames)
        f = frames[-1]
        d = apply_degradation(
            "patch_occlusion", 1.0, f.image, f.depth, intr, frame_index=1
        )
        out = pipe.process(
            image=d.image, depth=d.depth, intrinsics=d.intrinsics, frame_index=1
        )
        if out.report.status != "TRUSTED":
            self.assertTrue(out.report.diagnosis, "degraded frame with no diagnosis")


class TestReset(unittest.TestCase):
    def test_reset_clears_track_state(self) -> None:
        frames, intr = _frames()
        pipe = _pipeline(frames)
        for f in frames:
            pipe.process(
                image=f.image, depth=f.depth, intrinsics=intr, frame_index=f.index
            )
        pipe.reset()
        self.assertEqual(len(pipe.tracker.tracks), 0)


if __name__ == "__main__":
    unittest.main()
