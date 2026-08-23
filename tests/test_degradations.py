"""Unit tests for the controlled degradation suite.

The experiments are only meaningful if each degradation is *reproducible* and
*monotone in severity*. Those two properties are what these tests protect.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perceptionguard.data.degradations import (  # noqa: E402
    DEGRADATION_CHANNEL,
    SEVERITIES,
    apply_degradation,
    degradation_names,
)
from perceptionguard.data.scenes import (  # noqa: E402
    build_scene,
    camera_pose,
    default_intrinsics,
)
from perceptionguard.data.synthetic import render_frame  # noqa: E402


def _frame():
    intr = default_intrinsics()
    f = render_frame(build_scene("multi", 4, 24), intr, camera_pose(4, 24), index=4)
    return f.image, f.depth, intr


class TestRegistry(unittest.TestCase):
    def test_expected_degradations_present(self) -> None:
        names = set(degradation_names())
        for expected in (
            "none",
            "motion_blur",
            "low_light",
            "noise",
            "glare",
            "patch_occlusion",
            "distribution_shift",
            "depth_degradation",
            "calibration_error",
            "frame_delay",
        ):
            self.assertIn(expected, names)

    def test_every_degradation_declares_a_channel(self) -> None:
        # The channel is what lets the evaluation attribute a reliability drop
        # to the right part of the pipeline. A missing one is a silent gap.
        for name in degradation_names():
            self.assertIn(name, DEGRADATION_CHANNEL)

    def test_severity_grid_spans_clean_to_worst(self) -> None:
        self.assertEqual(SEVERITIES[0], 0.0)
        self.assertEqual(SEVERITIES[-1], 1.0)


class TestReproducibility(unittest.TestCase):
    def test_identical_inputs_give_identical_output(self) -> None:
        img, depth, intr = _frame()
        for name in degradation_names():
            a = apply_degradation(name, 0.7, img, depth, intr, frame_index=3)
            b = apply_degradation(name, 0.7, img, depth, intr, frame_index=3)
            np.testing.assert_array_equal(a.image, b.image, err_msg=name)

    def test_frame_index_changes_the_noise_draw(self) -> None:
        # Seeded per (name, severity, frame_index): different frames must not
        # receive an identical noise field, or temporal signals see a
        # suspiciously static input.
        img, depth, intr = _frame()
        a = apply_degradation("noise", 1.0, img, depth, intr, frame_index=0)
        b = apply_degradation("noise", 1.0, img, depth, intr, frame_index=1)
        self.assertFalse(np.array_equal(a.image, b.image))

    def test_severity_is_clipped_to_unit_range(self) -> None:
        img, depth, intr = _frame()
        self.assertEqual(
            apply_degradation("noise", 5.0, img, depth, intr).severity, 1.0
        )
        self.assertEqual(
            apply_degradation("noise", -2.0, img, depth, intr).severity, 0.0
        )

    def test_unknown_name_raises(self) -> None:
        img, depth, intr = _frame()
        with self.assertRaises(KeyError):
            apply_degradation("chromatic_aberration", 0.5, img, depth, intr)


class TestSeverityIsMonotone(unittest.TestCase):
    def test_none_at_zero_is_the_identity(self) -> None:
        img, depth, intr = _frame()
        out = apply_degradation("none", 0.0, img, depth, intr)
        np.testing.assert_array_equal(out.image, img)

    def test_motion_blur_reduces_high_frequency_energy(self) -> None:
        img, depth, intr = _frame()
        sharp = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        prev = sharp
        for sev in (0.25, 0.5, 0.75, 1.0):
            out = apply_degradation("motion_blur", sev, img, depth, intr)
            cur = cv2.Laplacian(
                cv2.cvtColor(out.image, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()
            self.assertLessEqual(cur, prev + 1e-9, f"blur not monotone at sev={sev}")
            prev = cur
        self.assertLess(prev, sharp)

    def test_low_light_reduces_luminance(self) -> None:
        img, depth, intr = _frame()
        prev = float(img.mean())
        for sev in (0.25, 0.5, 0.75, 1.0):
            cur = float(
                apply_degradation("low_light", sev, img, depth, intr).image.mean()
            )
            self.assertLessEqual(cur, prev + 1e-9)
            prev = cur

    def test_glare_adds_clipped_pixels(self) -> None:
        img, depth, intr = _frame()
        out = apply_degradation("glare", 1.0, img, depth, intr)
        self.assertGreater((out.image >= 254).mean(), (img >= 254).mean())


class TestChannelsAreRespected(unittest.TestCase):
    def test_calibration_error_perturbs_intrinsics_only(self) -> None:
        img, depth, intr = _frame()
        out = apply_degradation("calibration_error", 1.0, img, depth, intr)
        self.assertNotAlmostEqual(out.intrinsics.fx, intr.fx)
        # The image is untouched: the whole point is a calibration fault that
        # is invisible to any image-quality signal.
        np.testing.assert_array_equal(out.image, img)

    def test_depth_degradation_touches_depth_not_image(self) -> None:
        img, depth, intr = _frame()
        out = apply_degradation("depth_degradation", 1.0, img, depth, intr)
        np.testing.assert_array_equal(out.image, img)
        self.assertFalse(np.allclose(out.depth, depth, equal_nan=True))

    def test_patch_occlusion_also_removes_depth(self) -> None:
        # An occluder that hides RGB but leaves depth intact would be physically
        # impossible and would make the depth signals look better than reality.
        img, depth, intr = _frame()
        out = apply_degradation("patch_occlusion", 1.0, img, depth, intr)
        self.assertGreater(
            np.isnan(out.depth).sum(), np.isnan(depth).sum(), "depth not occluded"
        )

    def test_frame_delay_reports_a_time_scale(self) -> None:
        img, depth, intr = _frame()
        out = apply_degradation("frame_delay", 1.0, img, depth, intr, frame_index=1)
        self.assertGreater(out.dt_scale, 1.0)
        self.assertTrue(out.stale)


if __name__ == "__main__":
    unittest.main()
