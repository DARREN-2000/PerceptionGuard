"""Unit tests for data association, including the occlusion-split regression."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perceptionguard.perception.detector import Detection  # noqa: E402
from perceptionguard.tracking.tracker import Tracker, iou_xyxy  # noqa: E402


def det(x0, y0, x1, y1, label="vehicle", score=0.9, area=None):
    if area is None:
        area = int((x1 - x0) * (y1 - y0))
    return Detection(
        bbox=(float(x0), float(y0), float(x1), float(y1)),
        score=score,
        label=label,
        mask_area=area,
        cues={},
    )


class TestIou(unittest.TestCase):
    def test_identical(self) -> None:
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint(self) -> None:
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (50, 50, 60, 60)), 0.0)

    def test_hand_computed_half_overlap(self) -> None:
        # 10x10 boxes offset by 5 in x: intersection 5*10=50, union 200-50=150.
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)


class TestIdentityStability(unittest.TestCase):
    def test_constant_velocity_keeps_one_id(self) -> None:
        trk = Tracker()
        seen = set()
        for t in range(12):
            x = 100 + 4 * t
            active = trk.update([det(x, 100, x + 40, 140)])
            seen.update(a.track_id for a in active)
        self.assertEqual(len(seen), 1, f"expected a single ID, got {sorted(seen)}")

    def test_different_labels_never_associate(self) -> None:
        trk = Tracker()
        for _ in range(4):
            trk.update([det(100, 100, 140, 140, label="vehicle")])
        active = trk.update([det(100, 100, 140, 140, label="pedestrian")])
        labels = {a.label for a in active}
        self.assertNotIn(
            "vehicle",
            {a.label for a in active if a.track_id == 1} - labels,
        )
        # The pedestrian detection must not have been absorbed by the vehicle track.
        for a in active:
            self.assertEqual(a.label, "pedestrian")

    def test_short_gap_is_bridged_not_restarted(self) -> None:
        trk = Tracker()
        for t in range(5):
            x = 100 + 4 * t
            trk.update([det(x, 100, x + 40, 140)])
        trk.update([])  # one dropped frame
        active = trk.update([det(124, 100, 164, 140)])
        self.assertEqual([a.track_id for a in active], [1])


class TestOcclusionSplitRegression(unittest.TestCase):
    """Regression test for a measured defect.

    At frame 8 of the "multi" scene a pedestrian passed in front of the vehicle
    and split it into a 12 px and a 7 px sliver separated by a 24 px gap. The
    detector emitted two boxes, neither of which matched the prediction, so a
    duplicate ID was created (object 1 -> ids [1, 3]).

    The fix is deliberately in the tracker, not the detector: from a single
    frame, "one object behind a pole" and "two adjacent objects" are genuinely
    indistinguishable. Only the track carries the prior that resolves it.
    """

    def test_fragments_are_absorbed_by_the_existing_track(self) -> None:
        trk = Tracker()
        for _ in range(5):
            trk.update([det(240, 220, 290, 260)])
        before = [a.track_id for a in trk.update([det(240, 220, 290, 260)])]
        self.assertEqual(before, [1])

        # Occluder splits the object into two slivers.
        active = trk.update(
            [det(240, 220, 255, 260, area=560), det(278, 220, 290, 260, area=440)]
        )
        self.assertEqual(
            [a.track_id for a in active],
            [1],
            "occlusion split must not create a second identity",
        )

    def test_two_genuinely_separate_objects_stay_separate(self) -> None:
        # The guard against over-merging: the union of two far-apart objects
        # does not resemble either track's prediction, so nothing is merged.
        trk = Tracker()
        for _ in range(5):
            trk.update([det(100, 100, 140, 140), det(400, 100, 440, 140)])
        active = trk.update([det(100, 100, 140, 140), det(400, 100, 440, 140)])
        self.assertEqual(len({a.track_id for a in active}), 2)


if __name__ == "__main__":
    unittest.main()
