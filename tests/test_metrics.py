"""Unit tests for the evaluation metrics.

Every expected value here is hand-computed, not captured from a previous run of
the code under test. A regression test that records whatever the code happened
to print is only a change detector; these are correctness tests.

Stdlib ``unittest`` rather than pytest: the development sandbox has no network
and pytest is not installed, so the suite has to run with zero extra packages.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perceptionguard.evaluation.metrics import (  # noqa: E402
    auroc,
    match_boxes,
    pearson,
    prf,
    rankdata,
    spearman,
)


class TestPrf(unittest.TestCase):
    def test_known_values(self) -> None:
        p, r, f1 = prf(tp=3, fp=1, fn=1)
        self.assertAlmostEqual(p, 0.75)
        self.assertAlmostEqual(r, 0.75)
        self.assertAlmostEqual(f1, 0.75)

    def test_asymmetric(self) -> None:
        # precision 8/10 = 0.8, recall 8/16 = 0.5, F1 = 2*.8*.5/1.3
        p, r, f1 = prf(tp=8, fp=2, fn=8)
        self.assertAlmostEqual(p, 0.8)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(f1, 2 * 0.8 * 0.5 / 1.3)

    def test_empty_does_not_divide_by_zero(self) -> None:
        self.assertEqual(prf(tp=0, fp=0, fn=0), (0.0, 0.0, 0.0))


class TestRankdata(unittest.TestCase):
    def test_ties_are_averaged(self) -> None:
        # 20 occupies ranks 2 and 3 -> both get 2.5
        self.assertEqual(list(rankdata([10, 20, 20, 30])), [1.0, 2.5, 2.5, 4.0])

    def test_unsorted_input(self) -> None:
        self.assertEqual(list(rankdata([30, 10, 20])), [3.0, 1.0, 2.0])


class TestAuroc(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        self.assertAlmostEqual(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)

    def test_perfectly_inverted(self) -> None:
        self.assertAlmostEqual(auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_all_tied_is_chance(self) -> None:
        self.assertAlmostEqual(auroc([0.5] * 4, [0, 0, 1, 1]), 0.5)

    def test_hand_computed_partial_overlap(self) -> None:
        # positives {0.4, 0.9}, negatives {0.1, 0.5}. Pairs (pos>neg):
        # 0.4>0.1 yes, 0.4>0.5 no, 0.9>0.1 yes, 0.9>0.5 yes -> 3/4.
        self.assertAlmostEqual(auroc([0.1, 0.4, 0.5, 0.9], [0, 1, 0, 1]), 0.75)

    def test_single_class_is_nan_not_chance(self) -> None:
        # Returning 0.5 here would silently pollute averaged results with a
        # value that looks like a real measurement. NaN propagates visibly.
        self.assertTrue(math.isnan(auroc([0.1, 0.2, 0.3], [1, 1, 1])))
        self.assertTrue(math.isnan(auroc([0.1, 0.2, 0.3], [0, 0, 0])))


class TestCorrelation(unittest.TestCase):
    def test_pearson_perfect_linear(self) -> None:
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)

    def test_pearson_negative(self) -> None:
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_spearman_is_monotone_not_linear(self) -> None:
        # Cubic is monotone but not linear: Spearman sees 1.0, Pearson does not.
        x = [1, 2, 3, 4, 5]
        y = [1, 8, 27, 64, 125]
        self.assertAlmostEqual(spearman(x, y), 1.0)
        self.assertLess(pearson(x, y), 1.0)


class TestMatchBoxes(unittest.TestCase):
    def test_identical_boxes_all_match(self) -> None:
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        res = match_boxes(boxes, boxes)
        self.assertEqual((res.tp, res.fp, res.fn), (2, 0, 0))

    def test_below_threshold_is_not_a_match(self) -> None:
        # IoU of [0,0,10,10] and [9,9,19,19] is far below 0.5.
        res = match_boxes([[0, 0, 10, 10]], [[9, 9, 19, 19]])
        self.assertEqual((res.tp, res.fp, res.fn), (0, 1, 1))

    def test_one_to_one_no_double_counting(self) -> None:
        # Two predictions on one object: one TP, one FP -- never two TPs.
        res = match_boxes([[0, 0, 10, 10]], [[0, 0, 10, 10], [0, 0, 10, 10]])
        self.assertEqual((res.tp, res.fp, res.fn), (1, 1, 0))

    def test_label_awareness(self) -> None:
        boxes = [[0, 0, 10, 10]]
        matched = match_boxes(
            boxes, boxes, gt_labels=["vehicle"], pred_labels=["vehicle"]
        )
        self.assertEqual(matched.tp, 1)
        mismatched = match_boxes(
            boxes, boxes, gt_labels=["vehicle"], pred_labels=["pedestrian"]
        )
        self.assertEqual(mismatched.tp, 0)
        self.assertEqual(mismatched.fn, 1)

    def test_missing_prediction_is_a_false_negative(self) -> None:
        res = match_boxes([[0, 0, 10, 10]], [])
        self.assertEqual((res.tp, res.fp, res.fn), (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
