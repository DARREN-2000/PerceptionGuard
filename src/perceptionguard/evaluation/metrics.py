"""Evaluation primitives: detection matching, error metrics, ranking metrics.

Implemented from scratch because the sandbox has no scikit-learn/scipy. Each
function is small enough to unit-test against hand-computed values, which is
done in ``tests/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..tracking.tracker import iou_xyxy

__all__ = [
    "MatchResult",
    "match_boxes",
    "prf",
    "auroc",
    "rankdata",
    "spearman",
    "pearson",
    "summarize",
]


@dataclass
class MatchResult:
    matches: list[tuple[int, int, float]]  # (gt_idx, pred_idx, iou)
    unmatched_gt: list[int]
    unmatched_pred: list[int]

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fn(self) -> int:
        return len(self.unmatched_gt)

    @property
    def fp(self) -> int:
        return len(self.unmatched_pred)


def match_boxes(
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    *,
    gt_labels: Sequence[str] | None = None,
    pred_labels: Sequence[str] | None = None,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy highest-IoU one-to-one matching, optionally label-aware.

    Greedy rather than optimal: with a handful of well-separated objects the
    assignment coincides with the optimal one, and greedy keeps the metric
    trivially reproducible.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            if (
                gt_labels is not None
                and pred_labels is not None
                and gt_labels[i] != pred_labels[j]
            ):
                continue
            iou = iou_xyxy(gb, pb)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        matches.append((i, j, float(iou)))
        used_g.add(i)
        used_p.add(j)
    return MatchResult(
        matches=matches,
        unmatched_gt=[i for i in range(len(gt_boxes)) if i not in used_g],
        unmatched_pred=[j for j in range(len(pred_boxes)) if j not in used_p],
    )


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 with 0/0 defined as 0."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return float(p), float(r), float(f)


def rankdata(x: Sequence[float]) -> np.ndarray:
    """Ranks with ties averaged (needed for both AUROC and Spearman)."""
    a = np.asarray(x, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC via the Mann-Whitney U statistic, tie-corrected.

    ``scores`` should be *higher for the positive class*. Returns NaN when one
    class is absent, rather than a misleading 0.5 -- a degenerate label column is
    a fact about the experiment, not a performance number.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    u = r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation. Used for 'does reliability fall as severity rises?'

    Rank-based on purpose: the relationship between severity and reliability is
    monotone but definitely not linear, so Pearson would understate it.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def summarize(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }
