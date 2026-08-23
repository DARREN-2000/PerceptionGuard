"""Multi-object tracking with a constant-velocity Kalman filter.

Hand-rolled rather than pulled from ``filterpy``/``scipy`` because the sandbox
has no network, and because the residuals produced here are consumed directly by
the reliability monitor -- I want explicit access to the innovation term, not a
library that hides it.

State: ``[cx, cy, w, h, vx, vy]`` with a constant-velocity motion model on the
box centre and a static-size assumption. Measurement: ``[cx, cy, w, h]``.

The size-static assumption is wrong for an approaching object, and that is
intentional and useful: the resulting size innovation grows with approach rate,
which is real information about scene dynamics. It is exposed as a signal rather
than tuned away.

Association is greedy IoU. Not Hungarian: with the object counts here the
optimal assignment almost always coincides, and greedy keeps the failure mode
(identity switch under heavy overlap) simple to reason about when diagnosing
why the monitor missed something.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..perception.detector import Detection

__all__ = ["Track", "Tracker", "iou_matrix", "iou_xyxy"]


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two xyxy boxes."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(
    boxes_a: Sequence[Sequence[float]], boxes_b: Sequence[Sequence[float]]
) -> np.ndarray:
    m = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            m[i, j] = iou_xyxy(a, b)
    return m


def _bbox_to_z(bbox: Sequence[float]) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return np.array(
        [(x0 + x1) / 2.0, (y0 + y1) / 2.0, max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)]
    )


def _z_to_bbox(z: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, w, h = (float(v) for v in z[:4])
    w, h = max(w, 1e-6), max(h, 1e-6)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


@dataclass
class Track:
    """One tracked object and the temporal evidence accumulated about it."""

    track_id: int
    label: str
    x: np.ndarray  # (6,) state
    P: np.ndarray  # (6, 6) covariance
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    score: float = 0.0
    # Rolling temporal evidence, consumed by the reliability monitor.
    iou_history: list[float] = field(default_factory=list)
    innov_history: list[float] = field(default_factory=list)
    size_innov_history: list[float] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)
    depth_history: list[float] = field(default_factory=list)
    center3d: np.ndarray | None = None
    last_cues: dict[str, float] = field(default_factory=dict)
    HISTORY = 12

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _z_to_bbox(self.x)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3

    def _push(self, seq: list[float], value: float) -> None:
        seq.append(float(value))
        if len(seq) > self.HISTORY:
            seq.pop(0)


class Tracker:
    """Greedy-IoU multi-object tracker over a constant-velocity Kalman filter."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.25,
        max_age: int = 6,
        min_hits: int = 3,
        recover_dist: float = 0.9,
        recover_size_ratio: float = 2.0,
        process_pos_var: float = 4.0,
        process_vel_var: float = 1.0,
        process_size_var: float = 4.0,
        meas_pos_var: float = 4.0,
        meas_size_var: float = 9.0,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.recover_dist = float(recover_dist)
        self.recover_size_ratio = float(recover_size_ratio)
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_count = 0

        # Constant velocity on centre; size held static.
        self.F = np.eye(6)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        self.H = np.zeros((4, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.Q = np.diag(
            [
                process_pos_var,
                process_pos_var,
                process_size_var,
                process_size_var,
                process_vel_var,
                process_vel_var,
            ]
        )
        self.R = np.diag([meas_pos_var, meas_pos_var, meas_size_var, meas_size_var])

    def _new_track(self, det: Detection) -> Track:
        z = _bbox_to_z(det.bbox)
        x = np.zeros(6)
        x[:4] = z
        P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0])
        tr = Track(
            track_id=self._next_id,
            label=det.label,
            x=x,
            P=P,
            score=det.score,
            last_cues=dict(det.cues),
        )
        tr._push(tr.score_history, det.score)
        self._next_id += 1
        return tr

    def _predict(self, tr: Track) -> None:
        tr.x = self.F @ tr.x
        tr.P = self.F @ tr.P @ self.F.T + self.Q
        tr.age += 1
        tr.time_since_update += 1
        tr.x[2] = max(tr.x[2], 1e-6)
        tr.x[3] = max(tr.x[3], 1e-6)

    def _update(self, tr: Track, det: Detection, iou: float) -> None:
        z = _bbox_to_z(det.bbox)
        y = z - self.H @ tr.x  # innovation
        S = self.H @ tr.P @ self.H.T + self.R
        try:
            K = tr.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = tr.P @ self.H.T @ np.linalg.pinv(S)
        tr.x = tr.x + K @ y
        tr.P = (np.eye(6) - K @ self.H) @ tr.P
        tr.x[2] = max(tr.x[2], 1e-6)
        tr.x[3] = max(tr.x[3], 1e-6)

        tr.hits += 1
        tr.time_since_update = 0
        tr.score = det.score
        tr.label = det.label
        tr.last_cues = dict(det.cues)
        tr._push(tr.iou_history, iou)
        tr._push(tr.innov_history, float(np.linalg.norm(y[:2])))
        tr._push(tr.size_innov_history, float(np.linalg.norm(y[2:4])))
        tr._push(tr.score_history, det.score)

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        """Advance one frame. Returns the currently active (confirmed) tracks."""
        self.frame_count += 1
        for tr in self.tracks:
            self._predict(tr)

        dets = list(detections)
        unmatched_d = set(range(len(dets)))
        if self.tracks and dets:
            M = iou_matrix([t.bbox for t in self.tracks], [d.bbox for d in dets])
            # Greedy: repeatedly take the globally best remaining pair.
            pairs = sorted(
                ((M[i, j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1])),
                reverse=True,
            )
            used_t: set[int] = set()
            used_d: set[int] = set()
            for score, i, j in pairs:
                if score < self.iou_threshold:
                    break
                if i in used_t or j in used_d:
                    continue
                if self.tracks[i].label != dets[j].label:
                    continue
                self._update(self.tracks[i], dets[j], float(score))
                used_t.add(i)
                used_d.add(j)
            unmatched_d -= used_d

            # SECOND ASSOCIATION PASS. Measured need: object 1 fragmented into
            # track ids [1, 3] and object 2 into [2, 6]. Cause is a
            # constant-velocity prediction with a static-size assumption, which
            # drifts fastest exactly when an object is approaching -- the
            # predicted box falls under the IoU gate for one frame and a
            # duplicate track is born. A centre-distance gate normalized by box
            # size recovers these WITHOUT loosening the IoU gate globally, which
            # would instead cause identity swaps between neighbouring objects.
            still_t = [i for i in range(len(self.tracks)) if i not in used_t]
            cand: list[tuple[float, int, int]] = []
            for i in still_t:
                tb = self.tracks[i].bbox
                tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                for j in sorted(unmatched_d):
                    if self.tracks[i].label != dets[j].label:
                        continue
                    db = dets[j].bbox
                    dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
                    dw, dh = db[2] - db[0], db[3] - db[1]
                    diag = 0.5 * (float(np.hypot(tw, th)) + float(np.hypot(dw, dh)))
                    if diag <= 0:
                        continue
                    dist = float(np.hypot(tcx - dcx, tcy - dcy)) / diag
                    ratio = max(dw / max(tw, 1e-6), tw / max(dw, 1e-6))
                    if dist <= self.recover_dist and ratio <= self.recover_size_ratio:
                        cand.append((dist, i, j))
            cand.sort()
            used2_t: set[int] = set()
            used2_d: set[int] = set()
            for _, i, j in cand:
                if i in used2_t or j in used2_d:
                    continue
                self._update(
                    self.tracks[i],
                    dets[j],
                    iou_xyxy(self.tracks[i].bbox, dets[j].bbox),
                )
                used2_t.add(i)
                used2_d.add(j)
            unmatched_d -= used2_d

            # THIRD PASS: one track, SEVERAL detections. A thin occluder can
            # split one object into multiple components in a single frame --
            # measured at frame 8 of the "multi" scene, where a pedestrian cut
            # the vehicle (visible_ratio 0.48) into 12 px and 7 px slivers
            # separated by a 24 px gap, spawning a duplicate ID.
            #
            # This is deliberately NOT fixed in the detector: from one frame,
            # "one object split by an occluder" and "two adjacent objects" are
            # genuinely ambiguous, and a gap/fill heuristic cannot separate them.
            # The track supplies the missing prior -- a predicted extent for an
            # object already known to be one entity -- so the union is only
            # accepted when it matches that prediction.
            used3_t: set[int] = set()
            for i in range(len(self.tracks)):
                if i in used_t or i in used2_t:
                    continue
                cands = [
                    j
                    for j in sorted(unmatched_d)
                    if self.tracks[i].label == dets[j].label
                ]
                if len(cands) < 2:
                    continue
                pb = self.tracks[i].bbox
                chosen: list[int] = []
                best_iou = 0.0
                improved = True
                while improved:  # greedy: add a fragment only if the union improves
                    improved = False
                    for j in cands:
                        if j in chosen:
                            continue
                        trial = chosen + [j]
                        ux0 = min(dets[k].bbox[0] for k in trial)
                        uy0 = min(dets[k].bbox[1] for k in trial)
                        ux1 = max(dets[k].bbox[2] for k in trial)
                        uy1 = max(dets[k].bbox[3] for k in trial)
                        cand_iou = iou_xyxy(pb, (ux0, uy0, ux1, uy1))
                        if cand_iou > best_iou + 1e-6:
                            best_iou, chosen, improved = cand_iou, trial, True
                if len(chosen) < 2 or best_iou < self.iou_threshold:
                    continue
                members = [dets[k] for k in chosen]
                area = max(sum(m.mask_area for m in members), 1)
                merged = Detection(
                    bbox=(
                        min(m.bbox[0] for m in members),
                        min(m.bbox[1] for m in members),
                        max(m.bbox[2] for m in members),
                        max(m.bbox[3] for m in members),
                    ),
                    score=float(sum(m.score * m.mask_area for m in members) / area),
                    label=members[0].label,
                    mask_area=int(area),
                    cues={
                        **members[0].cues,
                        "merged_fragments": float(len(members)),
                    },
                )
                self._update(self.tracks[i], merged, best_iou)
                used3_t.add(i)
                unmatched_d -= set(chosen)

        for j in sorted(unmatched_d):
            self.tracks.append(self._new_track(dets[j]))

        # Tracks that missed this frame carry a zero into IoU history so that
        # "recent association quality" degrades on misses, not just on bad boxes.
        for tr in self.tracks:
            if tr.time_since_update > 0:
                tr._push(tr.iou_history, 0.0)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [
            t
            for t in self.tracks
            if t.time_since_update == 0
            and (t.hits >= self.min_hits or self.frame_count <= self.min_hits)
        ]

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self.frame_count = 0
