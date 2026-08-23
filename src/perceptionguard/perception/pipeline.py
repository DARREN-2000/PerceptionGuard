"""End-to-end perception pipeline: image/depth -> tracks -> 3D -> reliability.

Deliberately a plain sequential class, not a graph framework or a set of
microservices. The stages are strictly ordered and run in one process on one
frame; introducing queues or services here would add failure modes without
adding capability, and would make the latency numbers meaningless.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..geometry.camera import CameraIntrinsics, unproject_pixels
from ..reliability.monitor import ReliabilityMonitor, ReliabilityReport
from ..tracking.tracker import Track, Tracker
from .detector import Detection, Detector

__all__ = ["TrackEstimate", "PipelineOutput", "PerceptionPipeline"]


@dataclass
class TrackEstimate:
    """A track with its 3D interpretation."""

    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    score: float
    center_3d: np.ndarray | None = None  # (3,) camera frame
    depth: float = float("nan")
    depth_valid_frac: float = 0.0


@dataclass
class PipelineOutput:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    estimates: list[TrackEstimate] = field(default_factory=list)
    report: ReliabilityReport | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class PerceptionPipeline:
    """Detector -> tracker -> 3D lifting -> reliability monitor."""

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        monitor: ReliabilityMonitor,
        *,
        min_depth_samples: int = 8,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.monitor = monitor
        self.min_depth_samples = int(min_depth_samples)

    def reset(self) -> None:
        self.tracker.reset()

    @staticmethod
    def _robust_depth(
        depth: np.ndarray, bbox: Sequence[float], min_samples: int
    ) -> tuple[float, float]:
        """Median depth inside a box, plus the valid fraction.

        The median is used rather than the mean because a box always contains
        some background pixels at a very different range; a mean would be pulled
        toward them and bias every 3D estimate outward.
        """
        h, w = depth.shape[:2]
        x0 = max(int(round(bbox[0])), 0)
        y0 = max(int(round(bbox[1])), 0)
        x1 = min(int(round(bbox[2])), w - 1)
        y1 = min(int(round(bbox[3])), h - 1)
        if x1 <= x0 or y1 <= y0:
            return float("nan"), 0.0
        patch = depth[y0 : y1 + 1, x0 : x1 + 1].astype(np.float64)
        finite = np.isfinite(patch)
        frac = float(finite.mean()) if patch.size else 0.0
        if int(finite.sum()) < min_samples:
            return float("nan"), frac
        return float(np.median(patch[finite])), frac

    def process(
        self,
        *,
        image: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        frame_index: int = 0,
        dt: float = 0.05,
    ) -> PipelineOutput:
        import time

        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        detections = self.detector.detect(image)
        timings["detect"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        tracks = self.tracker.update(detections)
        timings["track"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        estimates: list[TrackEstimate] = []
        for tr in tracks:
            z, frac = self._robust_depth(depth, tr.bbox, self.min_depth_samples)
            center_3d = None
            if np.isfinite(z) and z > 0:
                cx = (tr.bbox[0] + tr.bbox[2]) / 2.0
                cy = (tr.bbox[1] + tr.bbox[3]) / 2.0
                # Lift the box centre. Note this estimates the centre of the
                # VISIBLE SURFACE, not the object centroid -- it is biased toward
                # the camera by roughly half the object depth. The evaluation
                # reports this bias rather than hiding it behind a fudge factor.
                center_3d = unproject_pixels(
                    np.array([[cx, cy]]), np.array([z]), intrinsics
                )[0]
                tr.center3d = center_3d
                tr._push(tr.depth_history, z)
            estimates.append(
                TrackEstimate(
                    track_id=tr.track_id,
                    label=tr.label,
                    bbox=tr.bbox,
                    score=tr.score,
                    center_3d=center_3d,
                    depth=z,
                    depth_valid_frac=frac,
                )
            )
        timings["lift3d"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        report = self.monitor.update(
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            detections=detections,
            tracks=tracks,
            dt=dt,
        )
        timings["monitor"] = (time.perf_counter() - t0) * 1e3
        timings["total"] = sum(timings.values())

        return PipelineOutput(
            frame_index=frame_index,
            detections=detections,
            tracks=tracks,
            estimates=estimates,
            report=report,
            timings_ms=timings,
        )
