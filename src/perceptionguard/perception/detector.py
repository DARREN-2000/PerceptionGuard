"""Object detection front-end.

Design note (the thing to defend in an interview):
``Detector`` is a Protocol, and the rest of the pipeline depends only on it.
The reliability monitor, geometry stage, tracker and evaluation harness never
import a concrete detector. That is what makes the interesting claim of this
project -- "fused reliability signals beat raw detector confidence" -- portable
across detectors rather than tied to one. ``TorchDetector`` (see
``inference/``) plugs into the same seam on a GPU host.

``ColorModelDetector`` is the offline reference backend. It is an appearance-model
detector: it is given the palette of object hues (weak supervision, NOT
ground-truth boxes and not the instance map) and segments in CIELAB. It is not a
neural detector and its score is not a learned probability.

Why it is still a legitimate experimental subject: the monitor's job is to predict
when *the detector it wraps* is wrong. This detector degrades for the physically
right reasons -- blur destroys boundary contrast, darkness collapses chroma, noise
fragments regions, occlusion cuts fill ratio. So the correlation between fused
signals and true error is a real measurement. What it cannot tell you is the
absolute mAP of a modern detector, or how a learned confidence head would rank.
Re-run the same harness with real weights before quoting numbers as
detector-general.

Confidence weights are fixed a priori from physical reasoning and are not tuned
against evaluation results, so the calibration study stays honest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

__all__ = ["Detection", "Detector", "ColorModelDetector", "ObjectAppearance"]


@dataclass
class Detection:
    """A single 2D detection, plus the raw cues that produced its score.

    ``cues`` is carried forward deliberately: the reliability monitor consumes
    these low-level quantities instead of re-deriving them, which keeps
    per-signal ablation possible.
    """

    bbox: tuple[float, float, float, float]  # xyxy
    score: float
    label: str
    mask_area: int = 0
    cues: dict[str, float] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(self.bbox[2] - self.bbox[0], 0.0)

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 0.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )


@runtime_checkable
class Detector(Protocol):
    """Everything downstream depends on this and nothing more."""

    name: str

    def detect(self, image: np.ndarray) -> list[Detection]: ...


@dataclass(frozen=True)
class ObjectAppearance:
    """The appearance prior for one object class."""

    label: str
    color_bgr: tuple[int, int, int]


class ColorModelDetector:
    """CIELAB hue-angle appearance detector with physically-motivated confidence.

    Matching is on HUE ANGLE, gated by a minimum chroma.

    Measured justification (``scripts/diag_color.py``): Lambertian shading scales
    surface radiance by 0.35-1.0, which moves Lab chroma *magnitude* a lot --
    ``|ab - ref|`` measured 9.2-26.9 across differently-shaded faces of the same
    object, so a Euclidean ab threshold cannot separate "different object" from
    "same object, darker face". Hue angle under that same shading moved by only
    0.1-0.7 deg. Hue is the invariant; chroma magnitude is illumination-coupled.

    ``min_chroma`` rejects neutral surfaces: the occluder and background measured
    chroma ~1.0 against 38-77 for palette objects, so the gate sits far from
    either class boundary.

    This deliberately does NOT make the detector immune to the illumination
    degradation. As the image darkens, chroma collapses toward the neutral axis
    and pixels fall below ``min_chroma``, so low-light frames still lose colour
    evidence -- the real physical effect the monitor must cope with. Chroma is
    exported as a cue so the monitor can observe that collapse directly.
    """

    name = "color_model_v1"

    # Fixed a priori. Rationale for each weight:
    #  colour   - primary evidence that this is the object at all
    #  contrast - boundary sharpness; the first thing motion blur destroys
    #  fill     - convexity/completeness; drops under occlusion and noise
    #  size     - small objects are intrinsically less reliable
    W_COLOR, W_CONTRAST, W_FILL, W_SIZE = 0.40, 0.25, 0.20, 0.15
    SCORE_FLOOR = 0.15  # a returned detection is never fully unconfident

    def __init__(
        self,
        appearances: Sequence[ObjectAppearance],
        *,
        hue_threshold_deg: float = 25.0,
        min_chroma: float = 8.0,
        min_area: int = 40,
        open_kernel: int = 3,
        hue_tau: float = 12.0,
        min_lightness: float = 12.0,
    ) -> None:
        if not appearances:
            raise ValueError("ColorModelDetector needs at least one appearance prior")
        self.appearances = list(appearances)
        self.hue_threshold_deg = float(hue_threshold_deg)
        self.min_chroma = float(min_chroma)
        self.min_area = int(min_area)
        self.hue_tau = float(hue_tau)
        self.min_lightness = float(min_lightness)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )

        self._ref_hue: dict[str, float] = {}
        self._ref_chroma: dict[str, float] = {}
        for a in self.appearances:
            lab_ref = cv2.cvtColor(
                np.array([[a.color_bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB
            )[0, 0].astype(np.float64)
            ab = lab_ref[1:] - 128.0
            self._ref_hue[a.label] = float(np.degrees(np.arctan2(ab[1], ab[0])))
            self._ref_chroma[a.label] = float(max(np.linalg.norm(ab), 1e-6))

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR image, got {image.shape}")

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Gradient magnitude is the boundary-contrast evidence source.
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)

        ab_img = lab[:, :, 1:] - 128.0
        chroma_img = np.linalg.norm(ab_img, axis=2)
        hue_img = np.degrees(np.arctan2(ab_img[:, :, 1], ab_img[:, :, 0]))
        lum_ok = lab[:, :, 0] >= self.min_lightness
        chroma_ok = chroma_img >= self.min_chroma

        detections: list[Detection] = []
        for app in self.appearances:
            # Wrapped angular difference in [0, 180].
            d_hue = np.abs((hue_img - self._ref_hue[app.label] + 180.0) % 360.0 - 180.0)
            mask = ((d_hue < self.hue_threshold_deg) & chroma_ok & lum_ok).astype(
                np.uint8
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            if not mask.any():
                continue

            n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for k in range(1, n_lbl):
                x, y, w, h, area = (int(v) for v in stats[k])
                if area < self.min_area or w < 3 or h < 3:
                    continue

                comp = lbl_map[y : y + h, x : x + w] == k

                fill = float(area) / float(w * h)
                mean_hue_dist = float(d_hue[y : y + h, x : x + w][comp].mean())
                mean_chroma = float(chroma_img[y : y + h, x : x + w][comp].mean())
                # Hue agreement is identity evidence; chroma ratio reports how
                # much colour signal survives the current illumination.
                hue_score = float(np.exp(-mean_hue_dist / self.hue_tau))
                chroma_ratio = float(
                    np.clip(mean_chroma / self._ref_chroma[app.label], 0.0, 1.0)
                )
                color_score = float(hue_score * (0.5 + 0.5 * chroma_ratio))

                # Contrast measured on the component boundary only: interior
                # gradient is near zero for flat-shaded faces and would dilute
                # the signal we actually care about.
                cu = comp.astype(np.uint8)
                border = cv2.dilate(cu, self._kernel) - cv2.erode(cu, self._kernel)
                bmask = border.astype(bool)
                g_roi = grad[y : y + h, x : x + w]
                contrast_raw = float(g_roi[bmask].mean()) if bmask.any() else 0.0
                contrast_score = float(np.clip(contrast_raw / 90.0, 0.0, 1.0))

                size_score = float(np.clip(np.sqrt(area) / 26.0, 0.0, 1.0))

                quality = (
                    self.W_COLOR * color_score
                    + self.W_CONTRAST * contrast_score
                    + self.W_FILL * fill
                    + self.W_SIZE * size_score
                )
                score = float(
                    np.clip(
                        self.SCORE_FLOOR + (1.0 - self.SCORE_FLOOR) * quality, 0.0, 1.0
                    )
                )

                detections.append(
                    Detection(
                        bbox=(float(x), float(y), float(x + w - 1), float(y + h - 1)),
                        score=score,
                        label=app.label,
                        mask_area=area,
                        cues={
                            "color_score": color_score,
                            "hue_score": hue_score,
                            "chroma_ratio": chroma_ratio,
                            "mean_chroma": mean_chroma,
                            "mean_hue_dist": mean_hue_dist,
                            "contrast_score": contrast_score,
                            "contrast_raw": contrast_raw,
                            "fill_ratio": fill,
                            "size_score": size_score,
                        },
                    )
                )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
