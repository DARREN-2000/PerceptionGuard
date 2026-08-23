"""Runtime perception reliability monitor.

The question this answers: *can the perception output be trusted right now?*

Design commitments, and why:

1. SIGNALS ARE INDEPENDENT AND TYPED BY CHANNEL. Image-quality signals cannot
   see a bad extrinsic/intrinsic calibration; depth signals cannot see hue
   shift. Keeping them separate is what lets the ablation study report *which*
   signal catches *which* failure, instead of one opaque number.

2. FUSION IS A WEIGHTED GEOMETRIC MEAN, not arithmetic. For a safety monitor,
   one collapsed signal must be able to drag the verdict down even when
   everything else looks fine -- an arithmetic mean lets 9 healthy signals
   outvote a catastrophic one. The geometric mean is the standard choice when
   signals are closer to multiplicative reliabilities than additive scores.

3. REFERENCE STATISTICS ARE FITTED ON CLEAN DATA, not hardcoded. Absolute
   Laplacian variance is meaningless across scenes; the ratio to a clean
   baseline is not. ``fit_reference`` must be called (or loaded) before scores
   are comparable, and the code refuses to silently invent a baseline.

4. THRESHOLDS ARE PLACEHOLDERS UNTIL CALIBRATED. The defaults below are NOT
   claimed to be correct; ``scripts/calibrate_monitor.py`` overrides them
   from measured error distributions. They exist only so the pipeline runs.

Known blind spots, stated up front because the evaluation confirms them:
  * A pure unproject/reproject round-trip is self-consistent by construction and
    carries zero information about calibration correctness, so it is NOT used as
    a signal. Calibration error is instead caught by the size-vs-depth residual,
    which requires an object-size prior.
  * With a single camera and no target, a globally-consistent but wrong
    calibration is only observable through such priors. If the prior is wrong,
    the monitor is wrong in the same direction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry.camera import CameraIntrinsics
from ..perception.detector import Detection
from ..tracking.tracker import Track

__all__ = [
    "ReferenceStats",
    "ReliabilityReport",
    "ReliabilityMonitor",
    "SIGNAL_CAUSES",
]

_EPS = 1e-6
_FEATURE_WIDTH = 320  # width at which image-quality features are computed

# Human-readable cause attached to each signal when it is the dominant deficit.
SIGNAL_CAUSES: dict[str, str] = {
    "confidence": "Low detector confidence",
    "sharpness": "Motion blur or defocus",
    "exposure": "Poor illumination (under-exposed)",
    "clipping": "Glare / sensor saturation",
    "noise": "Sensor noise",
    "chroma": "Colour information lost",
    "ood": "Distribution shift (input unlike calibration set)",
    "temporal": "Unstable tracking / temporal disagreement",
    "motion": "Erratic object motion (large filter innovation)",
    "depth_valid": "Missing depth returns",
    "depth_consistency": "Inconsistent depth within objects",
    "geometry": "Geometric inconsistency (calibration or depth scale error)",
    "timing": "Frame delay / dropped frames",
}


@dataclass
class ReferenceStats:
    """Clean-condition baselines. Fitted, never guessed."""

    lap_var: float = 1.0
    mean_lum: float = 128.0
    noise: float = 1.0
    chroma: float = 1.0
    dt: float = 0.05
    feat_mean: np.ndarray | None = None
    feat_std: np.ndarray | None = None
    fitted: bool = False


@dataclass
class ReliabilityReport:
    score: float
    status: str
    signals: dict[str, float] = field(default_factory=dict)  # normalized [0,1]
    raw: dict[str, float] = field(default_factory=dict)  # physical units
    deficits: dict[str, float] = field(default_factory=dict)
    diagnosis: list[str] = field(default_factory=list)
    n_tracks: int = 0

    @property
    def trusted(self) -> bool:
        return self.status == "TRUSTED"


class ReliabilityMonitor:
    """Fuses per-frame perception health signals into one reliability score."""

    # Placeholders -- overridden by calibration. See module docstring.
    THRESH_TRUSTED = 0.75
    THRESH_CAUTION = 0.45

    DEFAULT_WEIGHTS: dict[str, float] = {
        "confidence": 1.0,
        "sharpness": 1.0,
        "exposure": 0.8,
        "clipping": 0.8,
        "noise": 0.8,
        "chroma": 0.6,
        "ood": 1.0,
        "temporal": 1.0,
        "motion": 0.6,
        "depth_valid": 0.8,
        "depth_consistency": 0.8,
        "geometry": 1.2,  # highest: it is the only signal that sees calibration
        "timing": 0.6,
    }

    def __init__(
        self,
        class_heights: Mapping[str, float],
        *,
        weights: Mapping[str, float] | None = None,
        reference: ReferenceStats | None = None,
        enabled_signals: Sequence[str] | None = None,
    ) -> None:
        if not class_heights:
            raise ValueError("class_heights prior is required for the geometry signal")
        self.class_heights = dict(class_heights)
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        self.reference = reference or ReferenceStats()
        # Signal subset, for the ablation study.
        self.enabled = (
            list(enabled_signals) if enabled_signals is not None else list(self.weights)
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def image_features(image: np.ndarray) -> dict[str, float]:
        """Cheap global image-quality descriptors (all O(HW), no model needed)."""
        # Downsample before computing global statistics. Measured: the monitor
        # was 24.7 ms of a 59.2 ms frame (42% of budget), dominated by per-pixel
        # Laplacian/median/LAB conversions. Global image-quality statistics do
        # not need full resolution. Safe because fit_reference() runs through
        # this same function, so baseline and runtime values stay commensurate;
        # absolute Laplacian variance changes with scale, the RATIO does not.
        if image.shape[1] > _FEATURE_WIDTH:
            scale = _FEATURE_WIDTH / float(image.shape[1])
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Blur: variance of the Laplacian. Falls with any low-pass corruption.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Exposure and saturation.
        mean_lum = float(gray.mean())
        frac_clipped = float((gray >= 250).mean())
        frac_black = float((gray <= 5).mean())

        # Noise: MAD of the residual after a median filter. A median filter
        # preserves edges, so the residual is dominated by noise rather than by
        # scene structure -- this is why it is used instead of a Gaussian.
        med = cv2.medianBlur(gray, 3)
        resid = gray.astype(np.float64) - med.astype(np.float64)
        noise = float(1.4826 * np.median(np.abs(resid - np.median(resid))))

        # Colour content.
        ab = lab[:, :, 1:].astype(np.float64) - 128.0
        chroma_map = np.linalg.norm(ab, axis=2)
        chroma = float(chroma_map.mean())
        # Circular mean of hue, weighted by chroma: the OOD-relevant summary.
        hue = np.arctan2(ab[:, :, 1], ab[:, :, 0])
        w = chroma_map + _EPS
        hue_cos = float((np.cos(hue) * w).sum() / w.sum())
        hue_sin = float((np.sin(hue) * w).sum() / w.sum())

        return {
            "lap_var": lap_var,
            "mean_lum": mean_lum,
            "frac_clipped": frac_clipped,
            "frac_black": frac_black,
            "noise": noise,
            "chroma": chroma,
            "hue_cos": hue_cos,
            "hue_sin": hue_sin,
        }

    @staticmethod
    def _feature_vector(f: Mapping[str, float]) -> np.ndarray:
        """Vector used for the OOD distance. Log-scaled where heavy-tailed."""
        return np.array(
            [
                np.log(max(f["lap_var"], _EPS)),
                f["mean_lum"],
                f["frac_clipped"],
                np.log(max(f["noise"], _EPS)),
                f["chroma"],
                f["hue_cos"],
                f["hue_sin"],
            ],
            dtype=np.float64,
        )

    def fit_reference(
        self, images: Iterable[np.ndarray], *, dt: float = 0.05
    ) -> ReferenceStats:
        """Fit clean-condition baselines from clean frames.

        Uses a diagonal (per-dimension standardized) distance rather than a full
        covariance: with only tens of clean frames a full 7x7 covariance is
        rank-deficient and its inverse is numerically meaningless. Honest
        limitation -- correlated feature drift is invisible to a diagonal model.
        """
        feats = [self.image_features(im) for im in images]
        if not feats:
            raise ValueError("fit_reference needs at least one clean frame")
        V = np.stack([self._feature_vector(f) for f in feats], axis=0)
        self.reference = ReferenceStats(
            lap_var=float(np.median([f["lap_var"] for f in feats])),
            mean_lum=float(np.median([f["mean_lum"] for f in feats])),
            noise=float(max(np.median([f["noise"] for f in feats]), 0.05)),
            chroma=float(max(np.median([f["chroma"] for f in feats]), _EPS)),
            dt=float(dt),
            feat_mean=V.mean(axis=0),
            feat_std=np.maximum(V.std(axis=0), 1e-3),
            fitted=True,
        )
        return self.reference

    # ------------------------------------------------------------------
    # Per-frame evaluation
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        image: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        detections: Sequence[Detection],
        tracks: Sequence[Track],
        dt: float,
    ) -> ReliabilityReport:
        if not self.reference.fitted:
            raise RuntimeError(
                "ReliabilityMonitor.fit_reference() must be called before update(); "
                "absolute image statistics are not comparable across scenes."
            )
        ref = self.reference
        f = self.image_features(image)
        s: dict[str, float] = {}
        raw: dict[str, float] = dict(f)
        raw["dt"] = float(dt)

        # --- detector confidence ---------------------------------------
        scores = [t.score for t in tracks] or [d.score for d in detections]
        s["confidence"] = float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0
        raw["mean_score"] = float(np.mean(scores)) if scores else 0.0

        # --- image quality ---------------------------------------------
        # Sharpness as a ratio to the clean baseline, sqrt-compressed because
        # Laplacian variance falls roughly with the square of blur radius.
        s["sharpness"] = float(
            np.clip(np.sqrt(f["lap_var"] / max(ref.lap_var, _EPS)), 0.0, 1.0)
        )
        # Exposure penalizes darkness AND wash-out, hence the two-sided form.
        lum_ratio = f["mean_lum"] / max(ref.mean_lum, _EPS)
        s["exposure"] = float(np.clip(1.0 - abs(1.0 - lum_ratio), 0.0, 1.0))
        # Clipping and noise use an EXCESS-OVER-BASELINE exponential knee rather
        # than a ratio. Measured reason: on noiseless synthetic renders the fitted
        # clean noise baseline is ~0.05 DN, so a ratio normalization collapses to
        # ~0 on contact with any noise at all -- the first sweep reported
        # reliability 0.18 at noise severity 0.25 while recall was still 0.958,
        # i.e. a pure false alarm. The knee is referenced to an 8-bit sensor's
        # quantization floor so the signal measures *meaningful* excess noise.
        s["clipping"] = float(np.exp(-f["frac_clipped"] / 0.08))
        noise_floor = max(ref.noise, 1.0)
        noise_excess = max(f["noise"] - noise_floor, 0.0)
        s["noise"] = float(np.exp(-noise_excess / 12.0))
        s["chroma"] = float(np.clip(f["chroma"] / max(ref.chroma, _EPS), 0.0, 1.0))

        # --- OOD: standardized distance from the clean feature distribution --
        v = self._feature_vector(f)
        z = (v - ref.feat_mean) / ref.feat_std
        d_ood = float(np.sqrt(np.mean(z**2)))
        raw["ood_distance"] = d_ood
        # 3 standardized deviations -> ~0 reliability.
        s["ood"] = float(np.clip(1.0 - d_ood / 3.0, 0.0, 1.0))

        # --- temporal --------------------------------------------------
        if tracks:
            iou_means = [float(np.mean(t.iou_history)) for t in tracks if t.iou_history]
            s["temporal"] = (
                float(np.clip(np.mean(iou_means), 0.0, 1.0)) if iou_means else 0.0
            )
            innovs = [
                float(np.mean(t.innov_history)) for t in tracks if t.innov_history
            ]
            mean_innov = float(np.mean(innovs)) if innovs else 0.0
            raw["mean_innovation_px"] = mean_innov
            # 8 px of centre innovation is treated as fully unreliable motion.
            s["motion"] = float(np.exp(-mean_innov / 8.0))
        else:
            s["temporal"] = 0.0
            s["motion"] = 0.0
            raw["mean_innovation_px"] = 0.0

        # --- depth + geometry ------------------------------------------
        valid_fracs: list[float] = []
        disp_rels: list[float] = []
        size_resids: list[float] = []
        for t in tracks:
            x0, y0, x1, y1 = (int(round(v_)) for v_ in t.bbox)
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, depth.shape[1] - 1)
            y1 = min(y1, depth.shape[0] - 1)
            if x1 <= x0 or y1 <= y0:
                continue
            patch = depth[y0 : y1 + 1, x0 : x1 + 1].astype(np.float64)
            if patch.size == 0:
                continue
            finite = np.isfinite(patch)
            valid_fracs.append(float(finite.mean()))
            if finite.sum() < 8:
                continue
            vals = patch[finite]
            med = float(np.median(vals))
            if med <= 0:
                continue
            # Robust relative dispersion. A clean box spans the object's own
            # depth extent, so this is never 0; it is the RATIO to clean that
            # matters, which the normalization below encodes.
            mad = float(1.4826 * np.median(np.abs(vals - med)))
            disp_rels.append(mad / med)

            # Geometric consistency: does apparent size agree with the depth and
            # the class size prior? expected_px = fy * H / z.
            h_prior = self.class_heights.get(t.label)
            if h_prior:
                expected_px = intrinsics.fy * h_prior / med
                measured_px = max(t.x[3], 1e-6)
                size_resids.append(
                    abs(measured_px - expected_px) / max(expected_px, _EPS)
                )

        s["depth_valid"] = (
            float(np.clip(np.mean(valid_fracs), 0.0, 1.0)) if valid_fracs else 0.0
        )
        # 15% relative dispersion treated as fully inconsistent.
        mean_disp = float(np.mean(disp_rels)) if disp_rels else 0.0
        raw["depth_rel_dispersion"] = mean_disp
        s["depth_consistency"] = float(np.exp(-mean_disp / 0.15)) if disp_rels else 0.0
        mean_size_resid = float(np.mean(size_resids)) if size_resids else 0.0
        raw["size_depth_residual"] = mean_size_resid
        # 20% size/depth disagreement treated as fully inconsistent.
        s["geometry"] = float(np.exp(-mean_size_resid / 0.20)) if size_resids else 0.0

        # --- timing ----------------------------------------------------
        s["timing"] = float(np.clip(ref.dt / max(dt, _EPS), 0.0, 1.0))

        # --- fusion ----------------------------------------------------
        active = [k for k in self.enabled if k in s and self.weights.get(k, 0.0) > 0]
        if not active:
            raise ValueError("no active reliability signals")
        wsum = sum(self.weights[k] for k in active)
        log_acc = sum(self.weights[k] * np.log(max(s[k], _EPS)) for k in active)
        score = float(np.exp(log_acc / wsum))

        status = (
            "TRUSTED"
            if score >= self.THRESH_TRUSTED
            else "CAUTION"
            if score >= self.THRESH_CAUTION
            else "DEGRADED"
        )

        # Diagnosis: rank by weight-scaled deficit, so a small drop in a
        # heavily-weighted signal can outrank a large drop in a minor one.
        deficits = {k: self.weights[k] * (1.0 - s[k]) for k in active}
        ranked = sorted(deficits.items(), key=lambda kv: kv[1], reverse=True)
        diagnosis = [SIGNAL_CAUSES.get(k, k) for k, d in ranked[:3] if d > 0.12]

        return ReliabilityReport(
            score=score,
            status=status,
            signals={k: s[k] for k in active},
            raw=raw,
            deficits=deficits,
            diagnosis=diagnosis,
            n_tracks=len(tracks),
        )
