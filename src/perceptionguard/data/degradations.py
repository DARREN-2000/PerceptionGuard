"""Controlled, reproducible perception degradations.

Every degradation is:
  * PARAMETERIZED by a scalar severity in [0, 1] mapped to physical units,
  * DETERMINISTIC given (name, severity, frame_index) -- no hidden global RNG,
  * SELF-DESCRIBING via ``meta``, so the evaluation harness records exactly what
    was applied and the monitor's diagnosis can be scored against it.

The last point is the reason this module exists rather than a few inline
``cv2.GaussianBlur`` calls: the central experiment asks whether the monitor's
*diagnosis* matches the true cause, which requires the true cause as a label.

Degradations act on different parts of the input on purpose:
  image        -- blur, illumination, noise, glare, patch occlusion, shift
  depth        -- depth noise and dropout
  intrinsics   -- calibration error (pipeline is handed WRONG intrinsics while
                  ground truth stays rendered with the true ones)
  time         -- frame delay / drop
A monitor that only inspects the RGB image cannot see the last three, which is
precisely what the signal-ablation study is meant to expose.
"""

from __future__ import annotations

import hashlib as _hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..geometry.camera import CameraIntrinsics

__all__ = [
    "DegradedInput",
    "Degradation",
    "DEGRADATIONS",
    "apply_degradation",
    "degradation_names",
    "SEVERITIES",
]

SEVERITIES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class DegradedInput:
    """What the pipeline actually receives after corruption."""

    image: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    # True cause label + physical parameters, for scoring the diagnosis.
    name: str = "none"
    severity: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    # Set when the degradation asks the pipeline to reuse the previous frame
    # (dropped frame) so timing signals have something to detect.
    stale: bool = False
    dt_scale: float = 1.0


def _rng(name: str, severity: float, frame_index: int) -> np.random.Generator:
    """Deterministic per-(degradation, severity, frame) RNG.

    Reproducibility matters more than statistical independence across runs: the
    same command must produce the same corrupted frames, or the experiment is
    not repeatable.
    """
    key = f"{name}|{round(float(severity), 6)}|{int(frame_index)}".encode()
    seed = int.from_bytes(_hashlib.blake2b(key, digest_size=8).digest(), "big")
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# Image-domain degradations
# --------------------------------------------------------------------------


def _none(img, depth, intr, sev, idx):
    return img, depth, intr, {}


def _motion_blur(img, depth, intr, sev, idx):
    """Directional blur, emulating camera/platform motion during exposure.

    Kernel length 1..21 px. The angle rotates slowly with frame index so the
    blur direction is not degenerate across a sequence.
    """
    length = int(round(1 + 20 * sev))
    if length <= 1:
        return img, depth, intr, {"kernel_px": 1, "angle_deg": 0.0}
    if length % 2 == 0:
        length += 1
    angle = (idx * 7.0) % 180.0
    k = np.zeros((length, length), dtype=np.float64)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    if s <= 0:
        return img, depth, intr, {"kernel_px": length, "angle_deg": angle}
    k /= s
    return (
        cv2.filter2D(img, -1, k),
        depth,
        intr,
        {
            "kernel_px": length,
            "angle_deg": float(angle),
        },
    )


def _low_light(img, depth, intr, sev, idx):
    """Radiometric darkening: linear gain plus a gamma lift, then quantize.

    Gain 1.0 -> 0.08. Applied in float then re-quantized to uint8 so that the
    quantization loss at low signal is modelled, which is a real part of why
    dark frames hurt: the chroma resolution genuinely disappears.
    """
    gain = 1.0 - 0.92 * sev
    out = np.clip(img.astype(np.float64) * gain, 0, 255).astype(np.uint8)
    return out, depth, intr, {"gain": float(gain)}


def _noise(img, depth, intr, sev, idx):
    """Additive Gaussian sensor noise, sigma 0..30 DN."""
    sigma = 30.0 * sev
    if sigma <= 0:
        return img, depth, intr, {"sigma": 0.0}
    g = _rng("noise", sev, idx)
    n = g.normal(0.0, sigma, img.shape)
    out = np.clip(img.astype(np.float64) + n, 0, 255).astype(np.uint8)
    return out, depth, intr, {"sigma": float(sigma)}


def _glare(img, depth, intr, sev, idx):
    """A saturating light source: additive Gaussian blob driven to clipping.

    Distinct from low_light in that it DESTROYS information by clipping at 255
    rather than compressing it, so recovery is impossible and the monitor
    should treat it differently.
    """
    h, w = img.shape[:2]
    g = _rng("glare", sev, idx)
    cx = float(g.uniform(0.25, 0.75) * w)
    cy = float(g.uniform(0.25, 0.75) * h)
    radius = (0.10 + 0.35 * sev) * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = np.exp(-d2 / (2.0 * radius**2))
    amp = 320.0 * sev
    out = np.clip(img.astype(np.float64) + (blob * amp)[:, :, None], 0, 255).astype(
        np.uint8
    )
    frac_clipped = float((out.max(axis=2) >= 254).mean())
    return (
        out,
        depth,
        intr,
        {
            "center": (cx, cy),
            "radius_px": float(radius),
            "amplitude": float(amp),
            "frac_clipped": frac_clipped,
        },
    )


def _patch_occlusion(img, depth, intr, sev, idx):
    """Opaque neutral patches, emulating lens contamination / foreground clutter.

    Covers 0..35% of the frame across 1-3 patches. Depth is invalidated under the
    patch too: a real occluder blocks the depth sensor as well, and NOT doing
    this would let the depth branch cheat.
    """
    if sev <= 0:
        return img, depth, intr, {"coverage": 0.0, "n_patches": 0}
    h, w = img.shape[:2]
    g = _rng("patch_occlusion", sev, idx)
    n_patches = int(1 + round(2 * sev))
    target = 0.35 * sev
    out = img.copy()
    dout = depth.copy()
    covered = np.zeros((h, w), dtype=bool)
    for _ in range(n_patches):
        area = (target / n_patches) * h * w
        pw = int(np.clip(np.sqrt(area * g.uniform(0.6, 1.6)), 4, w))
        ph = int(np.clip(area / max(pw, 1), 4, h))
        x0 = int(g.integers(0, max(w - pw, 1)))
        y0 = int(g.integers(0, max(h - ph, 1)))
        out[y0 : y0 + ph, x0 : x0 + pw] = (105, 103, 100)
        dout[y0 : y0 + ph, x0 : x0 + pw] = np.nan
        covered[y0 : y0 + ph, x0 : x0 + pw] = True
    return (
        out,
        dout,
        intr,
        {
            "coverage": float(covered.mean()),
            "n_patches": n_patches,
        },
    )


def _distribution_shift(img, depth, intr, sev, idx):
    """Global hue rotation: a genuine covariate shift for this detector.

    This is the honest OOD test for an appearance-model detector, because hue is
    exactly what it keys on -- analogous to a colour-space/domain shift that a
    trained CNN was never exposed to. Rotating hue by up to 40 deg moves the
    input off the appearance manifold WITHOUT changing scene geometry, so the
    ground-truth boxes remain valid and localization error stays measurable.
    """
    shift = 40.0 * sev
    if shift <= 0:
        return img, depth, intr, {"hue_shift_deg": 0.0}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    # OpenCV hue is 0..179 for 0..360 deg.
    hsv[:, :, 0] = (hsv[:, :, 0] + int(round(shift / 2.0))) % 180
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, depth, intr, {"hue_shift_deg": float(shift)}


# --------------------------------------------------------------------------
# Depth-domain
# --------------------------------------------------------------------------


def _depth_degradation(img, depth, intr, sev, idx):
    """Range-dependent depth noise plus random dropout.

    Noise scales with z^2, which is the correct model for stereo/ToF
    triangulation error rather than a constant sigma. Dropout up to 40%
    emulates non-returns on dark or specular surfaces.
    """
    if sev <= 0:
        return img, depth, intr, {"rel_sigma": 0.0, "dropout": 0.0}
    g = _rng("depth", sev, idx)
    out = depth.astype(np.float64).copy()
    valid = np.isfinite(out)
    rel = 0.05 * sev  # 5% of z at z=10m, growing as z^2
    sigma = rel * (out**2) / 10.0
    out[valid] = out[valid] + g.normal(0.0, 1.0, valid.sum()) * sigma[valid]
    dropout = 0.40 * sev
    drop = g.random(out.shape) < dropout
    out[drop] = np.nan
    out[out <= 0] = np.nan
    return (
        img,
        out.astype(np.float32),
        intr,
        {
            "rel_sigma": float(rel),
            "dropout": float(dropout),
        },
    )


# --------------------------------------------------------------------------
# Calibration and timing
# --------------------------------------------------------------------------


def _calibration_error(img, depth, intr, sev, idx):
    """Hand the pipeline WRONG intrinsics; ground truth keeps the true ones.

    Focal length is scaled by up to 8% and the principal point shifted by up to
    12 px, plus a radial distortion term the true camera does not have. This is
    the classic silent failure: the image looks perfect, detection is unaffected,
    and only 3D output is wrong. Any monitor relying on image quality alone will
    miss it entirely -- which is one of the results this project reports.
    """
    if sev <= 0:
        return img, depth, intr, {"focal_scale": 1.0, "pp_shift_px": 0.0, "k1": 0.0}
    focal_scale = 1.0 + 0.08 * sev
    pp_shift = 12.0 * sev
    k1 = 0.12 * sev
    bad = CameraIntrinsics(
        fx=intr.fx * focal_scale,
        fy=intr.fy * focal_scale,
        cx=intr.cx + pp_shift,
        cy=intr.cy - pp_shift,
        width=intr.width,
        height=intr.height,
        dist=(k1, 0.0, 0.0, 0.0, 0.0),
    )
    return (
        img,
        depth,
        bad,
        {
            "focal_scale": float(focal_scale),
            "pp_shift_px": float(pp_shift),
            "k1": float(k1),
        },
    )


def _frame_delay(img, depth, intr, sev, idx):
    """Drop frames / stretch the inter-frame interval.

    Marked via ``stale``/``dt_scale`` rather than modifying pixels: the harness
    reuses the previous frame's input while ground truth advances, so tracking
    and timing signals degrade exactly as they would on an overloaded compute
    budget.
    """
    return img, depth, intr, {"dt_scale": float(1.0 + 4.0 * sev)}


DegradationFn = Callable[
    ..., tuple[np.ndarray, np.ndarray, CameraIntrinsics, dict[str, Any]]
]

DEGRADATIONS: dict[str, DegradationFn] = {
    "none": _none,
    "motion_blur": _motion_blur,
    "low_light": _low_light,
    "noise": _noise,
    "glare": _glare,
    "patch_occlusion": _patch_occlusion,
    "depth_degradation": _depth_degradation,
    "calibration_error": _calibration_error,
    "frame_delay": _frame_delay,
    "distribution_shift": _distribution_shift,
}

# Which channel each degradation attacks -- used when reporting which signals
# *could possibly* have detected it.
DEGRADATION_CHANNEL: dict[str, str] = {
    "none": "none",
    "motion_blur": "image",
    "low_light": "image",
    "noise": "image",
    "glare": "image",
    "patch_occlusion": "image+depth",
    "depth_degradation": "depth",
    "calibration_error": "intrinsics",
    "frame_delay": "time",
    "distribution_shift": "image",
}


def degradation_names() -> list[str]:
    return list(DEGRADATIONS)


def apply_degradation(
    name: str,
    severity: float,
    image: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    frame_index: int = 0,
) -> DegradedInput:
    """Apply one named degradation at one severity."""
    if name not in DEGRADATIONS:
        raise KeyError(
            f"unknown degradation {name!r}; available: {sorted(DEGRADATIONS)}"
        )
    sev = float(np.clip(severity, 0.0, 1.0))
    img_out, depth_out, intr_out, meta = DEGRADATIONS[name](
        image, depth, intrinsics, sev, int(frame_index)
    )
    return DegradedInput(
        image=img_out,
        depth=depth_out,
        intrinsics=intr_out,
        name=name,
        severity=sev,
        meta=meta,
        stale=bool(name == "frame_delay" and sev > 0 and (frame_index % 2 == 1)),
        dt_scale=float(meta.get("dt_scale", 1.0)),
    )
