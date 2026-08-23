"""Scene definitions used across all experiments.

Scenes are deterministic functions of ``(name, frame_index)``. That property is
what makes the degradation study valid: the *same* geometry can be replayed
under every corruption, so a change in perception error is attributable to the
corruption and not to scene variation.

The camera sits at the world origin looking down +Z and never moves; objects
move in the world. Camera-motion effects (viewpoint change) are applied as an
explicit pose perturbation instead, so they are measurable in the same way.

The occluder is deliberately painted a colour that is NOT in the detector's
appearance palette. It therefore occludes without being detectable, which is
the realistic case: perception is degraded by scene content the model was never
trained to represent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..geometry.camera import SE3, CameraIntrinsics
from ..perception.detector import ObjectAppearance
from .synthetic import Box3D

__all__ = [
    "OBJECT_PALETTE",
    "OCCLUDER_COLOR",
    "appearances",
    "default_intrinsics",
    "build_scene",
    "SCENES",
    "scene_names",
]

# Detectable object classes and their appearance priors (BGR).
OBJECT_PALETTE: dict[str, tuple[int, int, int]] = {
    "vehicle": (60, 90, 220),
    "pedestrian": (70, 200, 90),
    "cyclist": (220, 170, 50),
}

# Intentionally outside the palette: an undetectable occluding structure.
OCCLUDER_COLOR: tuple[int, int, int] = (120, 118, 115)


def appearances() -> list[ObjectAppearance]:
    return [ObjectAppearance(label=k, color_bgr=v) for k, v in OBJECT_PALETTE.items()]


def default_intrinsics(width: int = 640, height: int = 480) -> CameraIntrinsics:
    """A plausible forward-facing camera: ~60 deg horizontal FoV."""
    fx = fy = 0.9 * width / (2.0 * np.tan(np.deg2rad(60.0) / 2.0)) * 1.0
    return CameraIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=width / 2.0 - 0.5,
        cy=height / 2.0 - 0.5,
        width=width,
        height=height,
    )


def _approach(t: int, n: int) -> list[Box3D]:
    """A vehicle approaching head-on from 38 m to 6 m.

    Tests scale-dependent reliability: the same object becomes easier as it
    grows, so any monitor that ignores object size will be miscalibrated here.
    """
    frac = t / max(n - 1, 1)
    z = 38.0 - 32.0 * frac
    return [
        Box3D(
            obj_id=1,
            label="vehicle",
            center=np.array([0.4, 0.0, z]),
            size=np.array([1.9, 1.6, 4.3]),
            yaw=np.deg2rad(4.0),
            color=OBJECT_PALETTE["vehicle"],
        )
    ]


def _crossing(t: int, n: int) -> list[Box3D]:
    """A pedestrian crossing laterally at constant depth.

    Constant scale, constant appearance: isolates tracking/temporal signals
    from size effects.
    """
    frac = t / max(n - 1, 1)
    x = -6.0 + 12.0 * frac
    return [
        Box3D(
            obj_id=2,
            label="pedestrian",
            center=np.array([x, 0.35, 14.0]),
            size=np.array([0.7, 1.75, 0.5]),
            yaw=np.deg2rad(90.0),
            color=OBJECT_PALETTE["pedestrian"],
        )
    ]


def _occlusion(t: int, n: int) -> list[Box3D]:
    """A cyclist passing behind a nearer, undetectable pillar.

    This is the ground-truth source for occlusion-driven degradation: the exact
    visible_ratio is known per frame from the renderer.
    """
    frac = t / max(n - 1, 1)
    x = -5.0 + 10.0 * frac
    return [
        Box3D(
            obj_id=3,
            label="cyclist",
            center=np.array([x, 0.2, 16.0]),
            size=np.array([0.6, 1.7, 1.7]),
            yaw=np.deg2rad(90.0),
            color=OBJECT_PALETTE["cyclist"],
        ),
        Box3D(
            obj_id=90,
            label="occluder",
            center=np.array([0.0, 0.0, 11.0]),
            size=np.array([1.5, 4.5, 1.5]),
            yaw=0.0,
            color=OCCLUDER_COLOR,
        ),
    ]


def _multi(t: int, n: int) -> list[Box3D]:
    """Three classes at different depths plus an occluder: the general case."""
    frac = t / max(n - 1, 1)
    return [
        Box3D(
            obj_id=1,
            label="vehicle",
            center=np.array([-2.6, 0.0, 30.0 - 18.0 * frac]),
            size=np.array([1.9, 1.6, 4.3]),
            yaw=np.deg2rad(-6.0),
            color=OBJECT_PALETTE["vehicle"],
        ),
        Box3D(
            obj_id=2,
            label="pedestrian",
            center=np.array([-4.5 + 9.0 * frac, 0.35, 13.0]),
            size=np.array([0.7, 1.75, 0.5]),
            yaw=np.deg2rad(90.0),
            color=OBJECT_PALETTE["pedestrian"],
        ),
        Box3D(
            obj_id=3,
            label="cyclist",
            center=np.array([3.4, 0.2, 22.0 - 6.0 * frac]),
            size=np.array([0.6, 1.7, 1.7]),
            yaw=np.deg2rad(12.0),
            color=OBJECT_PALETTE["cyclist"],
        ),
        Box3D(
            obj_id=90,
            label="occluder",
            center=np.array([1.1, 0.0, 9.0]),
            size=np.array([1.2, 4.5, 1.2]),
            yaw=0.0,
            color=OCCLUDER_COLOR,
        ),
    ]


SCENES: dict[str, Callable[[int, int], list[Box3D]]] = {
    "approach": _approach,
    "crossing": _crossing,
    "occlusion": _occlusion,
    "multi": _multi,
}


def scene_names() -> list[str]:
    return list(SCENES)


def build_scene(name: str, frame_index: int, num_frames: int) -> list[Box3D]:
    if name not in SCENES:
        raise KeyError(f"unknown scene {name!r}; available: {sorted(SCENES)}")
    return SCENES[name](frame_index, num_frames)


def camera_pose(
    frame_index: int, num_frames: int, *, yaw_amplitude_deg: float = 0.0
) -> SE3:
    """Camera pose for a frame. Static by default.

    ``yaw_amplitude_deg`` drives the viewpoint-change degradation.
    """
    if abs(yaw_amplitude_deg) < 1e-9:
        return SE3()
    frac = frame_index / max(num_frames - 1, 1)
    yaw = np.deg2rad(yaw_amplitude_deg) * np.sin(2.0 * np.pi * frac)
    return SE3.from_yaw(float(yaw))
