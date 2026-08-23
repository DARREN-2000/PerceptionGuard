"""Synthetic RGB-D scene generator with exact ground truth.

Why a hand-written rasterizer instead of BlenderProc: this project's central
claim is that a reliability score tracks *actual* perception error. Validating
that requires per-frame ground truth for 3D centre, metric depth, object pose
and occlusion ratio, plus the ability to hold a scene fixed while varying one
degradation at a time. A small exact rasterizer gives all of that with no
external dependency and no download, which matters because the target sandbox
has no network access. BlenderProc would give photorealism, which this project
does not trade on.

Rendering is an exact pinhole projection with a true per-pixel z-buffer. Depth
is recovered analytically from each face's plane equation rather than by
barycentric interpolation, so the depth map is exact to float precision and can
serve as depth ground truth.

Lens distortion is deliberately *not* baked into the render. The renderer
produces an ideal pinhole image; distortion and calibration error are applied
downstream as controlled degradations, which keeps the ground truth clean.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry.camera import SE3, CameraIntrinsics, project_points

__all__ = [
    "Box3D",
    "InstanceGT",
    "Frame",
    "render_frame",
    "unit_box_corners",
    "BOX_FACES",
]

# Faces of a box as indices into ``unit_box_corners`` output, wound so that the
# 4 vertices of each face are in polygon order (required by fillConvexPoly).
BOX_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),  # -Z
    (4, 5, 6, 7),  # +Z
    (0, 1, 5, 4),  # -Y
    (3, 2, 6, 7),  # +Y
    (0, 3, 7, 4),  # -X
    (1, 2, 6, 5),  # +X
)

# Fixed light direction in the camera frame for simple Lambertian shading.
_LIGHT_DIR = np.array([-0.35, -0.75, 0.55])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)

_EPS_Z = 0.05  # metres; nearer than this is treated as behind the image plane


def unit_box_corners(size: Sequence[float]) -> np.ndarray:
    """Return ``(8, 3)`` box corners in the object frame, centred at origin.

    ``size`` is ``(sx, sy, sz)`` full extents. Corner order matches BOX_FACES.
    """
    sx, sy, sz = (float(v) / 2.0 for v in size)
    return np.array(
        [
            [-sx, -sy, -sz],
            [+sx, -sy, -sz],
            [+sx, +sy, -sz],
            [-sx, +sy, -sz],
            [-sx, -sy, +sz],
            [+sx, -sy, +sz],
            [+sx, +sy, +sz],
            [-sx, +sy, +sz],
        ],
        dtype=np.float64,
    )


@dataclass
class Box3D:
    """A cuboid object in the world frame."""

    obj_id: int
    label: str
    center: np.ndarray  # (3,) world
    size: np.ndarray  # (3,) full extents
    yaw: float = 0.0  # rotation about world Y
    color: tuple[int, int, int] = (60, 160, 240)  # BGR

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.size = np.asarray(self.size, dtype=np.float64).reshape(3)

    @property
    def T_wo(self) -> SE3:
        """Object-to-world transform."""
        return SE3.from_yaw(self.yaw, self.center)

    def corners_object(self) -> np.ndarray:
        return unit_box_corners(self.size)

    def corners_world(self) -> np.ndarray:
        return self.T_wo.apply(self.corners_object())


@dataclass
class InstanceGT:
    """Ground truth for one visible object instance in one frame."""

    obj_id: int
    label: str
    bbox: tuple[float, float, float, float]  # visible bbox, xyxy
    bbox_amodal: tuple[float, float, float, float]  # bbox ignoring occlusion
    center_cam: np.ndarray  # (3,) true 3D centre in camera frame
    depth: float  # true z of the centre
    visible_ratio: float  # visible px / unoccluded px, in [0, 1]
    num_pixels: int
    corners_px: np.ndarray  # (8, 2) projected corners (ideal pinhole)
    corners_object: np.ndarray  # (8, 3) object-frame corners, for PnP
    T_co: SE3  # true object pose in the camera frame


@dataclass
class Frame:
    """One rendered frame plus its ground truth."""

    index: int
    image: np.ndarray  # (H, W, 3) uint8 BGR
    depth: np.ndarray  # (H, W) float32, NaN where background
    instance_map: np.ndarray  # (H, W) int32, -1 where background
    instances: list[InstanceGT] = field(default_factory=list)
    intrinsics: CameraIntrinsics | None = None
    T_cw: SE3 | None = None
    timestamp: float = 0.0


def _rasterize(
    boxes: Sequence[Box3D],
    intr: CameraIntrinsics,
    T_cw: SE3,
    *,
    background: tuple[int, int, int],
    ambient: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Painter-free z-buffered rasterization.

    Returns ``(color, zbuf, idbuf)``. Correctness does not depend on draw order
    because every pixel keeps the nearest surface.
    """
    H, W = intr.height, intr.width
    color = np.zeros((H, W, 3), dtype=np.uint8)
    color[:] = np.asarray(background, dtype=np.uint8)
    zbuf = np.full((H, W), np.inf, dtype=np.float64)
    idbuf = np.full((H, W), -1, dtype=np.int32)

    K_inv = intr.K_inv

    for box in boxes:
        corners_cam = T_cw.apply(box.corners_world())
        if np.all(corners_cam[:, 2] <= _EPS_Z):
            continue

        for face in BOX_FACES:
            fc = corners_cam[list(face)]
            # Skip faces that straddle the image plane; near-plane clipping is
            # out of scope and scenes are authored to keep objects in front.
            if np.any(fc[:, 2] <= _EPS_Z):
                continue

            uv = project_points(fc, intr, apply_distortion=False)
            if not np.all(np.isfinite(uv)):
                continue

            n = np.cross(fc[1] - fc[0], fc[2] - fc[0])
            n_norm = float(np.linalg.norm(n))
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            d = float(n @ fc[0])
            if abs(d) < 1e-12:
                continue

            x0 = int(np.floor(uv[:, 0].min()))
            x1 = int(np.ceil(uv[:, 0].max()))
            y0 = int(np.floor(uv[:, 1].min()))
            y1 = int(np.ceil(uv[:, 1].max()))
            x0c, y0c = max(x0, 0), max(y0, 0)
            x1c, y1c = min(x1, W - 1), min(y1, H - 1)
            if x1c < x0c or y1c < y0c:
                continue

            sub_w = x1c - x0c + 1
            sub_h = y1c - y0c + 1
            mask = np.zeros((sub_h, sub_w), dtype=np.uint8)
            poly = np.round(uv - np.array([x0c, y0c])).astype(np.int32)
            cv2.fillConvexPoly(mask, poly, 1, lineType=cv2.LINE_8)
            if not mask.any():
                continue

            vv, uu = np.nonzero(mask)
            us = uu + x0c
            vs = vv + y0c

            # Exact depth from the plane equation: for a ray r = K^-1 [u,v,1]
            # (whose z component is 1), the surface point is t*r with
            # t = d / (n . r), hence z = t.
            rays = np.stack(
                [us.astype(np.float64), vs.astype(np.float64), np.ones(us.size)], axis=1
            )
            rays = rays @ K_inv.T
            denom = rays @ n
            ok = np.abs(denom) > 1e-12
            z = np.full(us.size, np.inf)
            z[ok] = d / denom[ok]
            ok &= np.isfinite(z) & (z > _EPS_Z)

            # Depth over a planar convex polygon is a projective function of the
            # pixel, so its extrema occur at the vertices. Any rasterized pixel
            # whose computed depth falls outside the corner depth range is a
            # rounding artifact from fillConvexPoly straying off the true face
            # -- and for near-edge-on faces the plane solution diverges there,
            # which previously corrupted the depth map at silhouettes. Reject.
            z_lo = float(fc[:, 2].min()) - 1e-3
            z_hi = float(fc[:, 2].max()) + 1e-3
            ok &= (z >= z_lo) & (z <= z_hi)
            if not np.any(ok):
                continue

            us, vs, z = us[ok], vs[ok], z[ok]
            closer = z < zbuf[vs, us]
            if not np.any(closer):
                continue
            us, vs, z = us[closer], vs[closer], z[closer]

            shade = ambient + (1.0 - ambient) * abs(float(n @ _LIGHT_DIR))
            shaded = np.clip(np.asarray(box.color, dtype=np.float64) * shade, 0, 255)

            zbuf[vs, us] = z
            idbuf[vs, us] = box.obj_id
            color[vs, us] = shaded.astype(np.uint8)

    return color, zbuf, idbuf


def render_frame(
    boxes: Sequence[Box3D],
    intr: CameraIntrinsics,
    T_cw: SE3 | None = None,
    *,
    index: int = 0,
    timestamp: float = 0.0,
    background: tuple[int, int, int] = (38, 34, 30),
    ambient: float = 0.35,
    min_pixels: int = 12,
) -> Frame:
    """Render RGB + exact depth + instance map + ground truth for one frame.

    ``visible_ratio`` is computed by rasterizing each object a second time on
    its own; the ratio of occluded-scene pixels to solo pixels is the exact
    occlusion fraction, which later lets us check whether the reliability
    monitor reacts to occlusion specifically.
    """
    T_cw = SE3() if T_cw is None else T_cw
    color, zbuf, idbuf = _rasterize(
        boxes, intr, T_cw, background=background, ambient=ambient
    )

    depth = np.where(np.isfinite(zbuf), zbuf, np.nan).astype(np.float32)

    instances: list[InstanceGT] = []
    for box in boxes:
        vis = idbuf == box.obj_id
        num_px = int(vis.sum())
        if num_px < min_pixels:
            continue

        _, _, solo_id = _rasterize(
            [box], intr, T_cw, background=background, ambient=ambient
        )
        solo_px = int((solo_id == box.obj_id).sum())
        visible_ratio = float(num_px / solo_px) if solo_px > 0 else 0.0

        ys, xs = np.nonzero(vis)
        bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

        T_co = T_cw @ box.T_wo
        corners_px = project_points(
            T_co.apply(box.corners_object()), intr, apply_distortion=False
        )
        finite = np.isfinite(corners_px).all(axis=1)
        if np.any(finite):
            fp = corners_px[finite]
            bbox_amodal = (
                float(fp[:, 0].min()),
                float(fp[:, 1].min()),
                float(fp[:, 0].max()),
                float(fp[:, 1].max()),
            )
        else:
            bbox_amodal = bbox

        center_cam = T_cw.apply(box.center.reshape(1, 3))[0]
        instances.append(
            InstanceGT(
                obj_id=box.obj_id,
                label=box.label,
                bbox=bbox,
                bbox_amodal=bbox_amodal,
                center_cam=center_cam,
                depth=float(center_cam[2]),
                visible_ratio=min(visible_ratio, 1.0),
                num_pixels=num_px,
                corners_px=corners_px,
                corners_object=box.corners_object(),
                T_co=T_co,
            )
        )

    return Frame(
        index=index,
        image=color,
        depth=depth,
        instance_map=idbuf,
        instances=instances,
        intrinsics=intr,
        T_cw=T_cw,
        timestamp=timestamp,
    )
