"""Pinhole camera model, rigid transforms, projection and PnP utilities.

All math here is deliberately explicit rather than hidden behind OpenCV where
the intent matters for reasoning about reliability. OpenCV is used for the
parts where it is the reference implementation (distortion, PnP).

Conventions
-----------
* Camera frame: +X right, +Y down, +Z forward (OpenCV convention).
* A pose ``T_cw`` maps world points into the camera frame: ``X_cam = R @ X_world + t``.
* Pixels are ``(u, v)`` with the origin at the top-left corner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = [
    "CameraIntrinsics",
    "SE3",
    "project_points",
    "unproject_pixels",
    "reprojection_error",
    "solve_pnp",
]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics with a Brown-Conrady distortion model.

    ``dist`` is ``(k1, k2, p1, p2, k3)`` to match the OpenCV ordering so the
    same vector can be handed to ``cv2.projectPoints`` / ``cv2.undistort``.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    @property
    def dist_vec(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=np.float64).reshape(-1, 1)

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(np.abs(self.dist_vec) > 1e-12))

    def perturbed(
        self,
        *,
        d_fx: float = 0.0,
        d_fy: float = 0.0,
        d_cx: float = 0.0,
        d_cy: float = 0.0,
        dist: tuple[float, float, float, float, float] | None = None,
    ) -> CameraIntrinsics:
        """Return a copy with additive perturbations.

        This is the entry point for the *calibration error* degradation: the
        pipeline is given wrong intrinsics while ground truth is rendered with
        the true ones, so calibration residuals become measurable.
        """
        return CameraIntrinsics(
            fx=self.fx + d_fx,
            fy=self.fy + d_fy,
            cx=self.cx + d_cx,
            cy=self.cy + d_cy,
            width=self.width,
            height=self.height,
            dist=self.dist if dist is None else dist,
        )

    def contains(self, uv: np.ndarray) -> np.ndarray:
        """Boolean mask of which ``(N, 2)`` pixels lie inside the image."""
        uv = np.atleast_2d(uv)
        return (
            (uv[:, 0] >= 0)
            & (uv[:, 0] <= self.width - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= self.height - 1)
        )


@dataclass(frozen=True)
class SE3:
    """Rigid transform. ``R`` is ``(3, 3)``, ``t`` is ``(3,)``."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "R", np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        )
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64).reshape(3))

    @staticmethod
    def from_rvec(rvec: np.ndarray, tvec: np.ndarray) -> SE3:
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        return SE3(R=R, t=np.asarray(tvec, dtype=np.float64).reshape(3))

    @staticmethod
    def from_yaw(yaw: float, t: np.ndarray | None = None) -> SE3:
        """Rotation about the camera-frame Y axis (yaw for a ground vehicle)."""
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return SE3(R=R, t=np.zeros(3) if t is None else t)

    @property
    def rvec(self) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(self.R)
        return rvec.reshape(3)

    def inverse(self) -> SE3:
        Rt = self.R.T
        return SE3(R=Rt, t=-Rt @ self.t)

    def __matmul__(self, other: SE3) -> SE3:
        return SE3(R=self.R @ other.R, t=self.R @ other.t + self.t)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform ``(N, 3)`` points."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return pts @ self.R.T + self.t


def project_points(
    points_cam: np.ndarray,
    intr: CameraIntrinsics,
    *,
    apply_distortion: bool = True,
) -> np.ndarray:
    """Project ``(N, 3)`` camera-frame points to ``(N, 2)`` pixels.

    Points at or behind the image plane are returned as NaN rather than being
    silently wrapped around, so callers must handle them explicitly.
    """
    pts = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    valid = pts[:, 2] > 1e-6
    if not np.any(valid):
        return out

    if apply_distortion and intr.has_distortion:
        uv, _ = cv2.projectPoints(
            pts[valid].reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            intr.K,
            intr.dist_vec,
        )
        out[valid] = uv.reshape(-1, 2)
    else:
        z = pts[valid, 2]
        out[valid, 0] = intr.fx * pts[valid, 0] / z + intr.cx
        out[valid, 1] = intr.fy * pts[valid, 1] / z + intr.cy
    return out


def unproject_pixels(
    uv: np.ndarray,
    depth: np.ndarray,
    intr: CameraIntrinsics,
    *,
    undistort: bool = True,
) -> np.ndarray:
    """Lift ``(N, 2)`` pixels with ``(N,)`` metric depth into camera-frame 3D.

    ``depth`` is z along the optical axis (not ray length), matching the depth
    buffer produced by the renderer and by RGB-D sensors.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    if uv.shape[0] != z.shape[0]:
        raise ValueError(f"uv/depth length mismatch: {uv.shape[0]} vs {z.shape[0]}")

    if undistort and intr.has_distortion:
        uv = cv2.undistortPoints(
            uv.reshape(-1, 1, 2), intr.K, intr.dist_vec, P=intr.K
        ).reshape(-1, 2)

    x = (uv[:, 0] - intr.cx) / intr.fx * z
    y = (uv[:, 1] - intr.cy) / intr.fy * z
    return np.stack([x, y, z], axis=1)


def reprojection_error(
    points_world: np.ndarray,
    uv_observed: np.ndarray,
    pose: SE3,
    intr: CameraIntrinsics,
) -> np.ndarray:
    """Per-point pixel reprojection error. NaN where projection is invalid."""
    uv_pred = project_points(pose.apply(points_world), intr)
    obs = np.atleast_2d(np.asarray(uv_observed, dtype=np.float64))
    return np.linalg.norm(uv_pred - obs, axis=1)


def solve_pnp(
    points_object: np.ndarray,
    uv_observed: np.ndarray,
    intr: CameraIntrinsics,
    *,
    use_ransac: bool = True,
    reproj_threshold: float = 3.0,
) -> tuple[SE3 | None, np.ndarray, float]:
    """Estimate object pose from 3D-2D correspondences.

    Returns ``(pose, inlier_mask, rmse)``. ``pose`` is None when PnP fails or
    there are too few correspondences; callers treat that as a hard
    geometric-consistency failure rather than substituting a guess.
    """
    obj = np.asarray(points_object, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(uv_observed, dtype=np.float64).reshape(-1, 2)
    n = obj.shape[0]
    if n < 4 or img.shape[0] != n:
        return None, np.zeros(n, dtype=bool), float("nan")

    finite = np.isfinite(img).all(axis=1)
    if int(finite.sum()) < 4:
        return None, np.zeros(n, dtype=bool), float("nan")
    obj_f, img_f = obj[finite], img[finite]

    inliers = None
    if use_ransac and obj_f.shape[0] >= 6:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_f.reshape(-1, 1, 3),
            img_f.reshape(-1, 1, 2),
            intr.K,
            intr.dist_vec,
            reprojectionError=float(reproj_threshold),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            obj_f.reshape(-1, 1, 3),
            img_f.reshape(-1, 1, 2),
            intr.K,
            intr.dist_vec,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None, np.zeros(n, dtype=bool), float("nan")

    pose = SE3.from_rvec(rvec, tvec)

    mask_f = np.ones(obj_f.shape[0], dtype=bool)
    if inliers is not None and len(inliers) > 0:
        mask_f = np.zeros(obj_f.shape[0], dtype=bool)
        mask_f[np.asarray(inliers).reshape(-1)] = True

    err = np.linalg.norm(project_points(pose.apply(obj_f), intr) - img_f, axis=1)
    used = err[mask_f]
    used = used[np.isfinite(used)]
    rmse = float(np.sqrt(np.mean(used**2))) if used.size else float("nan")

    mask = np.zeros(n, dtype=bool)
    mask[np.where(finite)[0][mask_f]] = True
    return pose, mask, rmse
