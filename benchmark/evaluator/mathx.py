"""Numpy re-implementations of the quaternion helpers used by mjlab.

All quaternions are ``w, x, y, z`` (matching MuJoCo and
``mjlab.utils.lab_api.math``), float64, shape ``(..., 4)``.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_inv(q: np.ndarray) -> np.ndarray:
    return quat_conj(normalize(q))


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), EPS)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by unit quaternion ``q`` (broadcasting over leading dims)."""
    qv = q[..., 1:]
    w = q[..., :1]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(q), v)


def quat_error_magnitude(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between two rotations, in radians, in ``[0, pi]``."""
    rel = quat_mul(quat_conj(normalize(a)), normalize(b))
    w = np.clip(np.abs(rel[..., 0]), -1.0, 1.0)
    v = np.linalg.norm(rel[..., 1:], axis=-1)
    return 2.0 * np.arctan2(v, w)


def yaw_quat(q: np.ndarray, w_last: bool = False) -> np.ndarray:
    """Keep only the yaw component of ``q``."""
    q = normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    out = np.zeros_like(q)
    out[..., 0] = np.cos(half)
    out[..., 3] = np.sin(half)
    if w_last:
        out = out[..., [3, 0, 1, 2]]
    return out


def projected_gravity(q: np.ndarray) -> np.ndarray:
    """Gravity unit vector expressed in the body frame of ``q``."""
    down = np.zeros(q.shape[:-1] + (3,), dtype=np.float64)
    down[..., 2] = -1.0
    return quat_apply_inverse(q, down)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix ``(..., 3, 3)`` with columns equal to the rotated axes."""
    q = normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    m[..., 0, 1] = 2.0 * (xy - wz)
    m[..., 0, 2] = 2.0 * (xz + wy)
    m[..., 1, 0] = 2.0 * (xy + wz)
    m[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    m[..., 1, 2] = 2.0 * (yz - wx)
    m[..., 2, 0] = 2.0 * (xz - wy)
    m[..., 2, 1] = 2.0 * (yz + wx)
    m[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return m


def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray, t02: np.ndarray, q02: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Express frame 02 in frame 01."""
    pos_b = quat_apply_inverse(q01, t02 - t01)
    quat_b = quat_mul(quat_conj(normalize(q01)), q02)
    return pos_b, quat_b


def axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of ``q``."""
    q = normalize(q)
    w = np.clip(q[..., 0:1], -1.0, 1.0)
    v = q[..., 1:]
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, w)
    return np.where(norm > EPS, v / np.maximum(norm, EPS) * angle, 0.0)


def slerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between single quaternions ``a`` and ``b``."""
    a = normalize(a)
    b = normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return normalize(a + blend * (b - a))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return normalize(
        (np.sin((1.0 - blend) * theta) * a + np.sin(blend * theta) * b) / sin_theta
    )


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic XYZ (roll, pitch, yaw) -> quaternion, mjlab convention."""
    hz, hy, hx = 0.5 * yaw, 0.5 * pitch, 0.5 * roll
    cr, sr = np.cos(hx), np.sin(hx)
    cp, sp = np.cos(hy), np.sin(hy)
    cy, sy = np.cos(hz), np.sin(hz)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
