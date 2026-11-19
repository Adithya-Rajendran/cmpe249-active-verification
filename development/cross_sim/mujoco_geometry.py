"""Geometry helpers for the development MuJoCo scene adapter.

These functions do not render or attest to backend qualification.
"""
from __future__ import annotations

import math
import numpy as np


def local_ycb(vertices_mm, rest_matrix, normals=None):
    """Apply the Isaac arrangement's resting frame and millimetre conversion."""
    v = np.asarray(vertices_mm, dtype=float) / 1000.0
    r = np.asarray(rest_matrix, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or r.shape != (3, 3):
        raise ValueError('expected Nx3 vertices and a 3x3 rest matrix')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('rest matrix must be a proper rotation')
    v = v @ r.T
    lo, hi = v.min(axis=0), v.max(axis=0)
    v -= [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
    n = None if normals is None else np.asarray(normals, dtype=float) @ r.T
    return v, n


def camera_axes(forward, up):
    """Return the MuJoCo camera's right, up and backward world axes."""
    f = np.asarray(forward, dtype=float)
    u = np.asarray(up, dtype=float)
    if np.linalg.norm(f) < 1e-12 or np.linalg.norm(u) < 1e-12:
        raise ValueError('zero camera axis')
    f /= np.linalg.norm(f)
    r = np.cross(f, u)
    if np.linalg.norm(r) < 1e-9:
        raise ValueError('parallel camera axes')
    r /= np.linalg.norm(r)
    return r, np.cross(r, f), -f


def fovy_degrees(f_px, height):
    if f_px <= 0 or height <= 0:
        raise ValueError('focal length and height must be positive')
    return math.degrees(2 * math.atan(height / (2 * f_px)))


def world_yaw(vertices, base, yaw_deg):
    a = math.radians(yaw_deg)
    r = np.array([[math.cos(a), -math.sin(a), 0],
                  [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return np.asarray(vertices, float) @ r.T + np.asarray(base, float)
