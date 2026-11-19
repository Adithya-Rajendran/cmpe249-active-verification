"""Resample a query image from a declared reference-to-query pixel map.

This is a mapping boundary, not a GeoSCD inference adapter. Map rows and
columns must already correspond one-for-one to reference pixels. Map values
are (x, y) query coordinates in the caller's explicitly declared query frame.
The transform maps original query pixel centers to that frame. No map resize,
model inference, or geometry validity is inferred here.
"""
from __future__ import annotations

import cv2
import numpy as np


def warp_query(query_rgb: np.ndarray, ref_to_query: np.ndarray, valid: np.ndarray,
               occluded: np.ndarray, query_to_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return query RGB and conservative validity at each reference pixel.

    ``ref_to_query[y, x]`` is the query location of reference pixel (x, y).
    Coordinates use pixel-center convention: integer (0, 0) is the first center.
    ``query_to_map`` is an invertible affine 3x3 matrix mapping original query
    pixel-center coordinates into the coordinate frame of map values. Pass the
    identity for original query pixels. Boolean ``valid`` and ``occluded`` are
    evaluated in reference space. Invalid results are black and false.
    Bilinear samples are accepted only within the query's pixel-center bounds.
    """
    query = np.asarray(query_rgb)
    coords = np.asarray(ref_to_query)
    valid = np.asarray(valid)
    occluded = np.asarray(occluded)
    transform = np.asarray(query_to_map)
    if query.ndim != 3 or query.shape[2] != 3 or query.dtype != np.uint8 or not query.size:
        raise ValueError('query_rgb must be a nonempty uint8 HxWx3 image')
    if coords.ndim != 3 or coords.shape[2] != 2 or coords.dtype.kind not in 'iuf':
        raise ValueError('ref_to_query must be an HxWx2 real coordinate map')
    if not coords.shape[0] or not coords.shape[1]:
        raise ValueError('reference map must not be empty')
    if valid.shape != coords.shape[:2] or valid.dtype != np.bool_:
        raise ValueError('valid must be a boolean mask in reference space')
    if occluded.shape != coords.shape[:2] or occluded.dtype != np.bool_:
        raise ValueError('occluded must be a boolean mask in reference space')
    if (transform.shape != (3, 3) or transform.dtype.kind not in 'iuf' or not np.isfinite(transform).all()
            or not np.allclose(transform[2], [0, 0, 1], rtol=0, atol=1e-12)):
        raise ValueError('query_to_map must be a finite affine 3x3 matrix')
    try:
        inverse = np.linalg.inv(transform.astype(np.float64, copy=False))
    except np.linalg.LinAlgError as exc:
        raise ValueError('query_to_map must be invertible') from exc
    coords = coords.astype(np.float64, copy=False)
    finite = np.isfinite(coords).all(axis=2)
    safe = np.where(finite[..., None], coords, 0.0)
    x = inverse[0, 0] * safe[..., 0] + inverse[0, 1] * safe[..., 1] + inverse[0, 2]
    y = inverse[1, 0] * safe[..., 0] + inverse[1, 1] * safe[..., 1] + inverse[1, 2]
    h, w = query.shape[:2]
    good = valid & ~occluded & finite & np.isfinite(x) & np.isfinite(y)
    good &= (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x = np.where(good, x, -1).astype(np.float32)
    y = np.where(good, y, -1).astype(np.float32)
    warped = cv2.remap(query, x, y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped[~good] = 0
    return warped, good
