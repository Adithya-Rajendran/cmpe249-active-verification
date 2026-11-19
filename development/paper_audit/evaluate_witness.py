"""Evaluator-only fixed-reference patch and RGB contrast measurements.

Inputs are private. Coordinates are 1200×675 primary pixels, with half-open boxes. This
module does not set thresholds or read scorer answers.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy()


def mask(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    return np.any(a[..., :3] != 0, axis=2) if a.ndim == 3 else a != 0



def primary_rgb(path):
    with Image.open(path) as im:
        if im.size != (2304,1296):
            raise ValueError('expected native RGB dimensions')
        return np.asarray(im.convert('RGB').resize((1200,675),getattr(Image, "Resampling", Image).LANCZOS)).copy()


def primary_mask(path):
    with Image.open(path) as im:
        if im.size != (2304,1296):
            raise ValueError('expected native mask dimensions')
        a = np.asarray(im)
    a = np.any(a[..., :3] != 0, axis=2) if a.ndim == 3 else a != 0
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((1200,675),getattr(Image, "Resampling", Image).NEAREST),dtype=bool)

def choose_patch(reference_mask, exposed_mask, reference_box, size=7, minimum=25):
    """Select a center by distance to the outside, then box-center distance, y, x."""
    from scipy.ndimage import distance_transform_edt
    ref = np.asarray(reference_mask, dtype=bool)
    exposed = np.asarray(exposed_mask, dtype=bool)
    if ref.ndim != 2 or ref.shape != exposed.shape or size != 7 or minimum != 25:
        raise ValueError('invalid masks or patch rule')
    x0, y0, x1, y1 = reference_box
    if not (0 <= x0 < x1 <= ref.shape[1] and 0 <= y0 < y1 <= ref.shape[0]):
        raise ValueError('invalid reference box')
    intersection = ref & exposed
    if not intersection.any():
        return None
    distance = distance_transform_edt(np.pad(intersection, 1, constant_values=False))[1:-1, 1:-1]
    yy, xx = np.nonzero(distance == distance.max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    i = min(range(len(xx)), key=lambda i: ((xx[i]-cx)**2 + (yy[i]-cy)**2, yy[i], xx[i]))
    x, y = int(xx[i]), int(yy[i])
    radius = size // 2
    if x < radius or y < radius or x + radius >= ref.shape[1] or y + radius >= ref.shape[0]:
        return None
    region = intersection[y-radius:y+radius+1, x-radius:x+radius+1]
    count = int(region.sum())
    if count < minimum:
        return None
    return {'xywh':[x-radius,y-radius,size,size], 'center_xy':[x,y],
            'intersection_pixels':count, 'inscribed_radius':float(distance[y,x])}


def contrast(a, b, patch, reference_mask, exposed_mask):
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError('RGB shapes differ')
    intersection = np.asarray(reference_mask, dtype=bool) & np.asarray(exposed_mask, dtype=bool)
    if intersection.shape != a.shape[:2]:
        raise ValueError('mask dimensions differ')
    x, y, w, h = patch['xywh']
    if min(x, y) < 0 or x + w > a.shape[1] or y + h > a.shape[0]:
        raise ValueError('patch outside image')
    support = intersection[y:y+h,x:x+w]
    if int(support.sum()) < 25:
        raise ValueError('insufficient patch support')
    delta = np.abs(a[y:y+h, x:x+w].astype(np.int16) - b[y:y+h, x:x+w].astype(np.int16))
    return float(delta[support].mean())


def witness_pass(change, retained_repeat, removed_repeat):
    values = [change, retained_repeat, removed_repeat]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError('invalid contrast')
    bound = max(retained_repeat, removed_repeat)
    return bool(change >= 8 and change - bound >= 4)


def exposure(visible_mask, alone_mask):
    if visible_mask.shape != alone_mask.shape:
        raise ValueError('mask shapes differ')
    n = int(alone_mask.sum())
    return {'visible_pixels': int(visible_mask.sum()), 'alone_pixels': n,
            'visible_fraction_of_alone': float(np.count_nonzero(visible_mask & alone_mask) / n) if n else None,
            'visible_outside_alone_pixels': int(np.count_nonzero(visible_mask & ~alone_mask))}
