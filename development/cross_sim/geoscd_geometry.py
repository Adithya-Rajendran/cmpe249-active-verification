"""Two-RGB GeoSCD-derived geometry registration; development only.

The geometry follows GeoSCD dd31369654e96d6843cc4bbcebce854a8bc2159b.
The comparator keeps floating projections instead of the upstream rounded map.
It uses no simulator data. Predicted visibility is not a support certificate.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import cv2
import numpy as np

from development.cross_sim.dense_alignment import warp_query

GEOSCD_REVISION = 'dd31369654e96d6843cc4bbcebce854a8bc2159b'
VGGT_REVISION = '860abec7937da0a4c03c41d3c269c366e82abdf9'
GRID = 512
CHECKPOINT_HASHES = {
    '.safetensors': 'f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e',
    '.pt': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',
}


def _torch_style_median(values):
    values = np.asarray(values).ravel()
    return float(np.partition(values, (len(values) - 1) // 2)[(len(values) - 1) // 2])


def project_geometry(ref_depth, query_depth, ref_k, query_k, ref_to_query):
    """Project reference depth into query pixels and estimate coarse visibility.

    Inputs use pixel-center intrinsics and the camera-from-camera 4x4 transform.
    Occlusion uses GeoSCD's median projected depth / MAD threshold and 3px
    dilation. Positive depths, finite values and full query FOV are required.
    """
    rd = np.asarray(ref_depth, dtype=np.float64)
    qd = np.asarray(query_depth, dtype=np.float64)
    rk, qk, transform = (np.asarray(a, dtype=np.float64) for a in (ref_k, query_k, ref_to_query))
    if rd.ndim != 2 or qd.shape != rd.shape or min(rd.shape) < 2:
        raise ValueError('depth maps must be equally sized 2D arrays')
    if rk.shape != (3, 3) or qk.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError('invalid camera matrix shape')
    if not all(np.isfinite(a).all() for a in (rk, qk, transform)):
        raise ValueError('camera matrices must be finite')
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError('transform must be affine')
    try:
        inv_k = np.linalg.inv(rk)
    except np.linalg.LinAlgError as exc:
        raise ValueError('reference intrinsics are singular') from exc
    h, w = rd.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    grid = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    point = (grid @ inv_k.T) * rd[..., None]
    projected = point @ transform[:3, :3].T + transform[:3, 3]
    z = projected[..., 2]
    uvw = projected @ qk.T
    with np.errstate(divide='ignore', invalid='ignore'):
        coords = uvw[..., :2] / uvw[..., 2:3]
    finite = np.isfinite(coords).all(axis=2) & np.isfinite(z) & np.isfinite(rd)
    valid = finite & (rd > 0) & (z > 0)
    valid &= (coords[..., 0] >= 0) & (coords[..., 0] <= w - 1)
    valid &= (coords[..., 1] >= 0) & (coords[..., 1] <= h - 1)
    # Upstream rounds before its depth consistency sample. Invalid query depth
    # must never be interpreted as co-visible.
    sx = np.where(valid, np.rint(coords[..., 0]), 0).astype(np.int64)
    sy = np.where(valid, np.rint(coords[..., 1]), 0).astype(np.int64)
    sampled = qd[sy, sx]
    valid &= np.isfinite(sampled) & (sampled > 0)
    difference = z - sampled
    if np.any(valid):
        values = difference[valid]
        median = _torch_style_median(values)
        mad = _torch_style_median(np.abs(values - median)) + 1e-6
        threshold = max(0.03 * _torch_style_median(z[valid]), median + 2.5 * mad)
        occluded = valid & (difference > threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        occluded = cv2.dilate(occluded.astype(np.uint8), kernel).astype(bool)
    else:
        threshold = None
        occluded = np.zeros((h, w), dtype=bool)
    usable = valid & ~occluded
    return coords, usable, {'coarse_usable_fraction': float(usable.mean()), 'depth_threshold': threshold}


def upsample_displacement(coords, usable, ref_shape, query_shape):
    """Map coarse predictions to native pixel centers with conservative support.

    Interpolate displacement, not absolute coordinates, to preserve identity
    under half-pixel resizing. Refuse any output whose positive-weight coarse neighbors are
    unavailable, invalid or outside the grid. No edge extrapolation is used.
    """
    coords = np.asarray(coords, dtype=np.float64)
    usable = np.asarray(usable)
    if coords.ndim != 3 or coords.shape[2] != 2 or usable.dtype != np.bool_ or usable.shape != coords.shape[:2]:
        raise ValueError('invalid coarse maps')
    ch, cw = usable.shape
    rh, rw = ref_shape
    qh, qw = query_shape
    if min(ch, cw, rh, rw, qh, qw) < 2:
        raise ValueError('image and grid dimensions must exceed one')
    y, x = np.indices((rh, rw), dtype=np.float64)
    cx, cy = (x + .5) * cw / rw - .5, (y + .5) * ch / rh - .5
    inside = (cx >= 0) & (cx <= cw - 1) & (cy >= 0) & (cy <= ch - 1)
    cx = np.clip(cx, 0, cw - 1)
    cy = np.clip(cy, 0, ch - 1)
    x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
    x1, y1 = np.minimum(x0 + 1, cw - 1), np.minimum(y0 + 1, ch - 1)
    ax, ay = (cx - x0)[..., None], (cy - y0)[..., None]
    finite = np.isfinite(coords).all(axis=2)
    point_ok = usable & finite
    weights = ((1-ax[...,0])*(1-ay[...,0]), ax[...,0]*(1-ay[...,0]),
               (1-ax[...,0])*ay[...,0], ax[...,0]*ay[...,0])
    good = inside.copy()
    for weight, ix, iy in zip(weights, (x0, x1, x0, x1), (y0, y0, y1, y1)):
        good &= (weight == 0) | point_ok[iy, ix]
    gy, gx = np.indices((ch, cw), dtype=np.float64)
    displacement = np.where(finite[..., None], coords - np.stack((gx, gy), axis=-1), 0)
    d = ((1-ay)*((1-ax)*displacement[y0,x0]+ax*displacement[y0,x1])
         + ay*((1-ax)*displacement[y1,x0]+ax*displacement[y1,x1]))
    mapped = np.stack(((cx + d[..., 0] + .5)*qw/cw - .5,
                       (cy + d[..., 1] + .5)*qh/ch - .5), axis=-1)
    good &= np.isfinite(mapped).all(axis=2) & (mapped[...,0] >= 0) & (mapped[...,0] <= qw-1)
    good &= (mapped[...,1] >= 0) & (mapped[...,1] <= qh-1)
    return mapped, good


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class GeometryProvider:
    """Local pinned VGGT geometry, no network access or implicit weight fetch."""

    def __init__(self, config):
        source_dir = config['source_dir']
        checkpoint = config['checkpoint']
        checkpoint_sha256 = config['checkpoint_sha256']
        device = config.get('device', 'cuda')
        precision = config.get('precision', 'bfloat16')
        import torch
        import subprocess

        source = Path(source_dir).resolve()
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != GEOSCD_REVISION:
            raise ValueError('GeoSCD revision mismatch')
        if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
            raise ValueError('GeoSCD checkout is not clean')
        expected = CHECKPOINT_HASHES.get(Path(checkpoint).suffix)
        if expected is None or checkpoint_sha256 != expected or _hash_file(checkpoint) != expected:
            raise ValueError('VGGT checkpoint does not match the pinned model revision')
        if (device != 'cuda' or precision != 'bfloat16' or not torch.cuda.is_available()
                or not torch.cuda.is_bf16_supported()):
            raise ValueError('this provider requires CUDA and bfloat16')
        sys.path.insert(0, str(source / 'src'))
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.torch, self.pose_decode = torch, pose_encoding_to_extri_intri
        self.device, self.precision = device, precision
        self.model = VGGT().to(device).eval()
        if str(checkpoint).endswith('.safetensors'):
            from safetensors.torch import load_file
            weights = load_file(str(checkpoint), device='cpu')
        else:
            weights = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.checkpoint_sha256 = checkpoint_sha256
        self._artifact = None

    def provenance(self):
        return {'geometry_revision': GEOSCD_REVISION, 'model_revision': VGGT_REVISION,
                'checkpoint_sha256': self.checkpoint_sha256, 'device': self.device,
                'precision': self.precision, 'torch_version': self.torch.__version__,
                'grid': GRID, 'preprocess': 'stretch1024_bicubic_to518_bilinear',
                'projection': 'float_ref_to_query', 'occlusion_dilation_px': 3,
                'upsampling': 'displacement_pixel_centers_positive_weight_neighbors_valid'}

    def geometry_artifact(self):
        if self._artifact is None:
            raise RuntimeError('no geometry artifact is available')
        return {key: value.copy() if isinstance(value, np.ndarray) else list(value)
                for key, value in self._artifact.items()}

    def _predict(self, ref, query):
        from PIL import Image
        torch = self.torch
        images = []
        for bgr in (ref, query):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            scaled = Image.fromarray(rgb).resize((1024, 1024), Image.Resampling.BICUBIC)
            images.append(np.asarray(scaled, dtype=np.float32).transpose(2, 0, 1) / 255.)
        batch = torch.from_numpy(np.stack(images)).to(self.device)
        batch = torch.nn.functional.interpolate(batch, (518, 518), mode='bilinear', align_corners=False)[None]
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                tokens, index = self.model.aggregator(batch)
            pose = self.model.camera_head(tokens)[-1]
            extrinsic, intrinsic = self.pose_decode(pose, batch.shape[-2:])
            depth, _ = self.model.depth_head(tokens, batch, index)
            depth = depth.permute(0, 1, 4, 2, 3).reshape(2, 1, 518, 518)
            depth = torch.nn.functional.interpolate(depth.float(), (GRID, GRID), mode='bilinear', align_corners=False)
        ex = extrinsic[0].float().cpu().numpy()
        intr = intrinsic[0].float().cpu().numpy() * np.array([[GRID/518, GRID/518, GRID/518], [GRID/518, GRID/518, GRID/518], [1,1,1]])
        mats = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
        mats[:, :3, :] = ex
        return depth[:, 0].cpu().numpy(), intr, mats[1] @ np.linalg.inv(mats[0])

    def register(self, reference_bgr, query_bgr, box):
        self._artifact = None
        ref, query = np.asarray(reference_bgr), np.asarray(query_bgr)
        if any(a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8 for a in (ref, query)):
            raise ValueError('both images must be uint8 BGR')
        x0, y0, x1, y1 = map(int, box)
        h, w = ref.shape[:2]
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('box must be inside reference image')
        depth, intr, transform = self._predict(ref, query)
        coarse, usable, diagnostics = project_geometry(depth[0], depth[1], intr[0], intr[1], transform)
        mapping, good = upsample_displacement(coarse, usable, ref.shape[:2], query.shape[:2])
        warped, valid = warp_query(query, mapping, good, np.zeros(good.shape, bool), np.eye(3))
        mx = np.where(valid, mapping[..., 0], -1).astype(np.float32)
        my = np.where(valid, mapping[..., 1], -1).astype(np.float32)
        nearest = cv2.remap(query, mx, my, interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        nearest[~valid] = 0
        bx0, by0 = max(0, x0 - 1), max(0, y0 - 1)
        bx1, by1 = min(w, x1 + 1), min(h, y1 + 1)
        self._artifact = {'bounds_xyxy': [bx0, by0, bx1, by1],
                          'query_xy': mapping[by0:by1, bx0:bx1].copy(),
                          'valid': valid[by0:by1, bx0:bx1].copy()}
        diagnostics.update(warped_nearest_bgr=nearest, registration_ok=bool(valid.any()),
                           reason='ok' if valid.any() else 'no_valid_geometry',
                           box_usable_fraction=float(valid[y0:y1, x0:x1].mean()),
                           native_usable_fraction=float(valid.mean()), precision=self.precision,
                           geometry_revision=GEOSCD_REVISION, model_revision=VGGT_REVISION,
                           checkpoint_sha256=self.checkpoint_sha256, torch_version=self.torch.__version__)
        return warped, valid, diagnostics
