"""Seeded procedural light-oak wood-grain texture for the simulated tabletop (numpy + the standard library only).

Why: the tabletop's finish is ONE fixed wood-grain texture (decision 2026-11-04, item 3): texture, physical scale,
orientation and material settings are fixed before any evidence result is seen, and a failure on grain is a valid dev
finding. Nothing here is randomised per arrangement or per render: one seed, one image.

The image covers the whole table top once (no tiling): image width = the table's x extent (1.20 m), image height = its y
extent (0.80 m), at 0.5 mm per pixel (2400 x 1600). The grain (the growth-ring stripes and the fibre streaks) runs along
the image width, i.e. along +x on the table (left-right as seen from the start stop P3). The UVs of the table-top mesh
(sim/tabletop.py) are u = (x - x_min) / 1.20, v = (y - y_min) / 0.80, so the texture is never stretched or repeated and the
renderer's clamp wrap mode (sim/isaac_rig.py _material) is harmless.

Model (all lengths in metres on the table): a glued panel of `n_boards` plain-sawn oak boards, each running the full
length along x. Board b (a band of y) was cut from a log whose axis lies near the board, at lateral offset y_c and depth
d(x) = d0 + slope * (x - x_mid) below the cut face, so its growth rings meet the face as stripes along x that close into
arches where d is small (the 'cathedral' figure):
  r          = sqrt((y - y_c)^2 + d(x)^2) + warp(x, y)
  ring phase = r / ring_spacing + phase_noise(x, y)
  latewood   = smoothstep(0.62, 0.84, frac(phase)) * (1 - smoothstep(0.93, 1.0, frac(phase)))   (dark band, sharp return)
  colour     = lerp(earlywood, latewood colour, latewood_strength * latewood) + board tone + fibre streaks + pores
where warp / phase_noise / fibre / pores are value noise (a seeded grid of Generator.random() values, smoothstep-
interpolated), each stretched along x, and y_c, d0, slope and the board tone are uniform draws per board. Random draws use
only numpy's Generator(PCG64(seed)).random(), so the stream does not depend on numpy's distribution algorithms (as
sim/arrange.py). The PNG is written by a minimal encoder (zlib level 9, no filter) so the file bytes depend only on the pixels
and the interpreter's zlib; the pixel sha256 is the primary pin.

    python -m sim.wood_texture --out PATH.png      # prints the pixel and file sha256
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np

PARAMS = {
    "name": "tabletop_light_oak_v1",
    "seed": 20261104,
    "size_m": [1.20, 0.80],            # (x, y) extent covered by the image = the table top: one tile, no repeat
    "m_per_px": 0.0005,                # 2400 x 1600 pixels
    "grain_axis": "x",                 # stripes and fibres run along the image width = table +x
    "n_boards": 4,                     # glued boards, each 0.20 m wide (along y), full length along x
    "log_offset_m": 0.05,              # y_c = board centre + U(-1, 1) * this
    "log_depth_m": [0.01, 0.09],       # d0 ~ U(lo, hi)
    "log_slope": 0.06,                 # slope ~ U(-1, 1) * this (depth change per metre along x)
    "board_offset_srgb": 6.0,          # per-board tone ~ U(-1, 1) * this
    "ring_spacing_m": 0.0042,
    "warp": [[0.30, 0.10, 0.012], [0.08, 0.03, 0.003]],      # [cell along x, cell along y, amplitude] (m), summed
    "phase_noise": [0.25, 0.06, 0.6],                         # [cell x, cell y, amplitude in ring periods]
    "earlywood_srgb": [206, 172, 128],
    "latewood_srgb": [168, 128, 86],
    "latewood_strength": 0.85,
    "fibre": [0.025, 0.0004, 7.0],                            # [cell x, cell y, amplitude in sRGB levels]
    "pores": [0.003, 0.0003, 0.80, 16.0],                     # [cell x, cell y, threshold, darkening in sRGB levels]
    "board_tone": [0.40, 0.40, 5.0],                          # [cell x, cell y, amplitude]: slow overall tone variation
}


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def value_noise(rng: np.random.Generator, xs: np.ndarray, ys: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """Smooth noise in [0, 1) on the grid xs (columns) x ys (rows), metres: a grid of rng.random() values every cell_x along
    x and cell_y along y, interpolated with smoothstep weights. Returns (len(ys), len(xs))."""
    gx = int(np.floor(xs.max() / cell_x)) + 2
    gy = int(np.floor(ys.max() / cell_y)) + 2
    g = rng.random((gy, gx))
    fx, fy = xs / cell_x, ys / cell_y
    i0, j0 = np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
    ty, tx = fy - i0, fx - j0
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    top = g[i0][:, j0] * (1 - tx) + g[i0][:, j0 + 1] * tx
    bot = g[i0 + 1][:, j0] * (1 - tx) + g[i0 + 1][:, j0 + 1] * tx
    return top * (1 - ty)[:, None] + bot * ty[:, None]


def generate(params: dict = PARAMS) -> np.ndarray:
    """The texture as an (H, W, 3) uint8 sRGB array; row 0 is the FAR edge (largest y), column 0 the left edge (smallest x)."""
    p = params
    w = int(round(p["size_m"][0] / p["m_per_px"]))
    h = int(round(p["size_m"][1] / p["m_per_px"]))
    xs = (np.arange(w) + 0.5) * p["m_per_px"]
    ys = (np.arange(h) + 0.5) * p["m_per_px"]
    rng = np.random.Generator(np.random.PCG64(int(p["seed"])))
    nb = int(p["n_boards"])
    bw = p["size_m"][1] / nb
    board = np.minimum((ys / bw).astype(np.int64), nb - 1)                    # board index per row
    draws = rng.random((nb, 4))                                               # y_c, d0, slope, tone per board
    yc = (np.arange(nb) + 0.5) * bw + (2 * draws[:, 0] - 1) * p["log_offset_m"]
    lo, hi = p["log_depth_m"]
    d0 = lo + (hi - lo) * draws[:, 1]
    slope = (2 * draws[:, 2] - 1) * p["log_slope"]
    tone = (2 * draws[:, 3] - 1) * p["board_offset_srgb"]
    depth = d0[board][:, None] + slope[board][:, None] * (xs[None, :] - p["size_m"][0] / 2)
    warp = sum(a * (2 * value_noise(rng, xs, ys, cx, cy) - 1) for cx, cy, a in p["warp"])
    r = np.sqrt((ys - yc[board])[:, None] ** 2 + depth ** 2) + warp
    cx, cy, a = p["phase_noise"]
    phase = r / p["ring_spacing_m"] + a * value_noise(rng, xs, ys, cx, cy)
    t = phase - np.floor(phase)
    late = _smoothstep(0.62, 0.84, t) * (1 - _smoothstep(0.93, 1.0, t))
    ew, lw = np.asarray(p["earlywood_srgb"], float), np.asarray(p["latewood_srgb"], float)
    col = ew + (lw - ew) * (p["latewood_strength"] * late)[..., None] + tone[board][:, None, None]
    cx, cy, a = p["fibre"]
    col += (a * (2 * value_noise(rng, xs, ys, cx, cy) - 1))[..., None]
    cx, cy, thr, dark = p["pores"]
    pn = value_noise(rng, xs, ys, cx, cy)
    col -= (dark * _smoothstep(thr, 1.0, pn) / max(1e-9, 1.0 - thr) * (1.0 - thr))[..., None]
    cx, cy, a = p["board_tone"]
    col += (a * (2 * value_noise(rng, xs, ys, cx, cy) - 1))[..., None]
    img = np.clip(np.round(col), 0, 255).astype(np.uint8)
    return img[::-1].copy()                      # row 0 = far edge (largest y): image 'up' = +y = away from P3


def pixels_sha256(img: np.ndarray) -> str:
    """sha256 of the C-order uint8 RGB pixel bytes prefixed by the shape ('HxWx3:')."""
    a = np.ascontiguousarray(img, np.uint8)
    return hashlib.sha256(f"{a.shape[0]}x{a.shape[1]}x{a.shape[2]}:".encode() + a.tobytes()).hexdigest()


def png_bytes(img: np.ndarray) -> bytes:
    """A minimal 8-bit RGB PNG: one IDAT, filter 0 on every row, zlib level 9; no other chunks (no time stamp)."""
    a = np.ascontiguousarray(img, np.uint8)
    h, w, c = a.shape
    if c != 3:
        raise ValueError("RGB only")
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def read_png_pixels(path: str | Path) -> np.ndarray:
    """Pixels of a PNG written by png_bytes (filter 0, RGB, one or more IDAT chunks)."""
    b = Path(path).read_bytes()
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(b):
        n = struct.unpack(">I", b[pos:pos + 4])[0]
        kind, data = b[pos + 4:pos + 8], b[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", data[:10])
            if (depth, ctype) != (8, 2):
                raise ValueError(f"{path}: only 8-bit RGB")
        elif kind == b"IDAT":
            idat += data
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + 3 * w)
    if np.any(raw[:, 0] != 0):
        raise ValueError(f"{path}: filtered rows (not written by png_bytes)")
    return raw[:, 1:].reshape(h, w, 3).copy()


def write(path: str | Path, params: dict = PARAMS) -> dict:
    img = generate(params)
    data = png_bytes(img)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return {"path": str(path), "width_px": img.shape[1], "height_px": img.shape[0], "pixels_sha256": pixels_sha256(img),
            "file_sha256": hashlib.sha256(data).hexdigest(), "mean_srgb": [round(float(v), 2) for v in img.reshape(-1, 3).mean(0)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.wood_texture", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    info = write(a.out)
    print(" ".join(f"{k}={v}" for k, v in info.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
