"""MuJoCo backend for sim/rig.py: rig -> MJCF, RGB and label renders per camera, visible fractions, and a CLI.

Why MuJoCo: it renders headless through EGL on the Jetson (MuJoCo 3.17, ~0.1 s per 2304x1296 segmentation render)
with exact per-geom segmentation, which is all a visibility check needs. Nothing is simulated: every body is static
or a mocap body (the mug and the occluder, so sweeps can move them without recompiling), and collisions are off.

Visible fraction of a target (an object label from rig.labels(), e.g. 'mug') at a camera = target pixels in the full
scene / target pixels with every other geom hidden (moved to geom group 5, which MuJoCo does not draw by default).
The 'alone' render depends only on the target's own pose, so it is cached by that pose (as pixel indices).

Limits (see sim/README.md): pinhole only (the fitted k1 is not rendered), flat colours, one headlight and one
directional light, the mug is a closed solid (top cap = rim opening) with a plate handle.

MuJoCo is not a project dependency. Run in a throwaway uv env, headless:
  cd <repo> && MUJOCO_GL=egl uv run --no-project --python 3.13.5 --with numpy==2.5.3 --with mujoco==3.17.0 \
      python -m sim.mujoco_rig fractions --positions <positions draft> [--occluders all] [--json out.json]
Subcommands: fractions (visible fraction and H/C/p pattern per occluder, next to the geometry's intended pattern),
render (RGB and label PNGs per stop), mjcf (write the model XML).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import struct
import sys
import time
import zlib
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")      # headless default; must be set before mujoco is imported
import mujoco  # noqa: E402

from sim import rig as R  # noqa: E402

HIDDEN_GROUP = 5          # mjvOption.geomgroup defaults to groups 0-2 visible, so group 5 is never drawn
ALONE_CACHE = 512         # cached 'alone' renders (pixel-index arrays, ~30 KB each for the mug at 2304x1296)


def _f(v: Iterable[float]) -> str:
    return " ".join(f"{x:.12g}" for x in v)


def _mesh_xml(m: R.Mesh) -> str:
    v = " ".join(f"{x:.12g}" for x in np.asarray(m.vertices, float).ravel())
    f = " ".join(str(int(i)) for i in np.asarray(m.faces).ravel())
    return f'    <mesh name="{m.name}" vertex="{v}" face="{f}"/>'


def build_mjcf(rig: R.Rig, occluder_slot: bool = True, offsamples: int = 0) -> str:
    """MJCF for the rig. The mug and the occluder are mocap bodies; with occluder_slot an occluder geom exists even when
    rig.occluder is None (hidden in group 5) so MujocoRig.set_occluder can place one later. Cameras use fovy from
    f_px and the image height, so pixels are square and the principal point is the image centre. offsamples is the
    offscreen multisampling: keep 0 for label renders (see MujocoRig)."""
    if not rig.cameras:
        raise ValueError("rig has no cameras (load it with a positions file)")
    W, H = rig.cameras[0].width, rig.cameras[0].height
    if any((c.width, c.height) != (W, H) for c in rig.cameras):
        raise ValueError("all cameras must share one image size")
    meshes = rig.meshes()
    L = ['<mujoco model="cmpe249_rig">',
         '  <compiler angle="radian"/>',
         '  <visual>',
         f'    <global offwidth="{W}" offheight="{H}"/>',
         f'    <quality offsamples="{offsamples}"/>',
         '    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0 0 0"/>',
         '  </visual>',
         '  <default><geom contype="0" conaffinity="0"/></default>',
         '  <asset>'] + [_mesh_xml(m) for m in meshes] + ['  </asset>', '  <worldbody>',
         '    <light name="key" directional="true" pos="0.4 -1.5 1.5" dir="-0.25 0.75 -0.6" diffuse="0.45 0.45 0.45"'
         ' specular="0 0 0" castshadow="false"/>',
         '    <body name="static">']
    for b in rig.boxes + [bx for t in rig.tags for bx in t.boxes()]:
        L.append(f'      <geom name="{b.name}" type="box" pos="{_f(b.centre)}" size="{_f(b.half_size)}" '
                 f'quat="{_f(b.quat)}" rgba="{_f(b.rgba)}"/>')
    L.append('    </body>')
    L.append(f'    <body name="mug" mocap="true" pos="{_f((rig.mug.x, rig.mug.y, 0.0))}">')
    for m in meshes:
        L.append(f'      <geom name="{m.name}" type="mesh" mesh="{m.name}" rgba="{_f(m.rgba)}"/>')
    L.append('    </body>')
    ob = rig.occluder_box()
    if ob is not None or occluder_slot:
        c, hs, grp = (ob.centre, ob.half_size, 0) if ob is not None else ((0.0, 0.1, 0.05), (0.01, 0.01, 0.01), HIDDEN_GROUP)
        L.append(f'    <body name="occluder" mocap="true" pos="{_f(c)}">')
        L.append(f'      <geom name="occluder" type="box" size="{_f(hs)}" group="{grp}" rgba="{_f(R.RGBA["occluder"])}"/>')
        L.append('    </body>')
    for cam in rig.cameras:
        right, up, _ = cam.axes()
        L.append(f'    <camera name="{cam.name}" pos="{_f(cam.position)}" xyaxes="{_f(right)} {_f(up)}" '
                 f'fovy="{cam.fovy_deg:.12g}"/>')
    L += ['  </worldbody>', '</mujoco>']
    return "\n".join(L) + "\n"


class MujocoRig:
    """Compiled rig + offscreen renderer. Use as a context manager (or call close()) so the GL context is released.

    offsamples must stay 0 for label renders: MuJoCo encodes segmentation ids as flat colours, and multisampling
    blends them at object edges into colours that decode to OTHER geoms (measured 2026-11-02 on the A-mug rig: 4x MSAA
    put stray 'tag' pixels along the mug outline and cost the mug 0.2-0.4 % of its pixels). Use a separate
    MujocoRig(rig, offsamples=4) for smooth RGB previews; it refuses render_labels."""

    def __init__(self, rig: R.Rig, occluder_slot: bool = True, offsamples: int = 0):
        self.rig = rig
        self.offsamples = offsamples
        self.xml = build_mjcf(rig, occluder_slot, offsamples)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        cam0 = rig.cameras[0]
        self.width, self.height = cam0.width, cam0.height
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.labels = rig.labels()
        self._label_index = {n: i for i, n in enumerate(self.labels)}
        self.geom_label = np.full(self.model.ngeom, R.BACKGROUND, np.int32)
        for g in range(self.model.ngeom):
            self.geom_label[g] = self._label_index[self._label_of_geom(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g))]
        self._mocap = {n: int(self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)])
                       for n in ("mug", "occluder") if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
        self._occ_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "occluder")
        self._alone_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.timing = {"renders": 0, "seconds": 0.0}

    def _label_of_geom(self, name: str) -> str:
        if name in ("mug_body", "mug_handle"):
            return "mug"
        if name == "occluder":
            return "occluder"
        if name.startswith("card"):
            return "tag_cards"
        if name.startswith("tag"):
            return name
        return "shelf"

    def label_id(self, label: str) -> int:
        return self._label_index[label]

    # ------------------------------------------------------------ scene edits (no recompilation)
    def set_occluder(self, spec: R.OccluderSpec | str | None) -> None:
        if self._occ_geom < 0:
            raise RuntimeError("model was built without an occluder slot")
        spec = R.resolve_occluder(self.rig.geometry, spec)
        self.rig = self.rig.with_occluder(spec)
        g = self._occ_geom
        if spec is None:
            self.model.geom_group[g] = HIDDEN_GROUP
        else:
            b = spec.box(self.rig.mug.x)
            hs = np.asarray(b.half_size)
            self.model.geom_size[g] = hs
            self.model.geom_rbound[g] = float(np.linalg.norm(hs))
            self.model.geom_aabb[g] = np.r_[0.0, 0.0, 0.0, hs]
            self.model.geom_group[g] = 0
            self.data.mocap_pos[self._mocap["occluder"]] = b.centre
        mujoco.mj_forward(self.model, self.data)

    def set_mug(self, x: float | None = None, y: float | None = None) -> None:
        """Move the mug axis to (x, y) on the shelf. An occluder follows the mug laterally (x centre = mug x + shift, as
        occ_box(u + shift, ...) in the analytic model) and keeps its absolute depth y_back."""
        x = self.rig.mug.x if x is None else x
        y = self.rig.mug.y if y is None else y
        if (x, y) == (self.rig.mug.x, self.rig.mug.y):
            return
        self.rig = self.rig.with_mug(x=x, y=y)
        self.data.mocap_pos[self._mocap["mug"]] = (x, y, 0.0)
        mujoco.mj_forward(self.model, self.data)
        if self.rig.occluder is not None:
            self.set_occluder(self.rig.occluder)

    # ------------------------------------------------------------ rendering
    def render_rgb(self, camera: str) -> np.ndarray:
        t0 = time.perf_counter()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        img = self.renderer.render().copy()
        self._tick(t0)
        return img

    def render_labels(self, camera: str, only: Sequence[str] | None = None) -> np.ndarray:
        """H x W int32 label mask (index into self.labels, -1 = background). only: draw just these labels."""
        if self.offsamples:
            raise RuntimeError("label renders need offsamples=0 (multisampling corrupts segmentation ids)")
        t0 = time.perf_counter()
        saved = None
        if only is not None:
            keep = np.isin(self.geom_label, [self._label_index[n] for n in only])
            saved = self.model.geom_group.copy()
            self.model.geom_group[~keep] = HIDDEN_GROUP
        try:
            self.renderer.enable_segmentation_rendering()
            self.renderer.update_scene(self.data, camera=camera)
            seg = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
            if saved is not None:
                self.model.geom_group[:] = saved
        ids, types = seg[..., 0], seg[..., 1]
        out = np.full(ids.shape, R.BACKGROUND, np.int32)
        is_geom = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        out[is_geom] = self.geom_label[ids[is_geom]]
        self._tick(t0)
        return out

    def _tick(self, t0: float) -> None:
        self.timing["renders"] += 1
        self.timing["seconds"] += time.perf_counter() - t0

    def _pose_key(self, target: str) -> tuple:
        """What the target's unoccluded render depends on (the mug and occluder move; everything else is static)."""
        if target == "mug":
            m = self.rig.mug
            return (m.x, m.y)
        if target == "occluder":
            return () if self.rig.occluder is None else dataclasses.astuple(self.rig.occluder) + (self.rig.mug.x,)
        return ()

    def alone_pixels(self, target: str, camera: str) -> np.ndarray:
        """Flat pixel indices of the target rendered with everything else hidden; cached by the target's pose (LRU of
        ALONE_CACHE entries, so long sweeps keep a bounded footprint)."""
        key = (target, camera, self._pose_key(target))
        if key in self._alone_cache:
            self._alone_cache.move_to_end(key)
        else:
            lab = self.render_labels(camera, only=[target])
            self._alone_cache[key] = np.flatnonzero(lab == self._label_index[target]).astype(np.int32)
            if len(self._alone_cache) > ALONE_CACHE:
                self._alone_cache.popitem(last=False)
        return self._alone_cache[key]

    def alone_mask(self, target: str, camera: str) -> np.ndarray:
        """Label mask of the target alone (index of `target`, -1 elsewhere)."""
        out = np.full(self.height * self.width, R.BACKGROUND, np.int32)
        out[self.alone_pixels(target, camera)] = self._label_index[target]
        return out.reshape(self.height, self.width)

    def visible_fraction(self, target: str, camera: str, full: np.ndarray | None = None) -> dict:
        """{'fraction', 'visible_px', 'alone_px'} for one target at one camera (full: a label mask to reuse)."""
        full = self.render_labels(camera) if full is None else full
        idx = self.alone_pixels(target, camera)
        n_vis = int(np.count_nonzero(full.ravel()[idx] == self._label_index[target]))
        return {"fraction": (n_vis / len(idx)) if len(idx) else None, "visible_px": n_vis, "alone_px": int(len(idx))}

    def fractions(self, target: str = "mug", cameras: Sequence[str] | None = None) -> list[dict]:
        cams = [c.name for c in self.rig.cameras] if cameras is None else list(cameras)
        return [{"camera": c, **self.visible_fraction(target, c)} for c in cams]

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self) -> MujocoRig:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def visible_fraction(rig: R.Rig, target: str, camera: str) -> float | None:
    """One-shot convenience (compiles a model and a renderer; use MujocoRig for sweeps)."""
    with MujocoRig(rig) as mr:
        return mr.visible_fraction(target, camera)["fraction"]


# ---------------------------------------------------------------- image output (stdlib PNG, no extra deps)

PALETTE = np.array([[230, 230, 220], [214, 160, 60], [70, 130, 210], [200, 40, 40], [40, 160, 60], [150, 60, 180],
                    [230, 140, 20], [250, 250, 250]], np.uint8)


def colourise(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(labels.shape + (3,), np.uint8)
    m = labels >= 0
    out[m] = PALETTE[labels[m] % len(PALETTE)]
    return out


def write_png(path: str | Path, img: np.ndarray) -> None:
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape[:2]
    ctype = 0 if img.ndim == 2 else {3: 2, 4: 6}[img.shape[2]]      # grey, RGB, RGBA
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- CLI

def _load(args: argparse.Namespace, occluder=None) -> R.Rig:
    return R.load_rig(args.geometry, args.positions, occluder=occluder, mug_z=args.mug_z, mug_x=args.mug_x,
                      handle=None if args.handle == "none" else args.handle, handle_thickness=args.handle_thickness,
                      roll_deg=args.roll_deg)


def cmd_fractions(args: argparse.Namespace) -> int:
    rig = _load(args)
    names = R.occluder_names(rig.geometry) if args.occluders == "all" else [n for n in args.occluders.split(",") if n]
    rows = []
    t0 = time.perf_counter()
    with MujocoRig(rig) as mr:
        for name in [None] + names:
            mr.set_occluder(name)
            fr = mr.fractions("mug")
            v = [r["fraction"] for r in fr]
            want = None if name is None else R.occluder_spec(rig.geometry, name).intended_pattern
            rows.append({"occluder": name or "none", "fractions": [None if x is None else round(x, 4) for x in v],
                         "pattern": R.pattern(v), "intended_pattern": want,
                         "alone_px": [r["alone_px"] for r in fr], "visible_px": [r["visible_px"] for r in fr]})
            print(f"{name or 'none':12s} {R.pattern(v)}  intended {want or '-':5s}  "
                  + " ".join("  -  " if x is None else f"{x:.3f}" for x in v))
        timing = dict(mr.timing, wall_s=round(time.perf_counter() - t0, 2))
    print(f"renders {timing['renders']} in {timing['seconds']:.1f} s ({timing['seconds'] / max(1, timing['renders']):.3f} "
          f"s each); wall {timing['wall_s']} s")
    if args.json:
        Path(args.json).write_text(json.dumps({"positions": str(args.positions), "mug_z": args.mug_z, "mug_x": args.mug_x,
                                               "handle": args.handle, "stops": [c.name for c in rig.cameras],
                                               "rows": rows, "timing": timing}, indent=1) + "\n")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    rig = _load(args, occluder=None if args.occluder == "none" else args.occluder)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with MujocoRig(rig) as mr:
        for cam in rig.cameras:
            write_png(out / f"{cam.name}_rgb.png", mr.render_rgb(cam.name))
            if args.labels:
                write_png(out / f"{cam.name}_labels.png", colourise(mr.render_labels(cam.name)))
    print(f"wrote {len(rig.cameras)} stop(s) to {out}")
    return 0


def cmd_mjcf(args: argparse.Namespace) -> int:
    rig = _load(args, occluder=None if args.occluder == "none" else args.occluder)
    Path(args.out).write_text(build_mjcf(rig))
    print(f"wrote {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.mujoco_rig", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fractions", cmd_fractions), ("render", cmd_render), ("mjcf", cmd_mjcf)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--geometry", default=str(R.DEFAULT_GEOMETRY_PATH))
        p.add_argument("--positions", required=True, help="positions file (schema cmpe249.positions.v1), e.g. a far draft")
        p.add_argument("--mug-z", type=float, default=0.24, help="mug axis depth behind the #3 front plane, m (design 'z')")
        p.add_argument("--mug-x", type=float, default=0.0)
        p.add_argument("--handle", default="toward", choices=["toward", "away", "left", "right", "none"])
        p.add_argument("--handle-thickness", type=float, default=None, help="m; default from the geometry")
        p.add_argument("--roll-deg", type=float, default=None, help="default: the P3 fitted roll for every stop")
        if name == "fractions":
            p.add_argument("--occluders", default="all", help="'all' or comma-separated names from the geometry")
            p.add_argument("--json", default=None)
        else:
            p.add_argument("--occluder", default="none")
        if name == "render":
            p.add_argument("--out-dir", required=True)
            p.add_argument("--labels", action="store_true", help="also write colourised label masks")
        if name == "mjcf":
            p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
