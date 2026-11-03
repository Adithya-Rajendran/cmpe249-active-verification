"""Isaac Sim backend for the simulated rig: scene spec -> USD stage -> RTX renders with target masks.

Why a second backend: MuJoCo (sim/mujoco_rig.py) gives exact visibility but flat, game-like images,
and DINOv2-style matching is sensitive to render realism (CNOS, ICCVW 2023: photoreal templates 40.4
vs simple renders 36.1 mAP on real images). Isaac Sim's RTX renderer is the realistic one, and it is
supported on the GB10 (DGX Spark, aarch64). Both backends read the same scene spec, so their visible
fractions must agree. On the 2026-11-02 toy scene they agreed to within 0.012 at every stop
(notes/2026-11-02-simulation.md).

Runs in the Isaac Sim 6.2 env on the GB10 (~/cmpe249-sim/isaacsim-6.2/.venv, Python 3.12), not in the
project env. Isaac Sim needs Python 3.12 and its own torch; the student approved a separate env on
2026-11-02. Isaac, omni and pxr are imported only inside the functions that need them, so the pure
parts (PLY reading, camera and intrinsics math) import and test anywhere with numpy.

    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 OMNI_KIT_ACCEPT_EULA=YES \\
      ~/.local/bin/uv run --no-project --python ~/cmpe249-sim/isaacsim-6.2/.venv/bin/python \\
      isaac_rig.py --scene scene.json --tags-dir tags --out out [--lighting L1 L2] \\
      [--target-asset models/obj_000014.ply] [--extra-asset models/obj_000012.ply@0.20,0.15,0[,z]]

Input: a `cmpe249.sim-scene.v1` JSON (`python -m sim.rig ... --scene-out`). World frame in metres:
x right as seen from the camera, y into the shelf, z up. It holds boxes {name, role, center, half,
material}, meshes {name, role, vertices, faces, material}, tags {id, dictionary, corners TL,TR,BR,BL,
size_m}, and cameras {name, position, forward, up, f_px, width, height}.

`--target-asset` swaps the role-"target" mesh for a textured model, such as the YCB-Video BOP mug
(`ycbv_models.zip`, obj 14; mm, Z-up; CC BY 4.0 per the YCB project). The model's base is set on the
target's base centre, so appearance runs use a real mug; geometry runs keep the measured-profile
mesh.

Output: rgb_<cam>_<L>.png per camera and lighting, and target_<cam>.png, the target's visible mask
(segmentation does not depend on lighting). visibility.json holds the target's visible fraction per camera: pixels with everything shown ÷
pixels with the occluder-role prims hidden. The occlusionRatio annotator returns -1 for these prims
on 6.2, so masks are used instead.

Lighting presets are rough stand-ins for the rig, not a photometric model. L1 = dome + two area
lights ("all lights", 195 lux at the reference in pilot 2). L2 = one area light off (66-69 lux).
Exposure is fixed by ACES filmIso, so L2 renders darker, as the real L2 photos are.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

SCENE_SCHEMA = "cmpe249.sim-scene.v1"
FOCAL_MM = 4.74  # Pi Camera Module 3 lens; apertures follow from f_px so f/aperture matches the fit
SETTLE_UPDATES = 200  # app.update() calls before the first capture (NVIDIA capture_pipeline.py pattern)
MATERIALS = {  # name -> (base colour RGB 0-1, roughness)
    "wood": ((0.72, 0.58, 0.42), 0.6),
    "white": ((0.90, 0.90, 0.88), 0.7),
    "occluder_white": ((0.95, 0.95, 0.95), 0.5),
    "floor": ((0.50, 0.47, 0.43), 0.8),
    "mug_blue": ((0.15, 0.35, 0.75), 0.3),
    "grey": ((0.6, 0.6, 0.6), 0.6),
}
LIGHTING = {  # preset -> (dome intensity, [(name, position, size_m, intensity, on)])
    "L1": (300.0, [("key_left", (-0.6, -1.0, 1.4), 0.4, 20000.0, True),
                   ("key_right", (0.6, -1.0, 1.4), 0.4, 20000.0, True)]),
    "L2": (300.0, [("key_left", (-0.6, -1.0, 1.4), 0.4, 20000.0, True),
                   ("key_right", (0.6, -1.0, 1.4), 0.4, 20000.0, False)]),
}


# --------------------------------------------------------------------------- pure helpers
def read_ply(path: str | Path) -> dict:
    """ASCII or binary_little_endian PLY with float vertex properties and triangle faces.

    Returns vertices (N,3), normals (N,3) or None, uv (N,2) or None, faces (M,3) int, and the
    `TextureFile` comment if present. (BOP models: ASCII, x y z nx ny nz texture_u texture_v.)"""
    path = Path(path)
    with open(path, "rb") as f:
        header, props, n_v, n_f, fmt, tex = [], [], 0, 0, None, None
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("comment TextureFile"):
                tex = line.split(maxsplit=2)[2]
            elif line.startswith("element vertex"):
                n_v, section = int(line.split()[2]), "vertex"
            elif line.startswith("element face"):
                n_f, section = int(line.split()[2]), "face"
            elif line.startswith("property") and n_f == 0 and "list" not in line:
                props.append(line.split()[-1])
            elif line == "end_header":
                break
            if len(header) > 200:
                raise ValueError(f"{path}: no end_header")
        if fmt == "ascii":
            body = f.read().decode("ascii").split("\n")
            v = np.array([[float(x) for x in body[i].split()] for i in range(n_v)], np.float64)
            faces = []
            for i in range(n_v, n_v + n_f):
                parts = body[i].split()
                k = int(parts[0])
                idx = [int(x) for x in parts[1:1 + k]]
                faces += [[idx[0], idx[j], idx[j + 1]] for j in range(1, k - 1)]  # fan-triangulate
            faces = np.array(faces, np.int64)
        elif fmt == "binary_little_endian":
            v = np.frombuffer(f.read(4 * len(props) * n_v), "<f4").reshape(n_v, len(props)).astype(np.float64)
            faces = []
            for _ in range(n_f):
                k = f.read(1)[0]
                idx = np.frombuffer(f.read(4 * k), "<i4")
                faces += [[idx[0], idx[j], idx[j + 1]] for j in range(1, k - 1)]
            faces = np.array(faces, np.int64)
        else:
            raise ValueError(f"{path}: unsupported PLY format {fmt!r}")
    col = {p: i for i, p in enumerate(props)}
    out = {"vertices": v[:, [col["x"], col["y"], col["z"]]], "faces": faces, "texture": tex,
           "normals": None, "uv": None}
    if all(k in col for k in ("nx", "ny", "nz")):
        out["normals"] = v[:, [col["nx"], col["ny"], col["nz"]]]
    for u_key, v_key in (("texture_u", "texture_v"), ("s", "t"), ("u", "v")):
        if u_key in col and v_key in col:
            out["uv"] = v[:, [col[u_key], col[v_key]]]
            break
    return out


def camera_matrix(position, forward, up) -> np.ndarray:
    """4x4 USD local-to-world matrix (row-vector convention) for a camera at `position` looking along
    `forward`. USD cameras look down their local -Z with +Y up, so the local axes map to
    x = right, y = up (re-orthogonalised), z = -forward."""
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, float)
    r = np.cross(f, u)
    if np.linalg.norm(r) < 1e-9:
        raise ValueError("camera forward and up are parallel")
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = r, u, -f, np.asarray(position, float)
    return m


def apertures(f_px: float, width: int, height: int, focal_mm: float = FOCAL_MM) -> tuple[float, float]:
    """(horizontal, vertical) aperture in the focal length's units, so the pinhole has focal length
    f_px pixels at width x height with square pixels."""
    return focal_mm * width / f_px, focal_mm * height / f_px


def quat_matrix(q_wxyz) -> np.ndarray:
    """3x3 rotation (column-vector convention) from a unit quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def box_matrix(center, half, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    """4x4 USD local-to-world matrix (row vectors) for a size-2 cube scaled to `half`, rotated by
    the quaternion, then moved to `center`: world = local * diag(half) * R^T + center."""
    m = np.eye(4)
    m[:3, :3] = np.diag(np.asarray(half, float)) @ quat_matrix(quat_wxyz).T
    m[3, :3] = np.asarray(center, float)
    return m


TAG_FRONT_OFFSET_M = 0.0006  # markers 0.6 mm in front of the front plane (cards are at 0.2-0.4 mm), as in MuJoCo


def tag_quad(corners, offset_m: float = TAG_FRONT_OFFSET_M) -> np.ndarray:
    """The tag's corners (TL, TR, BR, BL as seen from the front) moved toward the viewer by offset_m,
    so the marker sits in front of its card instead of z-fighting with it."""
    c = np.asarray(corners, float)
    n = np.cross(c[1] - c[0], c[3] - c[0])  # points into the shelf for this corner order
    return c - offset_m * n / np.linalg.norm(n)


def _safe(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def load_scene(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text())
    if spec.get("schema") != SCENE_SCHEMA:
        raise ValueError(f"{path}: schema {spec.get('schema')!r}, expected {SCENE_SCHEMA!r}")
    return spec


# ------------------------------------------------------------------------------ USD side
def _material(stage, name: str, rgb, roughness: float, texture: str | None = None):
    from pxr import Sdf, UsdShade

    path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if texture:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(float(c) for c in rgb))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _mesh(stage, path: str, vertices, faces, normals=None, uv=None):
    from pxr import Gf, UsdGeom, Vt

    m = UsdGeom.Mesh.Define(stage, path)
    m.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in vertices]))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for i in np.asarray(faces).reshape(-1)]))
    m.GetSubdivisionSchemeAttr().Set("none")
    if normals is not None:
        m.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in normals]))
        m.SetNormalsInterpolation("vertex")
    if uv is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", _texcoord_type(), UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in uv]))
    return m.GetPrim()


def _texcoord_type():
    from pxr import Sdf

    return Sdf.ValueTypeNames.TexCoord2fArray


def _asset_mesh(stage, path: str, ply: str | Path, base_xy, base_z: float, yaw_deg: float = 0.0):
    """A BOP model (mm, Z-up, roughly centred) placed so its lowest point sits at base_z and its
    bounding-box centre at base_xy, rotated by yaw about z. Returns (prim, texture path or None)."""
    d = read_ply(ply)
    v = d["vertices"] / 1000.0
    c = (v.max(0) + v.min(0)) / 2
    v = v - [c[0], c[1], v[:, 2].min()]
    a = math.radians(yaw_deg)
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    v = v @ rot.T + [base_xy[0], base_xy[1], base_z]
    n = None if d["normals"] is None else d["normals"] @ rot.T
    prim = _mesh(stage, path, v, d["faces"], n, d["uv"])
    tex = None if not d["texture"] else str(Path(ply).parent / d["texture"])
    return prim, tex


def build_stage(spec: dict, tags_dir: str | Path, lighting: str = "L1", target_asset: str | None = None,
                extra_assets: list[tuple[str, tuple[float, float, float, float]]] = ()) -> dict:
    """Populate the current Isaac stage from a scene spec. Returns {"cameras": {name: path},
    "targets": [paths], "occluders": [paths], "lights": {name: path}}."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    mats = {k: _material(stage, k, rgb, r) for k, (rgb, r) in MATERIALS.items()}
    for k, m in (spec.get("materials") or {}).items():  # the spec's own colours win over the defaults
        rgba = m.get("rgba", [0.6, 0.6, 0.6, 1.0])
        mats[k] = _material(stage, f"spec_{_safe(k)}", rgba[:3], m.get("roughness", 0.6))
    out = {"cameras": {}, "targets": [], "occluders": [], "lights": {}}

    for b in spec.get("boxes", []):
        p = f"/World/{_safe(b['role'])}_{_safe(b['name'])}"
        cube = UsdGeom.Cube.Define(stage, p)
        cube.GetSizeAttr().Set(2.0)  # -1..1, scaled to the half extents
        m = box_matrix(b["center"], b["half"], b.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)))
        UsdGeom.Xformable(cube.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*m.reshape(-1).tolist()))
        _bind(cube.GetPrim(), mats.get(b.get("material", "grey"), mats["grey"]))
        if b.get("role") == "occluder":
            out["occluders"].append(p)

    targets = [m for m in spec.get("meshes", []) if m.get("role") == "target"]
    body = next((m for m in targets if "body" in m["name"]), targets[0] if targets else None)
    for m in spec.get("meshes", []):
        p = f"/World/{_safe(m['role'])}_{_safe(m['name'])}"
        if m.get("role") == "target" and target_asset:
            if m is not body:
                continue  # the textured model brings its own handle
            v = np.asarray(m["vertices"], float)
            base = ((v[:, 0].max() + v[:, 0].min()) / 2, (v[:, 1].max() + v[:, 1].min()) / 2)
            prim, tex = _asset_mesh(stage, p, target_asset, base, float(v[:, 2].min()), m.get("yaw_deg", 0.0))
            _bind(prim, _material(stage, "target_tex", (0.8, 0.8, 0.8), 0.4, tex))
        else:
            prim = _mesh(stage, p, m["vertices"], m["faces"])
            _bind(prim, mats.get(m.get("material", "grey"), mats["grey"]))
        if m.get("role") == "target":
            out["targets"].append(p)
        if m.get("role") == "occluder":
            out["occluders"].append(p)

    for i, (ply, (x, y, yaw, z)) in enumerate(extra_assets):
        prim, tex = _asset_mesh(stage, f"/World/extra_{i}", ply, (x, y), z, yaw)
        _bind(prim, _material(stage, f"extra_tex_{i}", (0.8, 0.8, 0.8), 0.4, tex))

    for t in spec.get("tags", []):
        c = tag_quad(t["corners"])  # TL, TR, BR, BL as seen from the front, nudged in front of the card
        png = Path(tags_dir) / f"aruco_4x4_50_id{t['id']}.png"
        prim = _mesh(stage, f"/World/tag_{t['id']}", c, [[0, 3, 2], [0, 2, 1]],
                     uv=[(0, 1), (1, 1), (1, 0), (0, 0)])
        UsdGeom.Mesh(prim).GetDoubleSidedAttr().Set(True)  # a flipped winding must not hide the tag
        _bind(prim, _material(stage, f"tag_{t['id']}", (1, 1, 1), 0.8, str(png)))

    UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    for name, pos, size, _, _ in LIGHTING[lighting][1]:
        lt = UsdLux.RectLight.Define(stage, f"/World/Lights/{name}")
        lt.GetWidthAttr().Set(size)
        lt.GetHeightAttr().Set(size)
        target = np.array([0.0, 0.15, 0.1])
        m = camera_matrix(pos, target - np.asarray(pos), (0, 0, 1))  # RectLight emits along its -Z too
        UsdGeom.Xformable(lt.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*m.reshape(-1).tolist()))
        out["lights"][name] = lt.GetPrim().GetPath().pathString
    set_lighting(stage, out, lighting)

    for cam in spec.get("cameras", []):
        p = f"/World/Cameras/{cam['name']}"
        c = UsdGeom.Camera.Define(stage, p)
        h_ap, v_ap = apertures(cam["f_px"], cam["width"], cam["height"])
        c.GetFocalLengthAttr().Set(FOCAL_MM)
        c.GetHorizontalApertureAttr().Set(h_ap)
        c.GetVerticalApertureAttr().Set(v_ap)
        c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        m = camera_matrix(cam["position"], cam["forward"], cam["up"])
        UsdGeom.Xformable(c.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*m.reshape(-1).tolist()))
        out["cameras"][cam["name"]] = p
    return out


def set_lighting(stage, built: dict, preset: str) -> None:
    """Switch a built stage to a lighting preset (an off light gets intensity 0)."""
    from pxr import UsdLux

    dome_i, lights = LIGHTING[preset]
    UsdLux.DomeLight(stage.GetPrimAtPath("/World/Lights/dome")).GetIntensityAttr().Set(dome_i)
    for name, _, _, inten, on in lights:
        UsdLux.RectLight(stage.GetPrimAtPath(built["lights"][name])).GetIntensityAttr().Set(inten if on else 0.0)


def _tonemap() -> None:
    import carb

    s = carb.settings.get_settings()
    for k, v in (("/rtx/post/tonemap/op", 4), ("/rtx/post/tonemap/filmIso", 200.0),
                 ("/rtx/post/tonemap/whitepoint", 6500.0), ("/rtx/post/tonemap/enabled", True),
                 ("/rtx/post/aa/op", 3)):
        s.set(k, v)


def render(app, built: dict, spec: dict, out_dir: Path, lightings: list[str], rt_subframes: int = 16) -> dict:
    """RGB per camera for every lighting preset, and the target mask with the occluders shown and
    hidden (segmentation does not depend on lighting, so masks are taken once)."""
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    cams = {c["name"]: c for c in spec["cameras"]}
    anns = {}
    for name, path in built["cameras"].items():
        rp = rep.create.render_product(path, (cams[name]["width"], cams[name]["height"]))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        seg = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
        rgb.attach(rp)
        seg.attach(rp)
        anns[name] = (rgb, seg)

    def target_mask(seg_data):
        ids, info = seg_data["data"], seg_data["info"].get("idToLabels", {})
        want = [int(k) for k, v in info.items() if any(str(v).startswith(t) for t in built["targets"])]
        return np.isin(ids, want) if want else np.zeros(ids.shape, bool)

    shown = {}
    for i, lighting in enumerate(lightings):
        set_lighting(stage, built, lighting)
        for _ in range(SETTLE_UPDATES if i == 0 else 30):
            app.update()
        rep.orchestrator.step(rt_subframes=rt_subframes)
        for name, (rgb, seg) in anns.items():
            img = rgb.get_data()
            Image.fromarray(img[..., :3]).save(out_dir / f"rgb_{name}_{lighting}.png")
            if i == 0:
                mask = target_mask(seg.get_data())
                Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"target_{name}.png")
                shown[name] = int(mask.sum())
    for p in built["occluders"]:
        UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeInvisible()
    rep.orchestrator.step(rt_subframes=rt_subframes)
    vis = {}
    for name, (rgb, seg) in anns.items():
        full = int(target_mask(seg.get_data()).sum())
        vis[name] = {"target_px_visible": shown[name], "target_px_unoccluded": full,
                     "visible_fraction": round(shown[name] / full, 4) if full else None}
    for p in built["occluders"]:
        UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeVisible()
    return vis


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tags-dir", required=True, help="aruco_4x4_50_id<N>.png files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lighting", nargs="+", default=["L1"], choices=sorted(LIGHTING))
    ap.add_argument("--target-asset", default=None, help="PLY (BOP, mm) to replace the target mesh's look")
    ap.add_argument("--extra-asset", action="append", default=[],
                    help="PLY@x,y,yaw_deg[,z]: an extra textured object standing on z (default 0, the mug-shelf top)")
    ap.add_argument("--renderer", default="RaytracedLighting", choices=["RaytracedLighting", "PathTracing"])
    ap.add_argument("--rt-subframes", type=int, default=16)
    a = ap.parse_args(argv)
    spec = load_scene(a.scene)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    extras = []
    for e in a.extra_asset:
        ply, xyz = e.rsplit("@", 1)
        vals = [float(t) for t in xyz.split(",")]
        x, y, yaw, z = (vals + [0.0])[:4]
        extras.append((ply, (x, y, yaw, z)))
    w, h = spec["cameras"][0]["width"], spec["cameras"][0]["height"]

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": a.renderer, "width": w, "height": h})
    report = {"scene": str(a.scene), "renderer": a.renderer, "target_asset": a.target_asset,
              "lighting": a.lighting}
    try:
        _tonemap()
        built = build_stage(spec, a.tags_dir, a.lighting[0], a.target_asset, extras)
        report["visibility"] = render(app, built, spec, out, a.lighting, a.rt_subframes)
        (out / "visibility.json").write_text(json.dumps(report, indent=1))
        print("ISAAC_RIG_REPORT " + json.dumps(report), flush=True)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
