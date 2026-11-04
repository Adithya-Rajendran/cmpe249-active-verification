"""The tabletop's render path without Isaac (tests/fake_isaac.py, tests/shelf_golden.py's pxr call recorder): sim/isaac_rig.py
and sim/isaac_batch.py build a stage from a scene spec of sim/tabletop.py.

Pinned here (the shelf's own calls and settings are pinned by tests/test_shelf_goldens.py, which must pass unchanged):
  - the table top is a static spec mesh (24 vertices) with the spec's normals and UVs, bound to the textured table_top
    material: one texture file (sha256-verified), clamp wrap, sRGB, roughness / metallic from the spec;
  - the spec's lights stand at the spec positions and emit along the spec's forward; L2 switches key_right off;
  - room parts (floor, back and side walls, baseboards) and the table legs are boxes under /World/Shelf (the container the
    target-alone pass hides);
  - a texture with a wrong sha256, no sha256 or no file is refused (before Isaac starts, in main); UVs outside [0, 1] under a
    clamped texture, an inconsistent lighting block and a light that does not point at the aim are refused;
  - render_settings record the spec's presets, aim, materials and the texture's verified sha256, and change when any of them
    does (--skip-existing re-renders); the shelf's settings have none of these keys;
  - the shelf's isaac_rig.build_stage call log is pinned by a hash recorded from the code before this change.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sim import arrange as A
from sim import isaac_batch as B
from sim import isaac_rig as IR
from sim import tabletop as T
from sim import wood_texture as WT
from tests import fake_isaac as FI
from tests import shelf_golden as G

TEX_NAME = "tabletop_light_oak_v1"
RIG_SHELF_STAGE_LOG = {"n_calls": 2738, "sha256": "3d0c7722376bf535eb297dee340cfbf6a624a341e18870ce176f77b014c2b17c"}   # sim/isaac_rig.py at c354d81


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------- the recorded pxr call log, as a graph
class Log:
    """Navigate tests/shelf_golden.py's call log: entry i is [callee, args, kwargs], its result is named '#i'."""

    def __init__(self, entries: list):
        self.e = entries

    def defined(self, path: str, kind: str | None = None) -> str:
        """Result name of the Define call for `path` (kind: e.g. 'UsdGeom.Mesh')."""
        hits = [i for i, (c, a, _) in enumerate(self.e)
                if c.endswith(".Define") and len(a) > 1 and a[1] == path and (kind is None or c.endswith(f"{kind}.Define"))]
        if len(hits) != 1:
            raise AssertionError(f"{path!r} defined {len(hits)} times")
        return f"#{hits[0]}"

    def paths(self, kind: str) -> list[str]:
        return [a[1] for c, a, _ in self.e if c.endswith(f"{kind}.Define") and len(a) > 1]

    def result(self, callee: str, args: list | None = None) -> list[str]:
        return [f"#{i}" for i, (c, a, _) in enumerate(self.e) if c == callee and (args is None or a == args)]

    def inputs(self, shader_path: str) -> dict:
        """{input name: ('set', value) | ('connect', source)} of a shader / material prim."""
        ref = self.defined(shader_path)
        out = {}
        for i, (c, a, _) in enumerate(self.e):
            if c == f"{ref}.CreateInput":
                res = f"#{i}"
                for j, (c2, a2, _) in enumerate(self.e):
                    if c2 == f"{res}.Set":
                        out[a[0]] = ("set", a2[0])
                    elif c2 == f"{res}.ConnectToSource":
                        out[a[0]] = ("connect", a2)
        return out

    def prim_of(self, ref: str) -> list[str]:
        return self.result(f"{ref}.GetPrim")

    def bound_material(self, prim_path: str) -> str | None:
        """Path of the material the prim at `prim_path` is bound to (Apply(prim).Bind(material))."""
        mesh_or_cube = [f"#{i}" for i, (c, a, _) in enumerate(self.e) if c.endswith(".Define") and len(a) > 1 and a[1] == prim_path]
        assert len(mesh_or_cube) == 1, prim_path
        ref = mesh_or_cube[0]
        prims = {ref, *self.prim_of(ref)}
        for i, (c, a, _) in enumerate(self.e):
            if c == "pxr.UsdShade.MaterialBindingAPI.Apply" and a and a[0] in prims:
                for c2, a2, _ in self.e:
                    if c2 == f"#{i}.Bind":
                        return self.e[int(a2[0][1:])][1][1]
        return None

    def light_matrix(self, path: str) -> np.ndarray:
        """The 4x4 (row-vector) matrix set on the light at `path` (Xformable(light.GetPrim()).AddTransformOp().Set(Matrix4d))."""
        ref = self.defined(path)
        prims = set(self.prim_of(ref))
        for i, (c, a, _) in enumerate(self.e):
            if c == "pxr.UsdGeom.Xformable" and a and a[0] in prims:
                for j, (c2, a2, _) in enumerate(self.e):
                    if c2 == f"#{i}.AddTransformOp":
                        for c3, a3, _ in self.e:
                            if c3 == f"#{j}.Set":
                                return np.array(self.e[int(a3[0][1:])][1], float).reshape(4, 4)
        raise AssertionError(f"no transform set on {path}")

    def path_of(self, ref: str) -> str:
        """The prim path behind a recorded 'light.GetPrim().GetPath().pathString' reference ('#n.pathString'), or ref itself."""
        if not ref.endswith(".pathString"):
            return ref
        get_path = self.e[int(ref[1:].split(".")[0])]                                 # callee '#<GetPrim idx>.GetPath'
        get_prim = self.e[int(get_path[0][1:].split(".")[0])]                          # callee '#<Define idx>.GetPrim'
        return self.e[int(get_prim[0][1:].split(".")[0])][1][1]                        # Define(stage, path)

    def intensities(self) -> list[tuple[str, float]]:
        """(light path, intensity) of every set_lighting write, in call order."""
        out = []
        for i, (c, a, _) in enumerate(self.e):
            if c == "stage.GetPrimAtPath":
                res = f"#{i}"
                for j, (c2, a2, _) in enumerate(self.e):
                    if c2 in ("pxr.UsdLux.RectLight", "pxr.UsdLux.DomeLight") and a2 == [res]:
                        for k, (c3, a3, _) in enumerate(self.e):
                            if c3 == f"#{j}.GetIntensityAttr":
                                for c4, a4, _ in self.e:
                                    if c4 == f"#{k}.Set":
                                        out.append((self.path_of(a[0]), a4[0]))
        return out


def recorded_stage(arr: dict, textures_dir, build=None) -> tuple[list, dict]:
    """tests/shelf_golden.py stage_log with a textures dir: every pxr call of sim/isaac_batch.py build_stage, then
    set_lighting L2 and back to L1. Returns (log entries, built)."""
    log: list = []
    mods = {"pxr": G._Rec(log, "pxr"), "omni": G._Rec(log, "omni"), "omni.usd": G._Rec(log, "omni.usd"),
            "carb": G._Rec(log, "carb")}
    with mock.patch.dict(sys.modules, mods), mock.patch.object(B.ir, "read_ply", G._fake_ply):
        built = B.build_stage(json.loads(json.dumps(arr)), Path("/assets"), Path("/tags"), {}, textures_dir)
        stage = G._Rec(log, "stage")
        B.ir.set_lighting(stage, built, "L2")
        B.ir.set_lighting(stage, built, "L1")
    return log, built


def tabletop_spec(textures_dir: Path | None = None, tex_sha: str | None = None) -> dict:
    """The tabletop scene spec (sim/tabletop.py, the committed positions). With a textures dir, the spec's texture block gets
    that sha256 (a stand-in file) so the test needs not generate the real PNG."""
    spec = T.export_scene_spec(T.load_scene(positions=T.DEFAULT_POSITIONS_PATH))
    if tex_sha is not None:
        spec["materials"]["table_top"]["texture"]["file_sha256"] = tex_sha
    return spec


def tabletop_arrangement(spec: dict) -> dict:
    """A minimal tabletop arrangement for the stage builder and the fake renderer: the spec as the scene (with revisit cameras),
    a YCB-V target, a box occluder."""
    arr = FI.arrangement("TD5000", 5000)
    arr["scene"] = {**copy.deepcopy(spec), "revisit_cameras": copy.deepcopy(spec["cameras"])}
    arr["stops"] = [c["name"] for c in spec["cameras"]]
    arr["objects"]["target"] = {"kind": "ycbv", "ply": "obj_000021.ply", "rest_matrix": np.eye(3).tolist(),
                                "pose": {"base": [0.0, 0.05, 0.0], "yaw_deg": 10.0}}
    arr["objects"]["occluder"]["size_m"] = [0.08, 0.02, 0.2]
    arr["lightings"] = ["L1", "L2"]
    return arr


class _TexDir:
    """A scratch textures dir holding a stand-in PNG (bytes need not be a real image; only the sha256 is checked)."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.data = b"\x89PNG stand-in texture bytes"
        (self.dir / f"{TEX_NAME}.png").write_bytes(self.data)
        self.sha = _sha(self.data)

    def close(self):
        self.tmp.cleanup()


class TestTabletopStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex = _TexDir()
        cls.spec = tabletop_spec(tex_sha=cls.tex.sha)
        cls.arr = tabletop_arrangement(cls.spec)
        cls.entries, cls.built = recorded_stage(cls.arr, cls.tex.dir)
        cls.log = Log(cls.entries)

    @classmethod
    def tearDownClass(cls):
        cls.tex.close()

    def test_top_is_a_uv_mesh_with_the_spec_normals(self):
        top = next(m for m in self.spec["meshes"] if m["name"] == "table_top")
        path = "/World/Shelf/table_table_top"
        self.assertIn(path, self.log.paths("UsdGeom.Mesh"))
        ref = self.log.defined(path)
        self.assertEqual(len(self.log.result(f"{ref}.GetPointsAttr")), 1)
        vec2 = [a for c, a, _ in self.entries if c == "pxr.Gf.Vec2f"]
        n_cam = 2 * len(self.spec["cameras"])                                # Vec2f(near, far) of each camera of both sets
        self.assertEqual(len(vec2), 24 + n_cam)                            # the table top is the only mesh with UVs in the stage
        np.testing.assert_allclose(np.array(vec2[:24]), np.array(top["uv"]), atol=1e-9)
        self.assertTrue(0.0 <= np.array(vec2[:24]).min() and np.array(vec2[:24]).max() <= 1.0)
        pv = [(a, k) for c, a, k in self.entries if c.endswith(".CreatePrimvar")]
        self.assertEqual(len(pv), 1)
        self.assertEqual(pv[0][0][0], "st")
        self.assertEqual(pv[0][0][2], "pxr.UsdGeom.Tokens.vertex")
        normals = [a for c, a, _ in self.entries if c == "pxr.Gf.Vec3f"][24:48]       # points first, then normals
        np.testing.assert_allclose(np.array(normals), np.array(top["normals"]), atol=1e-9)
        self.assertTrue(any(c.endswith(".SetNormalsInterpolation") and a == ["vertex"] for c, a, _ in self.entries))

    def test_texture_is_bound_to_the_top_with_the_spec_parameters(self):
        self.assertEqual(self.log.bound_material("/World/Shelf/table_table_top"), "/World/Looks/spec_table_top")
        tex = self.log.inputs("/World/Looks/spec_table_top/tex")
        self.assertEqual(tex["file"][0], "set")
        self.assertEqual(tex["file"][1], str((self.tex.dir / f"{TEX_NAME}.png").resolve()))
        self.assertEqual(tex["wrapS"], ("set", "clamp"))
        self.assertEqual(tex["wrapT"], ("set", "clamp"))
        self.assertEqual(tex["sourceColorSpace"], ("set", "sRGB"))
        self.assertEqual(tex["st"][0], "connect")                          # straight from the 'st' primvar reader: no repeat, no transform
        self.assertNotIn("/World/Looks/spec_table_top/st_transform", self.log.paths("UsdShade.Shader"))
        self.assertEqual(self.log.inputs("/World/Looks/spec_table_top/st")["varname"], ("set", "st"))
        sh = self.log.inputs("/World/Looks/spec_table_top/Surface")
        self.assertEqual(sh["roughness"], ("set", 0.5))
        self.assertEqual(sh["metallic"], ("set", 0.0))
        self.assertEqual(sh["diffuseColor"][0], "connect")                 # the texture, not the mean colour
        # the textureless spec materials keep the plain colour path
        leg = self.log.inputs("/World/Looks/spec_table_leg/Surface")
        self.assertEqual(leg["roughness"], ("set", 0.6))
        self.assertEqual(leg["diffuseColor"][0], "set")
        self.assertNotIn("metallic", leg)

    def test_uv_transform_when_the_spec_asks_for_one(self):
        spec = copy.deepcopy(self.spec)
        spec["materials"]["table_top"]["texture"].update(uv_scale=[2.0, 1.0], uv_rotation_deg=90.0, uv_translate=[0.0, 0.5])
        # UVs of the mesh stay in [0, 1]: scale / rotation act on the lookup, not on the mesh
        log = Log(recorded_stage(tabletop_arrangement(spec), self.tex.dir)[0])
        tr = log.inputs("/World/Looks/spec_table_top/st_transform")
        self.assertEqual((tr["scale"], tr["rotation"], tr["translation"]), (("set", [2.0, 1.0]), ("set", 90.0), ("set", [0.0, 0.5])))
        self.assertEqual(log.inputs("/World/Looks/spec_table_top/tex")["st"][0], "connect")

    def test_spec_lights_stand_at_the_spec_positions_and_point_along_the_spec_forward(self):
        L = self.spec["lighting"]
        self.assertEqual(set(self.built["lights"]), {"key_left", "key_right"})
        self.assertEqual(self.built["lighting_presets"], IR.spec_lighting(self.spec)["presets"])
        for lt in L["presets"]["L1"]["lights"]:
            m = self.log.light_matrix(f"/World/Lights/{lt['name']}")
            np.testing.assert_allclose(m[3, :3], lt["position"], atol=1e-9)          # translation row
            np.testing.assert_allclose(-m[2, :3], lt["forward"], atol=1e-9)         # a RectLight emits along its local -Z
            aim = np.array(L["aim"]) - np.array(lt["position"])
            np.testing.assert_allclose(-m[2, :3], aim / np.linalg.norm(aim), atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.det(m[:3, :3])), 1.0, places=9)
        # not the shelf's aim: the shelf's would be (0, 0.15, 0.1) - position
        shelf = np.array(IR.SHELF_AIM) - np.array(L["presets"]["L1"]["lights"][0]["position"])
        m = self.log.light_matrix("/World/Lights/key_left")
        self.assertGreater(float(np.abs(-m[2, :3] - shelf / np.linalg.norm(shelf)).max()), 1e-3)
        sizes = [(a[0] if a else None) for c, a, _ in self.entries if c.endswith(".GetWidthAttr")]
        self.assertEqual(len(sizes), 2)

    def test_l2_switches_key_right_off_and_l1_back_on(self):
        writes = self.log.intensities()
        l = self.spec["lighting"]["presets"]
        n_each = 3                                                          # dome + two lights per set_lighting call
        calls = [writes[k:k + n_each] for k in range(0, len(writes), n_each)]
        # the two set_lighting calls made on the recorded stage after build_stage: L2, then L1
        self.assertEqual(len(calls), 2)
        for call, preset in zip(calls, ("L2", "L1")):
            got = dict(call)
            self.assertEqual(got["/World/Lights/dome"], l[preset]["dome_intensity"])
            for lt in l[preset]["lights"]:
                self.assertEqual(got[f"/World/Lights/{lt['name']}"], lt["intensity"] if lt["on"] else 0.0)
        self.assertEqual(dict(calls[0])["/World/Lights/key_right"], 0.0)
        self.assertEqual(dict(calls[1])["/World/Lights/key_right"], 20000.0)

    def test_room_parts_and_legs_are_boxes_under_the_hidden_container(self):
        cubes = self.log.paths("UsdGeom.Cube")
        want = [f"/World/Shelf/{b['role']}_{b['name']}" for b in self.spec["boxes"]]
        names = {b["name"] for b in self.spec["boxes"]}
        for n in ("floor", "wall_back", "wall_left", "wall_right", "baseboard_back", "baseboard_left", "baseboard_right",
                  "leg_near_left", "leg_far_right"):
            self.assertIn(n, names)
        for p in want:
            self.assertIn(p, cubes)
        self.assertEqual(self.log.bound_material("/World/Shelf/room_wall_left"), "/World/Looks/spec_wall")
        self.assertEqual(self.log.bound_material("/World/Shelf/room_floor"), "/World/Looks/spec_floor")
        self.assertEqual(self.log.bound_material("/World/Shelf/room_baseboard_right"), "/World/Looks/spec_baseboard")
        self.assertEqual(self.log.bound_material("/World/Shelf/table_leg_near_left"), "/World/Looks/spec_table_leg")
        self.assertEqual(self.built["shelf"], "/World/Shelf")
        self.assertTrue(all(p.startswith("/World/Shelf/") for p in want))

    def test_no_tags_and_cameras_from_the_spec(self):
        self.assertFalse([p for p in self.log.paths("UsdGeom.Mesh") if "tag_" in p])
        self.assertEqual(set(self.built["cameras"]), {c["name"] for c in self.spec["cameras"]})
        self.assertEqual(set(self.built["cameras_revisit"]), set(self.built["cameras"]))

    def test_the_stage_is_deterministic(self):
        again, _ = recorded_stage(self.arr, self.tex.dir)
        self.assertEqual(G.sha256_json(again), G.sha256_json(self.entries))


class TestTextureChecks(unittest.TestCase):
    def setUp(self):
        self.tex = _TexDir()
        self.addCleanup(self.tex.close)
        self.spec = tabletop_spec(tex_sha=self.tex.sha)

    def test_verify_texture_returns_path_and_sha(self):
        r = IR.verify_texture(self.spec["materials"]["table_top"]["texture"], self.tex.dir)
        self.assertEqual((r["sha256"], r["name"]), (self.tex.sha, TEX_NAME))
        self.assertEqual(r["path"], str((self.tex.dir / f"{TEX_NAME}.png").resolve()))
        self.assertEqual((r["params"]["wrap"], r["params"]["colour_space"], r["params"]["uv_transform"]), ("clamp", "sRGB", False))

    def test_a_wrong_sha256_is_refused_before_anything_is_built(self):
        spec = copy.deepcopy(self.spec)
        spec["materials"]["table_top"]["texture"]["file_sha256"] = "0" * 64
        with self.assertRaisesRegex(IR.TextureError, "refusing to render"):
            IR.verify_texture(spec["materials"]["table_top"]["texture"], self.tex.dir)
        with self.assertRaises(IR.TextureError):
            recorded_stage(tabletop_arrangement(spec), self.tex.dir)
        # and a changed file is caught the same way
        (self.tex.dir / f"{TEX_NAME}.png").write_bytes(self.tex.data + b"x")
        with self.assertRaises(IR.TextureError):
            recorded_stage(tabletop_arrangement(self.spec), self.tex.dir)

    def test_no_sha256_and_no_file_are_refused(self):
        spec = copy.deepcopy(self.spec)
        del spec["materials"]["table_top"]["texture"]["file_sha256"]
        with self.assertRaisesRegex(IR.TextureError, "no file_sha256"):
            IR.verify_texture(spec["materials"]["table_top"]["texture"], self.tex.dir)
        with self.assertRaisesRegex(IR.TextureError, "not found"):
            IR.verify_texture(self.spec["materials"]["table_top"]["texture"], self.tex.dir / "elsewhere")

    def test_file_key_is_resolved_against_the_textures_dir_or_absolute(self):
        spec = copy.deepcopy(self.spec)
        (self.tex.dir / "other.png").write_bytes(self.tex.data)
        spec["materials"]["table_top"]["texture"]["file"] = "other.png"
        self.assertTrue(IR.verify_texture(spec["materials"]["table_top"]["texture"], self.tex.dir)["path"].endswith("other.png"))
        spec["materials"]["table_top"]["texture"]["file"] = str(self.tex.dir / "other.png")
        self.assertTrue(IR.verify_texture(spec["materials"]["table_top"]["texture"], "/nonexistent")["path"].endswith("other.png"))

    def test_texture_params(self):
        self.assertEqual(IR.texture_params({"wrap": "clamp (sim/isaac_rig.py _material); UVs stay in [0, 1]"})["wrap"], "clamp")
        self.assertEqual(IR.texture_params({"wrap": "repeat"})["wrap"], "repeat")
        with self.assertRaises(IR.TextureError):
            IR.texture_params({"wrap": "tile"})
        with self.assertRaises(IR.TextureError):
            IR.texture_params({"colour_space": "linear"})
        self.assertTrue(IR.texture_params({"uv_rotation_deg": 90})["uv_transform"])

    def test_uvs_outside_the_unit_square_are_refused_under_a_clamped_texture(self):
        spec = copy.deepcopy(self.spec)
        spec["meshes"][0]["uv"][3] = [1.2, 0.5]
        with self.assertRaisesRegex(ValueError, "outside \\[0, 1\\]"):
            recorded_stage(tabletop_arrangement(spec), self.tex.dir)
        spec["materials"]["table_top"]["texture"]["wrap"] = "repeat"          # a repeating texture may use them
        recorded_stage(tabletop_arrangement(spec), self.tex.dir)
        spec["meshes"][0]["uv_interpolation"] = "faceVarying"
        with self.assertRaisesRegex(ValueError, "uv_interpolation"):
            recorded_stage(tabletop_arrangement(spec), self.tex.dir)

    def test_the_real_texture_matches_the_geometry_pins(self):
        """sim/wood_texture.py's PNG is what the spec pins: pixels always; the file bytes under the project's Python (3.13.5)."""
        with tempfile.TemporaryDirectory() as t:
            info = WT.write(Path(t) / f"{TEX_NAME}.png")
            spec = tabletop_spec()
            tex = spec["materials"]["table_top"]["texture"]
            self.assertEqual(info["pixels_sha256"], tex["pixels_sha256"])
            if info["file_sha256"] == tex["file_sha256"]:
                self.assertEqual(IR.verify_texture(tex, t)["sha256"], tex["file_sha256"])
            else:                                                             # another zlib: the file pin is not met, so it is refused
                with self.assertRaises(IR.TextureError):
                    IR.verify_texture(tex, t)


class TestSpecLighting(unittest.TestCase):
    def setUp(self):
        self.spec = tabletop_spec()

    def test_shelf_specs_have_no_lighting_and_use_the_shelf_presets(self):
        self.assertIsNone(IR.spec_lighting({"cameras": []}))
        self.assertIs(IR.lighting_presets({"cameras": []}), IR.LIGHTING)
        self.assertIs(IR.lighting_presets(None), IR.LIGHTING)
        self.assertEqual(IR.spec_render_settings({"cameras": [], "materials": {"a": {"rgba": [1, 1, 1, 1]}}}), {})

    def test_tabletop_presets_equal_the_shelf_numbers(self):
        lit = IR.spec_lighting(self.spec)
        self.assertEqual(set(lit["presets"]), {"L1", "L2"})
        for k, (dome, lights) in IR.LIGHTING.items():
            d2, l2 = lit["presets"][k]
            self.assertEqual(d2, dome)
            self.assertEqual([(n, s, i, on) for n, _, s, i, on in l2], [(n, s, i, on) for n, _, s, i, on in lights])
        self.assertEqual(lit["aim"], (0.0, 0.0, 0.05))

    def test_inconsistent_lighting_is_refused(self):
        s = copy.deepcopy(self.spec)
        s["lighting"]["presets"]["L2"]["lights"][0]["position"][0] += 0.1
        s["lighting"]["presets"]["L2"]["lights"][0]["forward"] = None
        with self.assertRaisesRegex(ValueError, "different lights"):
            IR.spec_lighting(s)
        s = copy.deepcopy(self.spec)
        s["lighting"]["presets"]["L1"]["lights"][0]["forward"] = [0.0, 0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "does not point toward aim"):
            IR.spec_lighting(s)
        s = copy.deepcopy(self.spec)
        s["lighting"]["presets"]["L1"]["lights"][0]["type"] = "sphere"
        with self.assertRaisesRegex(ValueError, "only rect"):
            IR.spec_lighting(s)
        s = copy.deepcopy(self.spec)
        s["lighting"]["presets"] = {}
        with self.assertRaises(ValueError):
            IR.spec_lighting(s)

    def test_forward_defaults_to_the_aim_direction(self):
        s = copy.deepcopy(self.spec)
        for p in s["lighting"]["presets"].values():
            for lt in p["lights"]:
                del lt["forward"]
        a, b = IR.spec_lighting(s)["lights"], IR.spec_lighting(self.spec)["lights"]
        for k in b:
            np.testing.assert_allclose(a[k]["forward"], b[k]["forward"], atol=1e-8)


class TestRenderSettings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex = _TexDir()
        cls.spec = tabletop_spec(tex_sha=cls.tex.sha)

    @classmethod
    def tearDownClass(cls):
        cls.tex.close()

    def settings(self, spec=None, **over):
        arr = tabletop_arrangement(spec or self.spec)
        kw = dict(renderer="RaytracedLighting", rt_subframes=16, settle=200, capture_updates=100, lightings=["L1", "L2"],
                  assets_dir="/nonexistent", tags_dir="/nonexistent", isaacsim="x", textures_dir=self.tex.dir)
        kw.update(over)
        return B.render_settings(arr, **kw)

    def test_the_spec_presets_texture_and_materials_are_recorded(self):
        s = self.settings()
        self.assertEqual(s["lighting_source"], "scene spec")
        self.assertEqual(s["lighting_aim"], [0.0, 0.0, 0.05])
        self.assertEqual(set(s["lighting_presets"]), {"L1", "L2"})
        self.assertEqual(s["lighting_presets"]["L2"][1][1][4], False)                  # key_right off in L2
        top = s["spec_materials"]["table_top"]
        self.assertEqual((top["roughness"], top["metallic"]), (0.5, 0.0))
        self.assertEqual((top["texture"]["sha256"], top["texture"]["file"], top["texture"]["wrap"]),
                         (self.tex.sha, f"{TEX_NAME}.png", "clamp"))
        self.assertEqual(set(s["spec_materials"]), {"table_top", "table_edge", "table_leg", "floor", "wall", "baseboard"})
        self.assertFalse(s["tags_enabled"])
        json.dumps(s)

    def test_a_requested_lighting_subset_records_only_those_presets(self):
        self.assertEqual(set(self.settings(lightings=["L1"])["lighting_presets"]), {"L1"})

    def test_anything_that_changes_pixels_changes_the_settings(self):
        base = B.sha256_json(self.settings())
        self.assertEqual(base, B.sha256_json(self.settings()))
        moved = copy.deepcopy(self.spec)
        for p in moved["lighting"]["presets"].values():
            p["lights"][0]["position"][0] += 0.001
            p["lights"][0]["forward"] = None
        self.assertNotEqual(base, B.sha256_json(self.settings(moved)))
        aim = copy.deepcopy(self.spec)
        for p in aim["lighting"]["presets"].values():
            for lt in p["lights"]:
                lt["forward"] = None
        aim["lighting"]["aim"][2] = 0.10
        self.assertNotEqual(base, B.sha256_json(self.settings(aim)))
        rough = copy.deepcopy(self.spec)
        rough["materials"]["table_top"]["roughness"] = 0.55
        self.assertNotEqual(base, B.sha256_json(self.settings(rough)))
        wall = copy.deepcopy(self.spec)
        wall["materials"]["wall"]["rgba"][0] = 0.80
        self.assertNotEqual(base, B.sha256_json(self.settings(wall)))
        other = _TexDir()
        self.addCleanup(other.close)
        (other.dir / f"{TEX_NAME}.png").write_bytes(b"another texture")
        spec2 = tabletop_spec(tex_sha=_sha(b"another texture"))
        self.assertNotEqual(base, B.sha256_json(self.settings(spec2, textures_dir=other.dir)))
        with self.assertRaises(IR.TextureError):                                      # the wrong file for the spec's sha256
            self.settings(self.spec, textures_dir=other.dir)

    def test_check_render_inputs_refuses_before_isaac(self):
        arr = tabletop_arrangement(self.spec)
        B.check_render_inputs([(arr, b"")], ["L1", "L2"], self.tex.dir)
        with self.assertRaisesRegex(B.RefusedArrangement, "TD5000.*sha256"):
            bad = copy.deepcopy(arr)
            bad["scene"]["materials"]["table_top"]["texture"]["file_sha256"] = "f" * 64
            B.check_render_inputs([(bad, b"")], ["L1"], self.tex.dir)
        with self.assertRaisesRegex(B.RefusedArrangement, "not a preset"):
            B.check_render_inputs([(arr, b"")], ["L3"], self.tex.dir)
        with self.assertRaisesRegex(B.RefusedArrangement, "TD5000"):
            bad = copy.deepcopy(arr)
            bad["scene"]["lighting"]["presets"]["L1"]["lights"][0]["type"] = "sphere"
            B.check_render_inputs([(bad, b"")], ["L1"], self.tex.dir)
        # a shelf arrangement needs no texture and any of the shelf presets
        B.check_render_inputs([(FI.arrangement(), b"")], ["L1", "L2"], "/nonexistent")


class TestMainWithFakeIsaac(unittest.TestCase):
    """B.main on a tabletop arrangement against the stub Isaac: the texture is verified before the app starts, and the
    settings in render.json are the tabletop's."""

    def setUp(self):
        self.tex = _TexDir()
        self.addCleanup(self.tex.close)
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.work = Path(self.work.name)
        (self.work / "assets").mkdir()

    def run_main(self, spec, textures_dir):
        arr = tabletop_arrangement(spec)
        f = self.work / "TD5000.json"
        f.write_text(json.dumps(arr) + "\n")
        FI.reset(1)
        started = []
        with contextlib.ExitStack() as st:
            for p in FI.patches(B):
                st.enter_context(p)
            sim_app = sys.modules["isaacsim"].SimulationApp
            st.enter_context(mock.patch.object(sys.modules["isaacsim"], "SimulationApp",
                                               lambda cfg: (started.append(cfg), sim_app(cfg))[1]))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    B.main(["--arrangements", str(f), "--assets-dir", str(self.work / "assets"), "--tags-dir", "tags",
                            "--textures-dir", str(textures_dir), "--out", str(self.work / "out")])
                    err = None
                except B.RefusedArrangement as e:
                    err = e
        return err, started

    def test_a_wrong_texture_never_starts_isaac(self):
        spec = tabletop_spec(tex_sha="0" * 64)
        err, started = self.run_main(spec, self.tex.dir)
        self.assertIsInstance(err, B.RefusedArrangement)
        self.assertIn("refusing to render", str(err))
        self.assertEqual(started, [])
        self.assertFalse((self.work / "out" / "TD5000").exists())

    def test_a_good_texture_renders_and_records_the_tabletop_settings(self):
        err, started = self.run_main(tabletop_spec(tex_sha=self.tex.sha), self.tex.dir)
        self.assertIsNone(err)
        self.assertEqual(len(started), 1)
        d = self.work / "out" / "TD5000"
        rec = json.loads((d / "render.json").read_text())
        self.assertEqual(rec["settings"]["spec_materials"]["table_top"]["texture"]["sha256"], self.tex.sha)
        self.assertEqual(rec["settings"]["lighting_source"], "scene spec")
        self.assertEqual(rec["textures_dir"], str(self.tex.dir))
        self.assertIn("lighting_presets_note", rec)
        lab = json.loads((d / "labels.json").read_text())
        self.assertEqual(lab["hashes"]["render_settings_sha256"], B.sha256_json(rec["settings"]))
        self.assertEqual(lab["lightings"], ["L1", "L2"])


class TestShelfPathStaysTheShelf(unittest.TestCase):
    def test_shelf_render_settings_have_none_of_the_tabletop_keys(self):
        arr = FI.arrangement()
        s = B.render_settings(arr, renderer="RaytracedLighting", rt_subframes=16, settle=200, capture_updates=100,
                              lightings=["L1", "L2"], assets_dir="/nonexistent", tags_dir="/nonexistent", isaacsim="x")
        self.assertEqual(set(s) & {"lighting_aim", "lighting_source", "lighting_lights", "spec_materials"}, set())
        self.assertEqual(s["lighting_presets"], {k: IR.LIGHTING[k] for k in ("L1", "L2")})

    def test_shelf_lights_keep_the_hard_coded_aim_and_no_presets_key(self):
        ctx = G.generator_context()
        arr = A.design_arrangement(0, ctx=ctx)
        log, built = recorded_stage(arr, "/nonexistent")
        self.assertNotIn("lighting_presets", built)
        m = Log(log).light_matrix("/World/Lights/key_left")
        pos = np.array(IR.LIGHTING["L1"][1][0][1])
        want = np.array(IR.SHELF_AIM) - pos
        np.testing.assert_allclose(-m[2, :3], want / np.linalg.norm(want), atol=1e-9)
        np.testing.assert_allclose(m[3, :3], pos, atol=1e-9)

    def test_a_shelf_mesh_target_is_not_a_static_mesh(self):
        arr = A.design_arrangement(0, ctx=G.generator_context())
        self.assertTrue(arr["scene"]["meshes"])
        self.assertFalse(any(IR.is_static_mesh(m) for m in arr["scene"]["meshes"]))
        log, _ = recorded_stage(arr, "/nonexistent")
        self.assertFalse([p for p in Log(log).paths("UsdGeom.Mesh") if p.startswith("/World/Shelf/")])


class TestRigBuildStage(unittest.TestCase):
    """sim/isaac_rig.py's own build_stage (the single-scene renderer)."""

    @staticmethod
    def rig_log(spec, textures_dir="/nonexistent", lighting="L1") -> tuple[list, dict]:
        log: list = []
        mods = {"pxr": G._Rec(log, "pxr"), "omni": G._Rec(log, "omni"), "omni.usd": G._Rec(log, "omni.usd"),
                "carb": G._Rec(log, "carb")}
        with mock.patch.dict(sys.modules, mods):
            built = IR.build_stage(json.loads(json.dumps(spec)), "/tags", lighting, None, [], textures_dir)
            IR.set_lighting(G._Rec(log, "stage"), built, "L2")
            log.append(["built", G._canon(built), {}])
        return log, built

    def test_shelf_rig_call_log_is_pinned(self):
        """Recorded from sim/isaac_rig.py before the tabletop edits (DES0's scene, the mug meshes as targets)."""
        spec = A.design_arrangement(0, ctx=G.generator_context())["scene"]
        log, built = self.rig_log(spec)
        self.assertNotIn("lighting_presets", built)
        self.assertEqual({"n_calls": len(log), "sha256": G.sha256_json(log)}, RIG_SHELF_STAGE_LOG)

    def test_tabletop_in_the_rig(self):
        tex = _TexDir()
        self.addCleanup(tex.close)
        spec = tabletop_spec(tex_sha=tex.sha)
        entries, built = self.rig_log(spec, tex.dir)
        log = Log(entries)
        self.assertEqual(log.bound_material("/World/table_table_top"), "/World/Looks/spec_table_top")
        self.assertEqual(log.inputs("/World/Looks/spec_table_top/tex")["sourceColorSpace"], ("set", "sRGB"))
        self.assertIn("/World/room_wall_left", log.paths("UsdGeom.Cube"))
        self.assertEqual(built["targets"], [])
        for lt in spec["lighting"]["presets"]["L1"]["lights"]:
            m = log.light_matrix(f"/World/Lights/{lt['name']}")
            np.testing.assert_allclose(m[3, :3], lt["position"], atol=1e-9)
            np.testing.assert_allclose(-m[2, :3], lt["forward"], atol=1e-9)
        with self.assertRaises(IR.TextureError):
            self.rig_log(spec, "/nonexistent")


if __name__ == "__main__":
    unittest.main()
