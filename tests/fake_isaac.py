"""A stub of Isaac Sim, omni.replicator, omni.usd, pxr and PIL, so that sim/isaac_batch.py's render_arrangement and main
run end to end on any machine (numpy and cv2 only). Used by tests/test_sim_batch.py; not a test module itself.

The fake scene (16 x 12 pixel frames, five stops):
  - the target is a 4 x 4 square at rows 4-7, cols 4-7 for the NOMINAL cameras ("/World/Cameras/<stop>") and shifted one
    pixel to the right, cols 5-8, for the REVISIT cameras ("/World/CamerasRevisit/<stop>"), when it is visible;
  - the occluder box covers cols 4-5 of rows 4-7 whenever it is 'placed' (the same image columns for both camera sets,
    which is not physical but makes labels taken at the wrong pose show up as different numbers), and sits at rows 0-1,
    cols 12-13 when 'parked';
  - the shelf floor is row 11.
RGB value = 100 (L1) / 200 (L2) + RUN (the marker of the run that wrote the file) + 10 for a revisit camera, so every PNG
records the lighting and camera set that were active at capture time and which run wrote it.

STATE records every render product / annotator creation (nothing may be reused across captures) and can inject a crash at
the N-th orchestrator step.
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import numpy as np

W, H = 16, 12
STOPS = ["P1", "P2", "P3", "P4", "P5"]
STATE: dict = {}
POSES: dict = {}
PNG_TEXT: dict = {}


def reset(run: int = 0, crash_at: int | None = None) -> None:
    STATE.clear()
    STATE.update({"run": run, "lighting": None, "visible": {}, "occluder_pose": None, "steps": 0, "crash_at": crash_at,
                  "rp_created": 0, "ann_created": 0, "captures": [], "destroyed": set(), "detached": set()})
    POSES.clear()
    PNG_TEXT.clear()


class FakeRP:
    def __init__(self, path, res):
        STATE["rp_created"] += 1
        self.id = STATE["rp_created"]
        self.path, self.res = path, res

    def destroy(self):
        STATE["destroyed"].add(self.id)


class FakeAnn:
    def __init__(self, name, init_params=None):
        STATE["ann_created"] += 1
        self.id = STATE["ann_created"]
        self.name = name
        self.rp = None

    def attach(self, rp):
        self.rp = rp

    def detach(self):
        STATE["detached"].add(self.id)
        self.rp = None

    @staticmethod
    def _vis(path):
        v = STATE["visible"]
        parts = path.split("/")
        return all(v.get("/".join(parts[:i]), True) for i in range(2, len(parts) + 1))

    def get_data(self):
        assert self.rp is not None and self.rp.id not in STATE["destroyed"], "annotator read after its render product was destroyed"
        revisit = "/CamerasRevisit/" in self.rp.path
        STATE["captures"].append({"step": STATE["steps"], "lighting": STATE["lighting"], "kind": self.name, "rp": self.rp.id,
                                  "ann": self.id, "path": self.rp.path, "revisit": revisit})
        if self.name == "rgb":
            v = (100 if STATE["lighting"] == "L1" else 200) + STATE["run"] + (10 if revisit else 0)
            return np.full((H, W, 4), v, np.uint8)
        ids = np.zeros((H, W), np.uint32)
        lab = {}
        if self._vis("/World/Shelf/shelf_floor"):
            ids[11, :] = 3
            lab["3"] = "/World/Shelf/shelf_floor"
        if self._vis("/World/Objects/target/mesh"):
            c0 = 5 if revisit else 4
            ids[4:8, c0:c0 + 4] = 1
            lab["1"] = "/World/Objects/target/mesh"
        if self._vis("/World/Objects/occluder/box"):
            if STATE["occluder_pose"] == "placed":
                ids[4:8, 4:6] = 2
            else:
                ids[0:2, 12:14] = 2
            lab["2"] = "/World/Objects/occluder/box"
        return {"data": ids, "info": {"idToLabels": lab}}


def _step(rt_subframes=1):
    STATE["steps"] += 1
    if STATE["crash_at"] is not None and STATE["steps"] >= STATE["crash_at"]:
        raise RuntimeError(f"injected Isaac crash at orchestrator.step #{STATE['steps']}")


class FakeOp:
    def Set(self, m):
        STATE["occluder_pose"] = next(k for k, p in POSES.items() if p == tuple(m))


class FakeApp:
    def __init__(self, cfg):
        pass

    def update(self):
        pass

    def close(self):
        pass


class _FakeImage:
    def __init__(self, a):
        self.a = np.asarray(a)

    def save(self, path, pnginfo=None, compress_level=3):
        import cv2

        cv2.imwrite(str(path), self.a if self.a.ndim == 2 else self.a[..., ::-1])
        PNG_TEXT[str(path)] = dict(getattr(pnginfo, "text", {}))


class _FakePngInfo:
    def __init__(self):
        self.text = {}

    def add_text(self, k, v):
        self.text[k] = v


def modules() -> dict:
    rep = types.ModuleType("omni.replicator.core")
    rep.create = types.SimpleNamespace(render_product=lambda path, res: FakeRP(path, res))
    rep.AnnotatorRegistry = types.SimpleNamespace(get_annotator=lambda name, init_params=None: FakeAnn(name, init_params))
    rep.orchestrator = types.SimpleNamespace(step=_step)
    omni = types.ModuleType("omni")
    omni_rep = types.ModuleType("omni.replicator")
    omni_rep.core = rep
    usd = types.ModuleType("omni.usd")
    usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: "STAGE")
    omni.replicator, omni.usd = omni_rep, usd
    pxr = types.ModuleType("pxr")
    pxr.Gf = types.SimpleNamespace(Matrix4d=lambda *a: tuple(a))
    isaacsim = types.ModuleType("isaacsim")
    isaacsim.SimulationApp = FakeApp
    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.fromarray = _FakeImage
    pil_png = types.ModuleType("PIL.PngImagePlugin")
    pil_png.PngInfo = _FakePngInfo
    pil.Image, pil.PngImagePlugin = pil_image, pil_png
    return {"omni": omni, "omni.replicator": omni_rep, "omni.replicator.core": rep, "omni.usd": usd, "pxr": pxr,
            "isaacsim": isaacsim, "PIL": pil, "PIL.Image": pil_image, "PIL.PngImagePlugin": pil_png}


def patches(B):
    """Context managers replacing the USD-building parts of isaac_batch with the fake scene (and sys.modules entries)."""

    def build_stage(arr, assets_dir, tags_dir, ply_cache):
        poses = {k: B.pose_matrix(p["base"], p["yaw_deg"]) for k, p in arr["objects"]["occluder"]["poses"].items()}
        POSES.clear()
        POSES.update({k: tuple(m.reshape(-1).tolist()) for k, m in poses.items()})
        STATE["visible"].clear()
        STATE["occluder_pose"] = "parked"
        return {"cameras": {c["name"]: f"/World/Cameras/{c['name']}" for c in arr["scene"]["cameras"]},
                "cameras_revisit": {c["name"]: f"/World/CamerasRevisit/{c['name']}" for c in arr["scene"]["revisit_cameras"]},
                "lights": {}, "shelf": "/World/Shelf", "objects": "/World/Objects", "distractors": [],
                "target": "/World/Objects/target", "occluder": "/World/Objects/occluder", "occluder_op": FakeOp(),
                "occluder_poses": poses}

    def visible(stage, path, on):
        STATE["visible"][path] = bool(on)

    def set_lighting(stage, built, preset):
        STATE["lighting"] = preset

    return [mock.patch.dict(sys.modules, modules()), mock.patch.object(B, "build_stage", build_stage),
            mock.patch.object(B, "_visible", visible), mock.patch.object(B.ir, "set_lighting", set_lighting),
            mock.patch.object(B.ir, "_tonemap", lambda: None)]


def arrangement(arr_id="SD1003", seed=1003, target="tomato_soup_can", parked_x=0.2, family="white_box", m16_side=None) -> dict:
    """A minimal arrangement in the generator's schema (the fake renderer needs only these fields)."""
    nominal = [{"name": s, "width": W, "height": H} for s in STOPS]
    revisit = [{"name": s, "width": W, "height": H} for s in STOPS]
    return {"schema": "cmpe249.sim-arrangement.v1", "arrangement_id": arr_id, "seed": seed, "split": "dev", "kind": "random",
            "target_id": target, "occluder_family": family, "m16_side": m16_side, "design_case": False,
            "predeclared_indistinguishable": False,
            "objects": {"target": {"kind": "ycbv"},
                        "occluder": {"family": family, "kind": "box", "label": family,
                                     "poses": {"placed": {"base": [0.0, 0.05, 0.0], "yaw_deg": 0.0},
                                               "parked": {"base": [parked_x, 0.3, 0.0], "yaw_deg": 0.0}}},
                        "distractors": []},
            "phases": [{"phase": ph, "target_present": ph != "removed", "camera": "nominal" if ph == "reference" else "revisit",
                        "occluder_pose": "parked" if ph in ("reference", "visible") else "placed"}
                       for ph in ("reference", "visible", "occluded", "removed")],
            "stops": STOPS, "lightings": ["L1", "L2"],
            "provenance": {"generator": {"sha256": "g" * 64}, "config": {"sha256": "c" * 64},
                           "positions": {"path": "configs/sim/positions_shelf_A.json", "sha256": "p" * 64}},
            "revisit": {"schema": "cmpe249.sim-revisit.v0", "ranges": {"translation_m": 0.01, "rotation_deg": 0.5},
                        "stops": {s: {"delta": {"dx_m": 0.001 * i, "dy_m": 0.0, "dz_m": 0.0, "dyaw_deg": 0.1, "dpitch_deg": 0.0,
                                                "droll_deg": 0.0},
                                      "attempt": i % 2, "n_rejected": i % 2, "margin_px": {"min": 10.0 + i},
                                      "nominal_margin_px": {"min": 14.0}} for i, s in enumerate(STOPS)}},
            "arrangement_sha256": "a" * 64, "scene": {"cameras": nominal, "revisit_cameras": revisit}}
