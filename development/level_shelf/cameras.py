"""Derive matched shelf camera poses from the released native arrangement files."""

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from sim.arrange import arrangement_hash, sha256_file
from sim.rig import Camera
from sim.tabletop import camera_spec


HERE = Path(__file__).resolve().parent
POSITIONS = HERE.parents[1] / "configs/sim/positions_shelf_A.json"
SLOTS = {"P1": ("P4", "original"), "P2": ("P4", "leveled"), "P3": ("P5", "frontal"),
         "P4": ("P5", "original"), "P5": ("P5", "leveled")}


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2) + "\n")


def floor_edge(scene):
    board = next(b for b in scene["boxes"] if b["name"] == "c3_floor_board")
    if board["quat_wxyz"] != [1, 0, 0, 0]:
        raise ValueError("Expected a physically level shelf")
    cx, cy, cz = board["center"]
    hx, hy, hz = board["half"]
    return [[cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz]]


def image_angle(camera, edge):
    dx, dy = np.diff(camera.project(edge), axis=0)[0]
    return math.degrees(math.atan2(dy, dx))


def make_camera(spec, slot, mode, edge, lens_x, center_y):
    camera = Camera(slot, tuple(spec["position"]), spec["heading_deg"], spec["pitch_down_deg"],
                    spec["roll_deg"], spec["f_px"], spec["width"], spec["height"], spec["k1_fit_not_rendered"])
    if mode == "leveled":
        camera = replace(camera, roll_deg=camera.roll_deg + image_angle(camera, edge))
    if mode == "frontal":
        camera = replace(camera, position=(lens_x, center_y, camera.position[2]),
                         heading_deg=0.0, pitch_down_deg=0.0, roll_deg=0.0)
    result = camera_spec(camera) if mode != "original" else dict(spec, name=slot)
    angle = image_angle(camera, edge)
    if mode != "original" and abs(angle) > 1e-7:
        raise ValueError("Camera did not level the projected front edge")
    return result, angle


def build(source, output):
    declaration = json.loads((HERE / "declaration.json").read_text())
    if sha256_file(source) != declaration["source_manifest_sha256"]:
        raise ValueError("Source manifest differs from the paper release")
    positions = json.loads(POSITIONS.read_text())
    center_y = (positions["arc_centre_behind_front_cm"] - positions["radius_cm"]) / 100
    original = json.loads(source.read_text())
    layouts = [a for a in original["arrangements"] if a["environment"] == "shelf"]
    if [a["base_id"] for a in layouts] != declaration["source_layouts"]:
        raise ValueError("Unexpected shelf layout panel")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(original, study_id=declaration["study_id"], arrangements=[])
    projected = []
    for layout in layouts:
        base = layout["base_id"]
        new_layout = dict(layout, levels={})
        for level, info in layout["levels"].items():
            path = source.parent / info["path"]
            if sha256_file(path) != info["sha256"]:
                raise ValueError(f"Source arrangement hash differs: {path}")
            source_arr = json.loads(path.read_text())
            rec = copy.deepcopy(source_arr)
            rec["study_id"] = declaration["study_id"]
            scene = rec["scene"]
            edge = floor_edge(scene)
            sign = source_arr["revisit"]["stops"]["P5"]["sign_translation"]
            dx = 0.06 * sign if level == "D6" else 0.0
            poses = {slot: {} for slot in SLOTS}
            for cameras, label in (("cameras", "nominal"), ("revisit_cameras", "revisit")):
                by_name = {camera["name"]: camera for camera in scene[cameras]}
                new_cameras = []
                for slot, (old, mode) in SLOTS.items():
                    camera, angle = make_camera(by_name[old], slot, mode, edge,
                                                dx if label == "revisit" else 0.0, center_y)
                    new_cameras.append(camera)
                    poses[slot][label] = {key: camera[key] for key in
                                          ("position", "heading_deg", "pitch_down_deg", "roll_deg")}
                    projected.append({"layout": base, "level": level, "slot": slot, "camera": label,
                                      "source_stop": old, "treatment": mode, "front_edge_deg": angle})
                scene[cameras] = new_cameras
            for slot, (old, mode) in SLOTS.items():
                poses[slot].update(source_stop=old, treatment=mode)
            rec["revisit"] = {"schema": "cmpe249.level-shelf-revisit.v1", "level": level, "stops": poses}
            rec["analytic"] = {"note": "Use captured instance masks for target visibility in the new camera views."}
            rec.pop("predeclared_indistinguishable", None)
            scene["provenance"]["cameras"]["poses"] = "Camera treatments from the level-shelf follow-up."
            rec["source_arrangement_file_sha256"] = info["sha256"]
            rec["arrangement_sha256"] = arrangement_hash(rec)
            write_json(output / path.name, rec)
            new_layout["levels"][level] = {"path": path.name, "sha256": sha256_file(output / path.name)}
        manifest["arrangements"].append(new_layout)
    write_json(output / "manifest.json", manifest)
    write_json(output / "edge-angles.json", projected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
