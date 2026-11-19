"""Build a collidable workshop with an exposed parking bay for the sliding door."""

import argparse
import json
from pathlib import Path

import numpy as np

from development.corridor_transfer.scene import box
from development.workshop_realism.release import read, require, sha
from development.workshop_realism.scene import generate as furnished, quad, wall
from sim.arrange import arrangement_hash

PARKING_WALL = 1.68
SILL_GAP = 0.014
TRAVEL = 0.98


def modify(arr):
    scene = arr["scene"]
    half = arr["design"]["corridor_width_m"] / 2
    depth = arr["design"]["door_depth_m"]
    door = arr["objects"]["occluder"]
    park_sign = int(np.sign(door["poses"]["parked"]["base"][0]))
    side = "left" if park_sign < 0 else "right"
    lo, hi = (-PARKING_WALL, half) if park_sign < 0 else (-half, PARKING_WALL)
    for pose in door["poses"].values():
        pose["base"][2] = SILL_GAP
    boxes = scene["boxes"]
    meshes = scene["meshes"]
    wing = "door_return_" + side
    old_trim = {f"side_base_{park_sign}", f"side_chair_rail_{park_sign}"}
    old_trim.update(f"conduit_{park_sign}_{i}" for i in range(2))
    old_trim.update(f"pipe_band_{park_sign}_{i}" for i in range(3))
    replaced = old_trim | {
        side + "_wall",
        wing,
        "door_track",
        "access_control",
        "reader_lamp",
    }
    boxes[:] = [part for part in boxes if part["name"] not in replaced]
    meshes[:] = [
        m
        for m in meshes
        if not (
            m["name"] == "sealed_concrete"
            or m["name"].startswith(side + "_masonry_")
            or m["name"].startswith("return_masonry_" + str(park_sign) + "_")
        )
    ]

    def add(name, center, size, material):
        boxes.append(box(name, center, size, material, "facility"))

    end = depth + 0.19
    for b in boxes:
        name = b["name"]
        if (
            name.startswith(("door_post_", "door_return_", "jamb_shoe_"))
            or name == "door_header"
        ):
            b["center"][1] += 0.105
        if name in ("floor", "ceiling"):
            b["center"][0] = (lo + hi) / 2
            b["half"][0] = (hi - lo) / 2 + 0.10
    for m in meshes:
        if m["name"].startswith("return_masonry_") or m["name"] == "doorway_id_plate":
            for v in m["vertices"]:
                v[1] += 0.105
    # The widened entry ends behind the stationary doorway and rejoins the original rear corridor.
    front = -2.75
    back = 2.16
    add(
        "parking_outer_wall",
        [park_sign * (PARKING_WALL + 0.05), (front + end) / 2, 1.32],
        [0.10, end - front, 2.64],
        "wall",
    )
    add(
        "parking_rear_side",
        [park_sign * (half + 0.05), (end + back) / 2, 1.32],
        [0.10, back - end, 2.64],
        "wall",
    )
    add(
        "parking_backing_wall",
        [park_sign * (PARKING_WALL + 0.53) / 2, depth + 0.125, 1.045],
        [PARKING_WALL - 0.53, 0.08, 2.09],
        "wall",
    )
    add(
        "parking_upper_infill",
        [park_sign * (PARKING_WALL + half) / 2, depth + 0.125, 2.365],
        [PARKING_WALL - half, 0.08, 0.55],
        "wall",
    )
    wall(
        "parking_lining",
        [park_sign * (PARKING_WALL - 0.002), front if park_sign < 0 else end, 0],
        [0, -park_sign * (end - front), 0],
        2.64,
        meshes,
    )
    wall(
        "rear_side_lining",
        [park_sign * (half - 0.002), end if park_sign < 0 else back, 0],
        [0, -park_sign * (back - end), 0],
        2.64,
        meshes,
    )
    wall(
        "parking_backing",
        [lo if park_sign < 0 else 0.53, depth + 0.082, 0],
        [PARKING_WALL - 0.53, 0, 0],
        2.09,
        meshes,
    )
    for label, z, sz, material in [
        ("skirting", 0.065, 0.126, "edge"),
        ("rub_rail", 0.83, 0.036, "galvanized"),
    ]:
        add(
            "outer_" + label,
            [park_sign * (PARKING_WALL - 0.015), (front + end) / 2, z],
            [0.026, end - front - 0.01, sz],
            material,
        )
    meshes.append(
        quad(
            "concrete_floor",
            [lo, -2.74, 0.0005],
            [hi - lo, 0, 0],
            [0, 4.85, 0],
            "concrete",
            ((hi - lo) / 1.9, 4.85 / 1.9),
        )
    )
    # A full, visible top rail spans only the actual 98 cm travel and its supporting wall.
    rail_x = park_sign * TRAVEL / 2
    rail_length = TRAVEL + 0.95 + 0.105
    cy = door["poses"]["placed"]["base"][1]
    top = SILL_GAP + door["size_m"][2]
    add(
        "rail_cover",
        [rail_x, cy, top + 0.032],
        [rail_length, 0.135, 0.04],
        "galvanized",
    )
    for f in (-1, 1):
        add(
            "guide_flange_" + str(f),
            [rail_x, cy + f * 0.061, top + 0.002],
            [rail_length, 0.012, 0.046],
            "iron",
        )
    for i, frac in enumerate([-0.42, 0, 0.42]):
        add(
            "rail_support_" + str(i),
            [rail_x + frac * rail_length, cy + 0.099, top + 0.083],
            [0.044, 0.244, 0.047],
            "galvanized",
        )
    reader_x = -park_sign * (0.53 + half) / 2
    add(
        "access_control_clear_side",
        [reader_x, depth + 0.037, 0.89],
        [0.103, 0.061, 0.157],
        "iron",
    )
    add(
        "access_lamp_clear_side",
        [reader_x, depth + 0.004, 0.921],
        [0.049, 0.004, 0.013],
        "lamp",
    )
    arr["study_id"] = "workshop-clearance-v2"
    arr["environment"] = "service_workshop_bay"
    arr["design"]["door_mechanism"] = {
        "axis": "x",
        "travel_m": TRAVEL,
        "parking_side": park_sign,
        "parking_wall_abs_x_m": PARKING_WALL,
        "floor_clearance_m": SILL_GAP,
        "nominal_joint_range": [0, TRAVEL],
    }
    arr["arrangement_sha256"] = arrangement_hash(arr)
    return arr


def sweep(arr):
    door = arr["objects"]["occluder"]
    p = np.asarray(door["poses"]["parked"]["base"])
    c = np.asarray(door["poses"]["placed"]["base"])
    w, d, h = door["size_m"]
    span = np.array(
        [np.minimum(p, c) + [-w / 2, -d / 2, 0], np.maximum(p, c) + [w / 2, d / 2, h]]
    )
    # A translating axis-aligned box traverses this entire box, with no gaps between its endpoint envelopes.
    solids = [
        (
            b["name"],
            np.array(b["center"]) - b["half"],
            np.array(b["center"]) + b["half"],
        )
        for b in arr["scene"]["boxes"]
    ]
    for m in arr["scene"]["meshes"]:
        v = np.asarray(m["vertices"])
        solids.append((m["name"], v.min(axis=0), v.max(axis=0)))
    gaps = {}
    hits = []
    for name, lo, hi in solids:
        delta = np.maximum(lo - span[1], span[0] - hi)
        if np.max(delta) <= 1e-7:
            hits.append(name)
        gaps[name] = float(np.linalg.norm(np.maximum(delta, 0)))
    return {
        "sweep_world_minmax_m": span.tolist(),
        "hits": hits,
        "minimum_static_clearance_m": min(gaps.values()),
        "nearest_surface": min(gaps, key=gaps.get),
        "static_shapes_checked": len(gaps),
        "gap_to_parking_wall_m": gaps.get("parking_outer_wall"),
        "door_open_edge_to_center_m": abs(float(p[0])) - w / 2,
    }


def generate(output, fixture=False):
    furnished(output, fixture)
    manifest = read(output / "manifest.json")
    manifest["study_id"] = "workshop-clearance-v2"
    for layout in manifest["arrangements"]:
        layout["environment"] = "service_workshop_bay"
        for item in layout["levels"].values():
            path = output / item["path"]
            arr = modify(read(path))
            proof = sweep(arr)
            require(
                not proof["hits"], f"Door sweep intersects {proof['hits']} in {path}"
            )
            path.write_text(json.dumps(arr, indent=2) + "\n")
            item["sha256"] = sha(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    generate(args.output, args.fixture)
