"""Dress the eight matched corridors and add two fixed, target-independent robot cameras."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from development.corridor_transfer.scene import (
    box,
    build as plain_corridor,
    camera,
    camera_record,
    object_record,
)
from sim.arrange import arrangement_hash

HERE = Path(__file__).resolve().parent
STOPS = ["P1", "P2", "P3", "P4", "P5"]
STATES = ["clear_present", "clear_removed", "placed_present", "placed_removed"]


def quad(name, origin, across, up, material, repeat=(1, 1)):
    p, u, v = (np.array(x, dtype=float) for x in (origin, across, up))
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    return {
        "name": name,
        "role": "cladding",
        "material": material,
        "vertices": [x.tolist() for x in (p, p + u, p + u + v, p + v)],
        "faces": [[0, 1, 2], [0, 2, 3]],
        "normals": [n.tolist()] * 4,
        "uv": [[0, 0], [repeat[0], 0], list(repeat), [0, repeat[1]]],
    }


def wall(name, origin, across, height, meshes):
    p, u = np.asarray(origin), np.asarray(across)
    length = float(np.linalg.norm(u))
    for tier, base, h, mat in [
        ("lower", 0, 0.82, "lower_brick"),
        ("upper", 0.82, height - 0.82, "upper_brick"),
    ]:
        meshes.append(
            quad(
                name + "_" + tier,
                p + [0, 0, base],
                u,
                [0, 0, h],
                mat,
                (length / 1.65, h / 1.65),
            )
        )


def materials():
    maps = json.loads((HERE / "materials/sources.json").read_text())["maps"]
    out = {
        name: {
            "roughness": 0.8,
            "texture": {
                "name": name,
                "file": name + ".png",
                "sha256": digest,
                "wrap": "clamp" if name.endswith(("plate", "face")) else "repeat",
                "colour_space": "sRGB",
            },
        }
        for name, digest in maps.items()
    }
    for name, rgb, rough, metal in [
        ("iron", [0.14, 0.17, 0.17], 0.55, 0.4),
        ("galvanized", [0.47, 0.50, 0.50], 0.44, 0.58),
        ("stainless", [0.55, 0.59, 0.60], 0.29, 0.68),
        ("ochre", [0.63, 0.30, 0.09], 0.53, 0.18),
        ("edge", [0.15, 0.22, 0.20], 0.73, 0.08),
        ("cabinet_red", [0.46, 0.12, 0.09], 0.53, 0.18),
        ("carton", [0.48, 0.34, 0.19], 0.88, 0),
        ("lamp", [0.92, 0.91, 0.76], 0.6, 0),
        ("black_rubber", [0.075, 0.08, 0.074], 0.84, 0),
        ("gray_enamel", [0.51, 0.54, 0.49], 0.51, 0.21),
    ]:
        out[name] = {"rgba": [*rgb, 1], "roughness": rough, "metallic": metal}
    return out


def arrange(source):
    arr = copy.deepcopy(source)
    scene = arr["scene"]
    idx = 8 if arr["base_id"] == "CORFIX" else int(arr["base_id"][-2:]) - 1
    sign = arr["design"]["target_side"]
    width = arr["design"]["corridor_width_m"]
    depth = arr["design"]["door_depth_m"]
    half = width / 2
    arr["study_id"] = "workshop-realism-v1"
    arr["environment"] = "service_workshop"
    arr["stops"] = STOPS
    obj = arr["objects"]
    door = obj["occluder"]
    door.update(size_m=[0.95, 0.085, 2.04], material="iron")
    scene["materials"].update(materials())
    boxes = scene["boxes"]
    meshes = scene["meshes"]
    for b in boxes:
        name = b["name"]
        if name in ("left_wall", "right_wall"):
            b["center"][1:] = [-0.29, 1.32]
            b["half"][1:] = [2.46, 1.32]
        elif name == "back_wall":
            b["center"][2] = 1.32
            b["half"][2] = 1.32
        elif name.startswith(("door_post", "door_return")):
            b["center"][2] = 1.045
            b["half"][2] = 1.045
        elif name == "door_header":
            b["center"][2] = 2.15
        elif name == "door_track":
            b["center"][2] = 2.065
        elif name == "floor":
            b["center"][1] = -0.28
            b["half"][1] = 2.52
        if name.startswith("door_post"):
            b["material"] = "galvanized"
        if name in ("door_track", "door_header"):
            b["material"] = "iron"
    boxes[:] = [b for b in boxes if b["name"] not in ("wall_fixture", "back_panel")]

    def add(n, p, s, m):
        boxes.append(box(n, p, s, m, "workshop"))

    def face(n, x, y, z, w, h, m):
        meshes.append(quad(n, [x, y, z], [w, 0, 0], [0, 0, h], m))

    meshes.append(
        quad(
            "sealed_concrete",
            [-half, -2.74, 0.0005],
            [width, 0, 0],
            [0, 4.84, 0],
            "concrete",
            (width / 1.9, 4.84 / 1.9),
        )
    )
    stripe = depth - 0.33
    meshes.append(
        quad(
            "worn_threshold",
            [-0.5, stripe - 0.0175, 0.00325],
            [1, 0, 0],
            [0, 0.035, 0],
            "floor_paint",
            (1, 0.07),
        )
    )
    wall("left_masonry", [-half + 0.002, -2.74, 0], [0, 4.84, 0], 2.64, meshes)
    wall("right_masonry", [half - 0.002, 2.10, 0], [0, -4.84, 0], 2.64, meshes)
    wall("rear_masonry", [-half, 2.098, 0], [width, 0, 0], 2.64, meshes)
    for side in (-1, 1):
        x0 = -half if side == -1 else 0.53
        wall(
            "return_masonry_" + str(side),
            [x0, depth - 0.023, 0],
            [half - 0.53, 0, 0],
            2.09,
            meshes,
        )
        add(
            "side_base_" + str(side),
            [side * (half - 0.014), -0.31, 0.063],
            [0.026, 4.81, 0.125],
            "edge",
        )
        add(
            "side_chair_rail_" + str(side),
            [side * (half - 0.011), -0.31, 0.832],
            [0.021, 4.81, 0.034],
            "edge",
        )
        for j, z in enumerate([1.68, 1.755]):
            add(
                f"conduit_{side}_{j}",
                [side * (half - 0.025), -0.31, z],
                [0.030, 4.76, 0.030],
                "galvanized",
            )
        for j, y in enumerate([-0.94, 0.32, 1.45]):
            add(
                f"pipe_band_{side}_{j}",
                [side * (half - 0.034), y, 1.715],
                [0.033, 0.035, 0.132],
                "iron",
            )
        add(
            "jamb_shoe_" + str(side),
            [side * 0.484, depth - 0.052, 0.14],
            [0.101, 0.036, 0.279],
            "galvanized",
        )
    add("ceiling", [0, -0.30, 2.675], [width + 0.12, 4.94, 0.07], "gray_enamel")
    for j, y in enumerate([-1.20, 1.10]):
        add(
            f"light_tray_{j}",
            [0.15 * (-1) ** j, y, 2.59],
            [0.21, 0.81, 0.055],
            "galvanized",
        )
        add(
            f"light_lens_{j}", [0.15 * (-1) ** j, y, 2.553], [0.17, 0.74, 0.025], "lamp"
        )
    # Steel rack, timber decks, boxed stock and a rolling tool chest stand behind the target.
    rack_x = -0.29
    ry = 1.77
    for i, x in enumerate([rack_x - 0.41, rack_x + 0.41]):
        for j, y in enumerate([ry - 0.19, ry + 0.19]):
            add(f"rack_upright_{i}_{j}", [x, y, 0.78], [0.042, 0.045, 1.55], "edge")
    for j, z in enumerate([0.185, 0.62, 1.12, 1.53]):
        add(f"rack_deck_{j}", [rack_x, ry, z], [0.85, 0.40, 0.033], "carton")
        add(
            f"rack_beam_{j}",
            [rack_x, ry - 0.218, z - 0.02],
            [0.88, 0.035, 0.042],
            "ochre",
        )
        meshes.append(
            quad(
                f"ply_deck_{j}",
                [rack_x - 0.425, ry - 0.2, z + 0.017],
                [0.85, 0, 0],
                [0, 0.40, 0],
                "ply",
            )
        )
    for j, (x, z, w, h) in enumerate(
        [
            (rack_x - 0.20, 0.204, 0.28, 0.24),
            (rack_x + 0.13, 0.204, 0.31, 0.16),
            (rack_x + 0.03, 0.638, 0.38, 0.28),
            (rack_x - 0.18, 1.138, 0.30, 0.20),
        ]
    ):
        y = ry + 0.008 * (idx % 3)
        d = 0.253
        add(f"packed_carton_{j}", [x, y, z + h / 2], [w, d, h], "carton")
        face(f"carton_label_{j}", x - w / 2, y - d / 2 - 0.001, z, w, h, "carton_face")
    cx = half - 0.269
    cy = 1.43
    for j, x in enumerate([cx - 0.15, cx + 0.15]):
        add(f"caster_{j}", [x, cy, 0.050], [0.066, 0.28, 0.098], "black_rubber")
    add("rolling_chest", [cx, cy, 0.41], [0.42, 0.365, 0.64], "cabinet_red")
    add("chest_lip", [cx, cy, 0.743], [0.445, 0.383, 0.034], "iron")
    for j in range(5):
        z = 0.175 + 0.108 * j
        add(f"drawer_{j}", [cx, cy - 0.187, z], [0.386, 0.018, 0.092], "cabinet_red")
        add(
            f"drawer_pull_{j}",
            [cx, cy - 0.207, z + 0.021],
            [0.32, 0.021, 0.012],
            "stainless",
        )
    obj["distractors"].append(
        object_record(19, [cx, cy, 0.761], 38 + 7 * (idx % 3), "distractor_1")
    )
    x1 = 0.53 + 0.5 * (half - 0.53)
    add("access_control", [x1, depth - 0.052, 0.89], [0.103, 0.061, 0.157], "iron")
    add("reader_lamp", [x1, depth - 0.084, 0.921], [0.049, 0.004, 0.013], "lamp")
    face(
        "doorway_id_plate",
        -0.5 * (half + 0.53) - 0.113,
        depth - 0.024,
        0.99,
        0.226,
        0.094,
        "service_plate",
    )
    add("junction_box", [0.26, 2.035, 1.16], [0.40, 0.12, 0.53], "gray_enamel")
    face("voltage_plate", 0.17, 1.973, 1.17, 0.18, 0.08, "cabinet_plate")
    add("conduit_down", [0.32, 2.045, 1.81], [0.026, 0.034, 0.77], "galvanized")
    # A neutral, UV-mapped front sheet and a grab handle travel with the sliding body.
    arr["door_dressing"] = {
        "skin": quad(
            "moving_skin",
            [-0.473, -0.0327, 0.003],
            [0.946, 0, 0],
            [0, 0, 2.034],
            "door_skin",
            (0.78, 1.42),
        ),
        "badge": quad(
            "moving_badge",
            [-0.165, -0.0331, 0.83],
            [0.33, 0, 0],
            [0, 0, 0.143],
            "door_plate",
        ),
        "boxes": [
            box(
                "grab_upper",
                [sign * 0.335, -0.036, 1.005],
                [0.034, 0.012, 0.032],
                "stainless",
            ),
            box(
                "grab_lower",
                [sign * 0.335, -0.036, 0.822],
                [0.034, 0.012, 0.032],
                "stainless",
            ),
            box(
                "grab_bar",
                [sign * 0.335, -0.040, 0.914],
                [0.025, 0.005, 0.203],
                "stainless",
            ),
            box("bottom_rail", [0, -0.034, 0.058], [0.944, 0.004, 0.111], "galvanized"),
        ],
    }
    for key in ("cameras", "revisit_cameras"):
        for name, height, dist, x, heading, pitch, yaw in [
            ("P4", 0.95, -2.16, -0.31, 7, 13.0, 1.0),
            ("P5", 0.48, -1.64, 0.34, -9, 8.0, 0.0),
        ]:
            a = heading * sign
            pos = np.array([x * sign, dist, height])
            moved = key == "revisit_cameras" and arr["level"] == "D6"
            if moved:
                pos += 0.06 * np.array(
                    [math.sin(math.radians(a)), math.cos(math.radians(a)), 0]
                )
                a += yaw * sign
            scene[key].append(camera_record(camera(name, pos, a, pitch)))
    # The source centers sit below the troffer housing. Entry fill represents light
    # from the adjoining aisle; the enclosing ceiling blocks most of the dome.
    scene["lighting"] = {
        "aim": [0, 0.2, 0.23],
        "presets": {
            "L1": {
                "dome_intensity": 360,
                "lights": [
                    {
                        "name": "aisle_lamp",
                        "position": [0.15, -1.20, 2.39],
                        "size_m": 0.88,
                        "intensity": 16400,
                        "on": True,
                    },
                    {
                        "name": "store_lamp",
                        "position": [-0.15, 1.10, 2.39],
                        "size_m": 0.86,
                        "intensity": 14600,
                        "on": True,
                    },
                    {
                        "name": "entry_fill",
                        "position": [-0.13, -2.70, 1.74],
                        "size_m": 1.45,
                        "intensity": 6200,
                        "on": True,
                    },
                ],
            }
        },
    }
    arr["design"]["appearance"] = (
        "photographed surfaces, stocked steel rack, spatial lights, moving steel door"
    )
    arr["arrangement_sha256"] = arrangement_hash(arr)
    return arr


def generate(out, fixture=False):
    out.mkdir(parents=True, exist_ok=False)
    items = []
    for label, seed in (
        [("CORFIX", 4169)]
        if fixture
        else [(f"COR{i:02}", 4170 + i) for i in range(1, 9)]
    ):
        levels = {}
        for level in ("D0", "D6"):
            obj = arrange(plain_corridor(seed, label, level))
            p = out / (label + level + ".json")
            p.write_text(json.dumps(obj, indent=2) + "\n")
            levels[level] = {
                "path": p.name,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        items.append(
            {
                "base_id": label,
                "seed": seed,
                "environment": "service_workshop",
                "levels": levels,
            }
        )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "study_id": "workshop-realism-v1",
                "split": "pilot",
                "levels": ["D0", "D6"],
                "states": STATES,
                "stops": STOPS,
                "lighting": "L1",
                "arrangements": items,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args()
    generate(a.out, a.fixture)
