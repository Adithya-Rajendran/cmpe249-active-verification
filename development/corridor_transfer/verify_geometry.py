"""Review the corridor design and independently check the cameras read from each simulator."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from development.level_shelf.verify_cameras import angle_between, edge_projection

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LEVELS = ("D0", "D6")
VIEWS = ("P1", "P2", "P3")
CAMERAS = {"nominal": "cameras", "revisit": "revisit_cameras"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(valid, message):
    if not valid:
        raise ValueError(message)


def close(a, b, tolerance, message):
    require(np.allclose(a, b, rtol=0, atol=tolerance), message)


def source_bounds(scene):
    boxes = {b["name"]: b for b in scene["boxes"]}
    for name in ("floor", "left_wall", "right_wall", "door_post_left", "door_post_right", "door_track", "floor_transverse_mark"):
        require(name in boxes, f"Missing corridor geometry {name}")
    require(not any("shelf" in n or "tabletop" in n for n in boxes), "Prior room geometry in corridor")
    mark = boxes["floor_transverse_mark"]
    close(mark["quat_wxyz"], [1, 0, 0, 0], 1e-9, "Stripe is rotated in world coordinates")
    center, half = np.asarray(mark["center"]), np.asarray(mark["half"])
    return np.array([center - half, center + half])


def camera_groups(scene):
    return {role: {c["name"]: c for c in scene[key]} for role, key in CAMERAS.items()}


def heading(c):
    x, y, _ = c["forward"]
    return math.degrees(math.atan2(x, y))


def check_rig(groups, bounds, level):
    angle_rows = []
    for role, cams in groups.items():
        require(set(cams) == set(VIEWS), "Unexpected camera set")
        oblique, leveled, frontal = (cams[v] for v in VIEWS)
        close(oblique["position"], leveled["position"], 1e-6, "Roll moved the lens")
        require(angle_between(oblique["forward"], leveled["forward"]) <= 1e-4, "Roll changed the viewing direction")
        close(frontal["right"], [1, 0, 0], 1e-5, "Frontal camera has roll or yaw")
        require(abs(heading(frontal)) < 1e-4, "Frontal camera does not look along the corridor")
        for view, camera in cams.items():
            require([camera["width"], camera["height"]] == [2304, 1296], "Native resolution changed")
            close([camera["f_px"], *camera["principal_point_px"]], [1731.26, 1152, 648], .1, "Paper intrinsics changed")
            angle, pixels = edge_projection(bounds, camera)
            if view == "P1":
                require(abs(angle) > .05, "Oblique view has no edge tilt")
            else:
                require(abs(angle) < .05, "Corrected or frontal stripe is tilted")
            angle_rows.append({"camera": role, "slot": view, "edge_angle_deg": angle,
                               "left_u": pixels[0, 0], "left_v": pixels[0, 1],
                               "right_u": pixels[1, 0], "right_v": pixels[1, 1]})
    require(abs(groups["nominal"]["P3"]["position"][0]) < 1e-6, "Frontal reference is off the corridor centerline")
    for view in VIEWS:
        before, after = groups["nominal"][view], groups["revisit"][view]
        delta = np.subtract(after["position"], before["position"])
        close(delta[1:], [0, 0], 1e-6, "Revisit motion is not lateral")
        close(abs(delta[0]), .06 if level == "D6" else 0, 1e-6, "Incorrect lens travel")
        yaw = abs(heading(after) - heading(before))
        close(yaw, 3 if level == "D6" and view != "P3" else 0, .001, "Incorrect change in heading")
    return angle_rows


def inspect(run, native=False):
    declaration = read(HERE / "declaration.json")
    endpoint = run / "endpoint-lock.json"
    locked = read(endpoint)
    manifest_path = run / "cohort/manifest.json"
    manifest = read(manifest_path)
    require(sha(HERE / "PROTOCOL.md") == locked["protocol_sha256"], "Protocol changed")
    require(sha(HERE / "declaration.json") == locked["declaration_sha256"], "Declaration changed")
    require(manifest["study_id"] == declaration["study_id"], "Wrong study")
    require(manifest["stops"] == list(VIEWS) and manifest["states"] == declaration["states"], "Wrong conditions")
    layouts = manifest["arrangements"]
    ids = [r["base_id"] for r in layouts]
    seeds = [r["seed"] for r in layouts]
    require(ids == declaration["layouts"] and seeds == declaration["layout_seeds"], "Panel differs from declaration")
    seed_review = read(run / "seed-review.json")
    require(seed_review["passed"] and set(seed_review["seeds"]) == set(seeds + [declaration["fixture_seed"]]), "Seed audit failed")

    design_path = run / "design-release.json"
    if native:
        release = read(design_path)
        require(sha(manifest_path) == release["manifest_sha256"] and
                sha(endpoint) == release["endpoint_lock_sha256"], "Release binding changed")
    with (PROJECT / "paper/evidence/arrangements.csv").open() as f:
        old = list(csv.DictReader(f))
    targets, static_signatures, arrangements, observations, rows = Counter(), set(), {}, {}, []
    for entry in layouts:
        base = entry["base_id"]
        require(set(entry["levels"]) == set(LEVELS), "Missing design level")
        static_nominal = None
        for level in LEVELS:
            rec = entry["levels"][level]
            source = manifest_path.parent / rec["path"]
            require(source.resolve().is_relative_to(manifest_path.parent.resolve()) and sha(source) == rec["sha256"], "Arrangement changed")
            arr = read(source)
            require((arr["base_id"], arr["level"], arr["seed"], arr["environment"]) ==
                    (base, level, entry["seed"], declaration["environment"]), "Wrong scene identity")
            scene, objects = arr["scene"], arr["objects"]
            require(objects["target"]["obj_id"] in declaration["target_bop_ids"], "Unexpected target")
            require(objects["target"]["pose"]["base"][2] == 0 and objects["occluder"]["label"] == "sliding_door",
                    "Target or door outside declared setting")
            static = {k: v for k, v in scene.items() if k not in CAMERAS.values()}
            static = json.dumps([static, objects, scene["cameras"]], sort_keys=True)
            if level == "D0":
                static_nominal = static
                signature = hashlib.sha256(json.dumps([scene["boxes"], objects], sort_keys=True).encode()).hexdigest()
                require(signature not in static_signatures, "Duplicate physical arrangement")
                require(all(int(o["seed"]) != entry["seed"] and int(o["target_bop"]) != objects["target"]["obj_id"]
                            for o in old), "Previously evaluated seed or target identity")
                static_signatures.add(signature)
                targets[objects["target"]["obj_id"]] += 1
                arrangements[base] = {"seed": arr["seed"], "target": arr["target_id"], "geometry_sha256": signature}
            else:
                require(static == static_nominal, "Scene or reference camera changed between D0 and D6")
            bounds = source_bounds(scene)
            design_cams = camera_groups(scene)
            check_rig(design_cams, bounds, level)
            if not native:
                continue
            for engine in declaration["engines"]:
                observed = run / "operator" / engine / base / level / "observed.json"
                observations[str(observed.relative_to(run))] = sha(observed)
                obs = read(observed)
                native_bounds = obs["static_bounds"][declaration["level_edge"]]
                close(native_bounds, bounds, 1e-5, "Rendered stripe differs from design")
                poses = obs["cameras"]
                measured_rows = check_rig(poses, native_bounds, level)
                for row in measured_rows:
                    role, slot = row["camera"], row["slot"]
                    c, target = poses[role][slot], design_cams[role][slot]
                    position_error = float(np.linalg.norm(np.subtract(c["position"], target["position"])))
                    axis_error = max(angle_between(c[a], target[a]) for a in ("right", "up", "forward"))
                    require(position_error <= .0001 and axis_error <= .02, "Measured camera differs from release")
                    rows.append(dict(engine=engine, layout=base, level=level, **row, position_error_m=position_error,
                                     axis_error_deg=axis_error))
    require(targets == Counter({i: 4 for i in declaration["target_bop_ids"]}), "Target balance changed")
    if not native:
        return {"study_id": declaration["study_id"], "new_render_authorized": True,
                "design_locked_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha(manifest_path),
                "endpoint_lock_sha256": sha(endpoint), "protocol_sha256": locked["protocol_sha256"],
                "declaration_sha256": locked["declaration_sha256"], "seed_review_sha256": sha(run / "seed-review.json"),
                "generator_sha256": sha(HERE / "scene.py"), "reviewer_sha256": sha(Path(__file__)), "layouts": arrangements}
    require(len(rows) == 192 and len(observations) == 32, "Native camera inventory incomplete")
    angles = run / "camera-angles.csv"
    with angles.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return {"passed": True, "engines": declaration["engines"], "layouts": ids, "source_records": len(observations),
            "camera_poses_checked": len(rows), "manifest_sha256": sha(manifest_path), "design_release_sha256": sha(design_path),
            "verifier_sha256": sha(Path(__file__)), "angles_csv_sha256": sha(angles), "observed_files": observations,
            "max_leveled_edge_abs_deg": max(abs(r["edge_angle_deg"]) for r in rows if r["slot"] != "P1")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("phase", choices=["inspect", "release", "native"])
    args = parser.parse_args()
    result = inspect(args.run, native=args.phase == "native")
    if args.phase == "inspect":
        print(json.dumps(result, indent=2))
    else:
        name = "design-release.json" if args.phase == "release" else "camera-verification.json"
        with (args.run / name).open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        print(f"Verified and saved {name}")
