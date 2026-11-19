"""Compare rendered development data with the declared geometry and capture checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from development.cross_sim import qualification_geometry as geometry
from development.cross_sim import qualification_visibility as visibility

PROTOCOL = Path(__file__).with_name("qualification_v1.json")
PROTOCOL_SHA = "16c111c97146f1ba876dd1ea0c00bf44f5682fa67f4ccfa8b8510cca460f5539"
AXES = ("right", "up", "forward")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def limit(failures, key, value, bound, minimum=False):
    good = value >= bound if minimum else value <= bound
    if not math.isfinite(value) or not good:
        failures.append({"check": key, "value": value, "limit": bound})


def check_transforms(arrangement, observed, meshes, limits, failures):
    poses = {"target": arrangement["objects"]["target"]["pose"]}
    for i, obj in enumerate(arrangement["objects"]["distractors"]):
        poses[f"distractor_{i}"] = obj["pose"]
    for side, pose in arrangement["objects"]["occluder"]["poses"].items():
        poses["blocker_" + side] = pose
    stats = {"position_m": 0.0, "rotation_deg": 0.0, "bounds_m": 0.0,
             "camera_position_m": 0.0, "camera_axis_deg": 0.0,
             "intrinsics_px": 0.0, "landmark_px": 0.0}
    expected = {}
    for name, pose in poses.items():
        theta = math.radians(pose["yaw_deg"])
        c, s = math.cos(theta), math.sin(theta)
        expected[name] = (np.asarray(pose["base"]), np.array([[c,-s,0],[s,c,0],[0,0,1]]))
    require(set(observed["objects"]) == set(poses), "compiled object inventory differs")
    require(set(observed["static_bounds"]) == set(meshes) - set(poses), "compiled static inventory differs")

    def measure_pose(name, position, rotation):
        p, r = expected[name]
        require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6), "invalid rotation for " + name)
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation @ r.T)-1)/2, -1, 1)))
        stats["rotation_deg"] = max(stats["rotation_deg"], float(angle))
        stats["position_m"] = max(stats["position_m"], float(np.linalg.norm(position - p)))

    for name, mesh in meshes.items():
        if name in poses:
            item = observed["objects"][name]
            measure_pose(name, np.asarray(item["position"]), np.asarray(item["rotation"]))
            actual_bounds = item["bounds"]
        else:
            actual_bounds = observed["static_bounds"][name]
        err = np.max(np.abs(np.asarray(actual_bounds) - mesh["bounds"]))
        require(np.isfinite(err), "nonfinite bounds for " + name)
        stats["bounds_m"] = max(stats["bounds_m"], float(err))

    for state in observed.get("states", observed.get("usd_per_state", {})):
        present = not state.endswith("_removed")
        block = "blocker_placed" if state.startswith("placed_") else "blocker_parked"
        if "usd_per_state" in observed:
            row = observed["usd_per_state"][state]
            target, blocker = (row["/World/Objects/" + n] for n in ("target", "occluder"))
            require(target["visibility"] == ("inherited" if present else "invisible")
                    and blocker["visibility"] == "inherited", "unexpected USD state " + state)
            for key, item in (("target", target), (block, blocker)):
                m = np.asarray(item["world_from_local_row"])
                measure_pose(key, m[3,:3], m[:3,:3].T)
        else:
            row = observed["states"][state]
            require(row["target_enabled"] is present, "unexpected MuJoCo state " + state)
            b = row["blocker"]
            measure_pose(block, np.asarray(b["position"]), np.asarray(b["rotation"]))

    # Compare projected physical landmarks after passing through both observed transforms.
    for group, cameras in observed["cameras"].items():
        for stop, actual in cameras.items():
            ideal = geometry.camera_record(arrangement, stop, revisit=group == "revisit")
            require([actual["width"], actual["height"]] == [ideal["width"], ideal["height"]], "camera size")
            stats["camera_position_m"] = max(stats["camera_position_m"],
                float(np.linalg.norm(np.asarray(actual["position"]) - ideal["position"])))
            for axis in AXES:
                a, b = np.array(actual[axis]), np.array(ideal[axis])
                require(abs(np.linalg.norm(a) - 1) < 1e-6, "invalid camera axis")
                deg = np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)), a @ b))
                stats["camera_axis_deg"] = max(stats["camera_axis_deg"], float(deg))
            optics = [abs(actual["f_px"] - ideal["f_px"]),
                      np.max(np.abs(np.array(actual["principal_point_px"]) - ideal["principal_point_px"]))]
            if "usd_per_state" in observed:
                state = next(iter(observed["usd_per_state"].values()))
                root = "Cameras" if group == "nominal" else "CamerasRevisit"
                usd = state[f"/World/{root}/{stop}"]
                fy = usd["focal_length"] * actual["height"] / usd["vertical_aperture"]
                optics.append(abs(fy - ideal["f_px"]))
            stats["intrinsics_px"] = max(stats["intrinsics_px"], float(max(optics)))
            for name in poses:
                pts = meshes[name]["vertices"]
                sample = pts[::max(1, len(pts)//64)]
                uv, depth = geometry.project(sample, ideal)
                eligible = (depth > 0.01) & np.all((uv >= 0) & (uv < [ideal["width"], ideal["height"]]), axis=1)
                if not eligible.any():
                    continue
                p, rot = expected[name]
                item = observed["objects"][name]
                real = (sample[eligible] - p) @ rot @ np.asarray(item["rotation"]).T + item["position"]
                measured, z = geometry.project(real, actual)
                require(np.all(z > 0), "landmark behind observed camera")
                err = np.linalg.norm(measured - uv[eligible], axis=1).max()
                stats["landmark_px"] = max(stats["landmark_px"], float(err))

    for key, name in {"position_m":"max_position_error_m", "bounds_m":"max_position_error_m",
                      "camera_position_m":"max_position_error_m", "rotation_deg":"max_axis_angle_error_deg",
                      "camera_axis_deg":"max_axis_angle_error_deg", "intrinsics_px":"max_intrinsics_error_px",
                      "landmark_px":"max_landmark_error_px"}.items():
        limit(failures, key, stats[key], limits[name])
    return stats


def get_image(path, size, mask=False):
    with Image.open(path) as image:
        require(image.format == "PNG" and image.size == tuple(size), f"invalid PNG size: {path}")
        if mask:
            require(image.mode in ("L", "1"), f"invalid mask channel type: {path}")
            im = np.asarray(image, dtype=np.uint8)
            require(not np.any((im > 1) & (im != 255)), f"nonbinary mask: {path}")
            return im != 0
        require(image.mode == "RGB", f"invalid RGB channels: {path}")
        return np.array(image)


def compare_images(root, scene, meshes, stop, states, protocol, qualification, failures):
    limits, size = protocol["requirements"], protocol["image_size"]
    band = limits["mask_edge_band_px"]
    kernel = np.ones((2*band+1, 2*band+1), np.uint8)
    results = {"states": {}}
    models = {}
    prefix = stop + "/"

    def mask(role, state):
        return get_image(root / "masks" / f"{role}_{state}_{stop}.png", size, True)

    def rendered(state):
        return get_image(root / "rgb" / f"{state}_{stop}.png", size)

    def projection(group, name):
        key = (group, name)
        if key not in models:
            cam = geometry.camera_record(scene, stop, revisit=group == "revisit")
            models[key] = geometry.rasterize(meshes[name], cam)
        return models[key]

    def enforce(label, result):
        if result["expected_pixels"]:
            limit(failures, prefix + label, result["error_fraction"], limits["max_mask_error_outside_band_over_expected_area"])
        elif result["observed_pixels"]:
            failures.append({"check": prefix + label, "unexpected_pixels": result["observed_pixels"]})

    def compare(actual, ideal, partial=False):
        return geometry.compare_mask(actual, ideal, excess_only=partial, band_px=band,
                                     threshold=limits["max_mask_error_outside_band_over_expected_area"])

    def state_mask(label, actual, group, subject, active):
        silhouette = projection(group, subject)
        full = compare(actual, silhouette)
        # A passing full comparison already satisfies either occlusion condition.
        if full["pass"]:
            return {"comparison": "full", **full}
        camera = geometry.camera_record(scene, stop, revisit=group == "revisit")
        key = (group, subject, tuple(active))
        if key not in models:
            foreground = [meshes[name] for name in active if name != subject]
            models[key] = visibility.visible_mask(meshes[subject], foreground, camera)
        witness = models[key]
        if not witness["occluded_pixels"]:
            enforce(label, full)
            return {"comparison": "full", **full}
        excess = compare(actual, silhouette, True)
        enforce(label, excess)
        visible = compare(actual, witness["visible"])
        if not actual.any() and cv2.erode(witness["visible"].astype(np.uint8), kernel).any():
            failures.append({"check": prefix + label, "message": "geometrically visible interior has no rendered pixels"})
        return {"comparison": "occlusion_verified", **excess,
                "source_occluded_pixels": witness["occluded_pixels"],
                "visible_projection_diagnostic": visible}

    alone = get_image(root / "masks" / f"alone_{stop}.png", size, True)
    results["alone_pixels"] = int(alone.sum())
    results["alone_projection"] = compare(alone, projection("revisit", "target"))
    enforce("alone_projection", results["alone_projection"])
    cache = {}
    for state in states:
        ref = state == "reference"
        group = "nominal" if ref else "revisit"
        block = "blocker_placed" if state.startswith("placed_") else "blocker_parked"
        target, blocker = mask("target", state), mask("blocker", state)
        image = rendered(state)
        visible = int(target.sum())
        require(not np.any(target & blocker), "overlapping instance masks in " + prefix + state)
        row = {"target_pixels": visible, "blocker_pixels": int(blocker.sum())}
        active = [name for name in meshes if name not in ("target", "blocker_parked", "blocker_placed")]
        active.append(block)
        if not state.endswith("_removed"):
            active.append("target")
        if state.endswith("_removed"):
            limit(failures, prefix + state + "/removed_target_pixels", visible, limits["max_removed_target_pixels"])
        else:
            row["target_projection"] = state_mask(state + "/target_projection", target, group, "target", active)
        row["blocker_projection"] = state_mask(state + "/blocker_projection", blocker, group, block, active)
        if ref or state == "clear_present":
            if visible:
                ys, xs = np.nonzero(target)
                row["reference_box" if ref else "target_box"] = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
            baseline = results["alone_pixels"]  # References exist only at D0, with an identical revisit camera.
            row["visible_over_alone"] = visible / max(1, baseline)
            if qualification:
                limit(failures, prefix + state + "/pixels", visible, limits["min_reference_and_clear_target_pixels"], True)
                limit(failures, prefix + state + "/fraction", row["visible_over_alone"], limits["min_clear_target_fraction_of_alone"], True)
            elif ref:
                require(visible > 0, "cannot supply a reference box")
        results["states"][state] = row
        if state in {"clear_present","clear_removed","repeat_clear_present"}:
            cache[state] = (image, target, blocker)

    first, target, blocker = cache["clear_present"]
    interior = cv2.erode(target.astype(np.uint8), kernel).astype(bool)
    if interior.any():
        change = cv2.absdiff(first, cache["clear_removed"][0])[interior].mean()
        results["clear_removal_target_mae"] = float(change)
    else:
        results["clear_removal_target_mae"] = None
    if qualification:
        limit(failures, prefix + "clear_removal_target_mae", results["clear_removal_target_mae"] or 0.,
              limits["min_clear_target_rgb_change_mae"], True)
        last, rt, rb = cache["repeat_clear_present"]
        diff = cv2.absdiff(first, last)
        results["repeat_rgb_mae"] = float(diff.mean())
        results["repeat_target_rgb_mae"] = float(diff[interior].mean()) if interior.any() else None
        limit(failures, prefix + "repeat_rgb_mae", results["repeat_rgb_mae"], limits["max_repeat_rgb_mae"])
        if interior.any():
            limit(failures, prefix + "repeat_target_rgb_mae", results["repeat_target_rgb_mae"], limits["max_repeat_clear_target_rgb_mae"])
        for role, a, b in (("target",target,rt), ("blocker",blocker,rb)):
            xor = np.count_nonzero(a ^ b) / max(1, np.count_nonzero(a | b))
            results["repeat_" + role + "_xor_over_union"] = float(xor)
            limit(failures, prefix + role + "_repeat_xor", xor, limits["max_repeat_mask_xor_over_union"])
    return results


def expected_outputs(stops, states):
    result = {"observed.json"}
    for stop in stops:
        result.add(f"masks/alone_{stop}.png")
        for state in states:
            result.add(f"rgb/{state}_{stop}.png")
            for role in ("target", "blocker"):
                result.add(f"masks/{role}_{state}_{stop}.png")
    return result


def audit(manifest_path, root, assets, engine, binding_path=None, select=None):
    root, assets = Path(root).resolve(), Path(assets).resolve()
    protocol, manifest = read(PROTOCOL), read(manifest_path)
    require(sha(PROTOCOL) == PROTOCOL_SHA, "qualification declaration changed")
    split = manifest["split"]
    require(split in {"qualification", "pilot"} and manifest["study_id"] == protocol["study_id"], "unexpected study or split")
    qual = split == "qualification"
    stops = protocol["stops"]
    require(manifest["stops"] == stops and manifest["states"] == protocol["states"]
            and manifest["levels"] == protocol[split + "_levels"], "manifest differs from declared split")
    require(len(manifest["arrangements"]) == (2 if qual else 10), "incomplete arrangement cohort")
    report = {"study_id": manifest["study_id"], "split": split, "engine": engine, "root": str(root),
              "manifest_sha256": sha(manifest_path), "protocol_sha256": PROTOCOL_SHA,
              "reviewer_sha256": sha(__file__), "geometry_sha256": sha(geometry.__file__),
              "visibility_sha256": sha(visibility.__file__),
              "units": {}, "files": {}, "failures": []}
    binding = None
    if engine == "isaac":
        require(binding_path is not None, "supply Isaac capture binding")
        binding = read(binding_path)
        require(binding["manifest_sha256"] == report["manifest_sha256"]
                and binding["qualification_declaration_sha256"] == PROTOCOL_SHA
                and binding["split"] == split and binding["study_id"] == manifest["study_id"], "Isaac binding mismatch")
        report["capture_binding_sha256"] = sha(binding_path)
        for path, expected in binding["source"].items():
            require(sha(path) == expected, "Isaac source digest mismatch: " + path)
        for path, expected in binding["assets"].items():
            require(sha(assets/path) == expected, "Isaac asset digest mismatch: " + path)
        report["runtime"], report["settings"] = binding["runtime"], binding["settings"]

    @lru_cache(maxsize=None)
    def input_hash(path):
        return sha(path)

    for entry in manifest["arrangements"]:
        base = entry["base_id"]
        for level, definition in entry["levels"].items():
            label = base + "/" + level
            if select and select != label:
                continue
            folder = root / base / level
            print("Auditing", engine, label, flush=True)
            failures = []
            unit = report["units"][label] = {"failures": failures, "views": {}}
            try:
                path = Path(manifest_path).parent / definition["path"]
                require(sha(path) == definition["sha256"], "arrangement digest mismatch")
                arr = read(path)
                require(arr["base_id"] == base and arr["level"] == level and arr["split"] == "dev"
                        and arr["environment"] == entry["environment"] and arr["seed"] == entry["seed"]
                        and type(arr["seed"]) is int and 0 <= arr["seed"] <= 9999, "invalid arrangement identity")
                states = list(protocol["states"])
                if level == "D0":
                    states.insert(0, "reference")
                if qual:
                    states.append("repeat_clear_present")
                marker = folder / ("completion.json" if engine == "isaac" else "complete.json")
                completion = read(marker)
                filenames = expected_outputs(stops, states)
                require(filenames == set(completion["outputs"]), "completion ledger inventory mismatch")
                actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
                require(actual == filenames | {marker.name}, "unexpected or missing output files")
                for name in sorted(filenames):
                    digest = sha(folder / name)
                    require(digest == completion["outputs"][name], "output digest mismatch: " + name)
                    report["files"][label + "/" + name] = digest
                report["files"][label + "/" + marker.name] = sha(marker)
                observed = read(folder / "observed.json")
                if binding is not None:
                    require(completion["binding_sha256"] == observed["binding_sha256"] == report["capture_binding_sha256"]
                            and observed["arrangement_sha256"] == definition["sha256"], "wrong Isaac unit binding")
                else:
                    identity = completion["identity"]
                    require(identity["arrangement_sha256"] == definition["sha256"]
                            and identity["manifest_sha256"] == report["manifest_sha256"]
                            and identity["declaration_sha256"] == PROTOCOL_SHA, "wrong MuJoCo unit binding")
                    for source, expected in {**identity["sources_sha256"], **identity["assets_sha256"]}.items():
                        require(input_hash(source) == expected, "MuJoCo source or asset digest mismatch: " + source)
                    unit["runtime"] = identity["runtime"]
                actual_states = observed["usd_per_state"] if binding is not None else observed["states"]
                require(set(actual_states) == set(states), "missing physical state observation")
                require(set(observed["cameras"]) == {"nominal","revisit"} and
                        all(set(group) == set(stops) for group in observed["cameras"].values()), "camera inventory mismatch")
                meshes = geometry.source_geometry(arr, assets / "models")
                unit["arrangement_file_sha256"] = definition["sha256"]
                unit["geometry"] = check_transforms(arr, observed, meshes, protocol["requirements"], failures)
                for stop in stops:
                    unit["views"][stop] = compare_images(folder, arr, meshes, stop, states, protocol, qual, failures)
            except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
                failures.append({"check": "integrity_or_schema", "message": str(exc)})
            if failures:
                report["failures"].append(label)
                print(label, failures[:3], flush=True)
    require(bool(report["units"]), "unknown unit")
    report["status"] = "FAIL" if report["failures"] else ("PARTIAL" if select else "PASS")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "renders", "assets", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engine", choices=("mujoco","isaac"), required=True)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--unit", help="Inspect one base/level. A partial report cannot release the cohort.")
    args = parser.parse_args()
    require(not args.out.exists(), "choose a fresh output path")
    result = audit(args.manifest, args.renders, args.assets, args.engine, args.binding, args.unit)
    with args.out.open("x") as dest:
        json.dump(result, dest, indent=2, allow_nan=False)
        dest.write("\n")
    print(result["status"], len(result["units"]), "units,", len(result["files"]), "files,", result["failures"], flush=True)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
