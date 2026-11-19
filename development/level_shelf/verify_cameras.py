"""Check the floor edge and paired viewpoints using camera transforms read from each renderer."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angle_between(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def edge_projection(bounds, camera):
    lo, hi = bounds
    edge = np.array([[lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]]])
    offset = edge - camera["position"]
    depth = offset @ camera["forward"]
    if min(depth) <= 0:
        raise ValueError("Shelf edge falls behind the measured camera")
    axes = np.column_stack((camera["right"], -np.array(camera["up"])))
    pixels = (offset @ axes) * camera["f_px"] / depth[:, None] + camera["principal_point_px"]
    dx, dy = pixels[1] - pixels[0]
    return math.degrees(math.atan2(dy, dx)), pixels


def verify(run):
    manifest_path = run / "cohort/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    release_path = run / "design-release.json"
    release = json.loads(release_path.read_text())
    if sha(manifest_path) != release["manifest_sha256"]:
        raise ValueError("Manifest hash differs from design release")
    observed_files = {}
    rows = []
    engines = ["isaac", "mujoco"]
    layouts = [layout["base_id"] for layout in manifest["arrangements"]]
    if layouts != [f"PAM{idx:02}" for idx in range(1, 9)]:
        raise ValueError("Unexpected layout panel")
    for engine in engines:
        for layout in manifest["arrangements"]:
            base = layout["base_id"]
            for level, entry in layout["levels"].items():
                source = manifest_path.parent / entry["path"]
                if sha(source) != entry["sha256"]:
                    raise ValueError(f"Changed camera recipe {source}")
                scene = json.loads(source.read_text())["scene"]
                path = run / "operator" / engine / base / level / "observed.json"
                observed_files[str(path.relative_to(run))] = sha(path)
                observed = json.loads(path.read_text())
                bounds = observed["static_bounds"]["c3_floor_board"]
                if not np.allclose(bounds, [[-.395, 0, -.018727], [.395, .305, 0]], rtol=0, atol=1e-5):
                    raise ValueError(f"Shelf geometry changed at {path}")
                groups = observed["cameras"]
                for role, key in (("nominal", "cameras"), ("revisit", "revisit_cameras")):
                    cameras = groups[role]
                    if set(cameras) != set(manifest["stops"]):
                        raise ValueError("Missing a measured viewpoint")
                    for requested in scene[key]:
                        slot = requested["name"]
                        c = cameras[slot]
                        if [c["width"], c["height"]] != [2304, 1296]:
                            raise ValueError("Native resolution differs")
                        intrinsics = [c["f_px"], *c["principal_point_px"]]
                        expected = [requested["f_px"], *requested["principal_point_px"]]
                        np.testing.assert_allclose(intrinsics, expected, rtol=0, atol=0.1)
                        pos_error = np.linalg.norm(np.subtract(c["position"], requested["position"]))
                        axis_error = max(angle_between(c[axis], requested[axis]) for axis in ("forward", "up", "right"))
                        if pos_error > 1e-4 or axis_error > .02:
                            raise ValueError(f"Unmatched camera transform at {engine} {base} {level} {role} {slot}")
                        if slot in ("P2", "P5"):
                            original = cameras[{"P2": "P1", "P5": "P4"}[slot]]
                            np.testing.assert_allclose(c["position"], original["position"], rtol=0, atol=1e-6)
                            if angle_between(c["forward"], original["forward"]) > 1e-4:
                                raise ValueError("Roll correction changed the optical axis")
                        if slot == "P3":
                            np.testing.assert_allclose(c["forward"], [0, 1, 0], rtol=0, atol=1e-6)
                            np.testing.assert_allclose(c["up"], [0, 0, 1], rtol=0, atol=1e-6)
                        angle, pixels = edge_projection(bounds, c)
                        if slot in ("P2", "P3", "P5") and abs(angle) >= .05:
                            raise ValueError("The observed shelf edge fails the level requirement")
                        left, right = pixels
                        rows.append(dict(engine=engine, layout=base, level=level, camera=role, slot=slot,
                                         edge_angle_deg=angle, left_u=left[0], left_v=left[1], right_u=right[0],
                                         right_v=right[1], position_error_m=pos_error, axis_error_deg=axis_error))
                for slot in manifest["stops"]:
                    distance = np.linalg.norm(np.subtract(groups["nominal"][slot]["position"],
                                                          groups["revisit"][slot]["position"]))
                    if not math.isclose(distance, .06 if level == "D6" else 0, abs_tol=1e-6):
                        raise ValueError("Unexpected lens travel distance")
    if len(rows) != 320 or len(observed_files) != 32:
        raise ValueError("Incomplete native geometry record")
    angles = run / "camera-angles.csv"
    with angles.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    evidence = {"passed": True, "engines": engines, "layouts": layouts, "source_records": len(observed_files),
                "camera_poses_checked": len(rows), "manifest_sha256": sha(manifest_path),
                "design_release_sha256": sha(release_path), "verifier_sha256": sha(Path(__file__)),
                "angles_csv_sha256": sha(angles), "observed_files": observed_files,
                "max_leveled_edge_abs_deg": max(abs(r["edge_angle_deg"]) for r in rows if r["slot"] in ("P2", "P3", "P5"))}
    receipt = run / "camera-verification.json"
    receipt.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Verified {len(rows)} native camera poses. Maximum corrected edge angle {evidence['max_leveled_edge_abs_deg']:.8g} degrees.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
