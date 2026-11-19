"""Verify the matched design and seal the visually accepted fixture before panel capture."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / "runs/2026-11-18-corridor-transfer-v1"
METHODS = {"baseline_200": "homography", "geometry": "dense", "vggt_qwen": "registered"}


def sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_design(run):
    lock = read(run / "endpoint-lock.json")
    declaration_file = lock.get("declaration_file", "declaration.json")
    decl = read(HERE / declaration_file)
    corrected = "door_amendment_sha256" in lock
    if corrected:
        from development.workshop_realism.door_clearance import sweep

        require(
            sha(HERE / "DOOR_AMENDMENT.md") == lock["door_amendment_sha256"],
            "Door amendment changed",
        )
    manifest = read(run / "cohort/manifest.json")
    old = read(OLD / "cohort/manifest.json")
    require(
        sha(HERE / "PROTOCOL.md") == lock["protocol_sha256"]
        and sha(HERE / declaration_file) == lock["declaration_sha256"],
        "Changed endpoint",
    )
    require(
        [x["base_id"] for x in manifest["arrangements"]] == decl["layouts"]
        and manifest["stops"] == decl["stops"],
        "Different layouts or views",
    )
    require(
        manifest["states"] == decl["states"]
        and manifest["levels"] == decl["pilot_levels"],
        "Different states or levels",
    )
    cameras = 0
    n = 0
    for new, previous in zip(manifest["arrangements"], old["arrangements"]):
        require(
            new["base_id"] == previous["base_id"] and new["seed"] == previous["seed"],
            "Broken layout pair",
        )
        for level in manifest["levels"]:
            item = new["levels"][level]
            orig = previous["levels"][level]
            source = run / "cohort" / item["path"]
            old_source = OLD / "cohort" / orig["path"]
            require(
                sha(source) == item["sha256"] and sha(old_source) == orig["sha256"],
                "Broken scene hash",
            )
            a, b = read(source), read(old_source)
            n += 1
            expected_door = {
                name: {
                    **pose,
                    "base": [
                        *pose["base"][:2],
                        0.014 if corrected else pose["base"][2],
                    ],
                }
                for name, pose in b["objects"]["occluder"]["poses"].items()
            }
            require(
                a["objects"]["target"] == b["objects"]["target"]
                and a["objects"]["occluder"]["poses"] == expected_door,
                "Changed positions",
            )
            if corrected:
                proof = sweep(a)
                require(
                    not proof["hits"] and proof["minimum_static_clearance_m"] >= 0.01,
                    "Insufficient full-travel door clearance",
                )
            require(
                a["objects"]["occluder"]["size_m"][0]
                == b["objects"]["occluder"]["size_m"][0],
                "Door width changed",
            )
            for group in ["cameras", "revisit_cameras"]:
                c = a["scene"][group]
                require(
                    c[:3] == b["scene"][group]
                    and [x["name"] for x in c] == decl["stops"],
                    "Reference camera geometry changed",
                )
                cameras += 3
            for idx, height, pitch in [(3, 0.95, 13.0), (4, 0.48, 8.0)]:
                before = a["scene"]["cameras"][idx]
                after = a["scene"]["revisit_cameras"][idx]
                require(
                    before["position"][2] == after["position"][2] == height
                    and before["pitch_down_deg"] == after["pitch_down_deg"] == pitch,
                    "Wrong camera mount",
                )
                travel = np.linalg.norm(
                    np.subtract(after["position"], before["position"])
                )
                require(
                    abs(travel - (0.06 if level == "D6" else 0)) < 1e-8,
                    "Approach motion differs",
                )
            for m in a["scene"]["materials"].values():
                if "texture" in m:
                    require(
                        sha(HERE / "materials" / m["texture"]["file"])
                        == m["texture"]["sha256"],
                        "Material differs",
                    )
    return {
        "specifications": n,
        "matched_camera_records": cameras,
        "source_files": {
            name: sha(HERE / name)
            for name in [
                "scene.py",
                "capture.py",
                "materials.py",
                "materials/sources.json",
                declaration_file,
                "PROTOCOL.md",
            ]
            + (["door_clearance.py", "DOOR_AMENDMENT.md"] if corrected else [])
        },
        "endpoint_lock_sha256": sha(run / "endpoint-lock.json"),
        "cohort_manifest_sha256": sha(run / "cohort/manifest.json"),
        "previous_manifest_sha256": sha(OLD / "cohort/manifest.json"),
    }


def seal(run):
    release = validate_design(run)
    review = read(run / "fixture-review.json")
    require(
        review["accepted"] and set(review["engines"]) == {"isaac", "mujoco"},
        "Visual review incomplete",
    )
    for name, digest in review["images"].items():
        require(sha(name) == digest, "Fixture image changed")
    release.update(
        fixture_review_sha256=sha(run / "fixture-review.json"),
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        panel_authorized=True,
    )
    write_once(run / "design-release.json", release)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "seal"])
    p.add_argument("run", type=Path)
    a = p.parse_args()
    if a.mode == "seal":
        seal(a.run)
    else:
        print(json.dumps(validate_design(a.run), indent=2))
