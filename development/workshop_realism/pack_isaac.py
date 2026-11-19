"""Package the native Isaac facilities after checking door motion and contact in each layout."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from development.workshop_realism.release import HERE, PROJECT, read, require, sha


def build(run, dest):
    panel = run / "robot/isaac-panel"
    results = read(panel / "summary.json")
    require(results["passed"], "Native facility acceptance failed")
    reports = {row["layout"]: row for row in results["layouts"]}
    designs = read(run / "cohort/manifest.json")["arrangements"]
    require(
        len(reports) == len(results["layouts"]) == len(designs) == 8
        and set(reports) == {r["base_id"] for r in designs},
        "The panel must contain all eight declared facilities",
    )
    descriptions = []
    for entry in designs:
        layout = entry["base_id"]
        recipe = entry["levels"]["D0"]
        scene = run / "cohort" / recipe["path"]
        proof = read(panel / layout / "report.json")
        asset = panel / layout / "facility.usdz"
        require(
            proof == reports[layout]
            and proof["pass"]
            and all(proof["checks"].values())
            and sha(scene) == proof["scene_sha256"] == recipe["sha256"]
            and sha(asset) == proof["usdz_sha256"],
            "The native test, model and recipe must agree for " + layout,
        )
        poses = read(scene)["objects"]["occluder"]["poses"]
        travel = poses["parked"]["base"][0]
        door_y = poses["placed"]["base"][1]
        states = {r["stage"]: r for r in proof["records"]}
        require(
            abs(states["closed"]["door_x"]) < 0.012
            and abs(states["middle"]["door_x"] - 0.5 * travel) < 0.012
            and abs(states["open"]["door_x"] - travel) < 0.012
            and door_y - 0.25 < states["blocked"]["probe_y"] < door_y - 0.12
            and states["crossed"]["probe_y"] > door_y + 0.30
            and all(
                abs(a - b) < 0.00001
                for a, b in zip(
                    proof["reopened_joint_limits_m"], sorted((0.0, travel)), strict=True
                )
            ),
            "Measured door motion, contact or passage failed for " + layout,
        )
        with ZipFile(asset) as native:
            members = native.namelist()
            require(
                members[0].endswith(".usdc")
                and native.testzip() is None
                and sum(n.lower().endswith(".png") for n in members)
                == proof["packaged_textures"]
                >= 13
                and set(proof["resolved_texture_members"])
                == {n for n in members if n.lower().endswith(".png")}
                and proof["packaged_colliders"] >= 40,
                "Incomplete USDZ for " + layout,
            )
            stage = native.read(members[0])
            require(stage.startswith(b"PXR-USDC"), "Invalid binary USD stage")
        descriptions.append(
            {
                "layout": layout,
                "file": layout + "/facility.usdz",
                "usdz_sha256": proof["usdz_sha256"],
                "embedded_stage_sha256": hashlib.sha256(stage).hexdigest(),
                "textures": proof["packaged_textures"],
                "resolved_inside_package": len(proof["resolved_texture_members"]),
                "colliders": proof["packaged_colliders"],
                "joint_limits_m": proof["reopened_joint_limits_m"],
                "blocked_distance_before_door_m": door_y - states["blocked"]["probe_y"],
                "crossing_distance_past_door_m": states["crossed"]["probe_y"] - door_y,
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    require(not dest.exists(), "Package already exists")
    partial = dest.with_suffix(".partial")
    with ZipFile(partial, "x", compression=ZIP_DEFLATED) as bundle:
        for row in descriptions:
            layout = row["layout"]
            bundle.write(panel / row["file"], row["file"], compress_type=ZIP_STORED)
            bundle.write(panel / layout / "report.json", layout + "/report.json")
        bundle.write(HERE / "ROBOTICS.md", "ROBOTICS.md")
        bundle.write(HERE / "materials/sources.json", "materials/sources.json")
        bundle.write(HERE / "materials.py", "materials.py")
        bundle.write(PROJECT / "paper/evidence/ASSET-CREDITS.md", "YCB-CREDITS.md")
        credits = (
            (HERE / "ASSETS.md")
            .read_text()
            .replace("../../paper/evidence/ASSET-CREDITS.md", "YCB-CREDITS.md")
        )
        bundle.writestr("ASSETS.md", credits)
        bundle.writestr("manifest.json", json.dumps(descriptions, indent=2) + "\n")
    with ZipFile(partial) as complete:
        require(
            complete.testzip() is None, "Facility bundle failed its integrity check"
        )
    partial.rename(dest)
    return {
        "layouts": descriptions,
        "bytes": dest.stat().st_size,
        "sha256": sha(dest),
        "native_test": "Driven door, guided contact cart, and reopened textured USDZ in Isaac Sim",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    print(json.dumps(build(args.run, args.dest), indent=2))
