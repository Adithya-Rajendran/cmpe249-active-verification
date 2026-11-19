"""Package the eight native MuJoCo training models, shared textures and runnable episode interface."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from development.workshop_realism.release import HERE, PROJECT, read, require, sha


def build(run, dest):
    panel = run / "robot/panel-release"
    proof = read(panel / "summary.json")
    require(proof["passed"] and len(proof["results"]) == 8, "Native panel did not pass")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".partial")
    require(not dest.exists(), "Output already exists")
    assets = {}
    receipt = []
    with ZipFile(partial, "x", allowZip64=True, compression=ZIP_DEFLATED) as archive:
        for r in proof["results"]:
            layout = r["layout"]
            source = panel / layout / "facility.xml"
            require(sha(source) == r["physics_model_sha256"], "Native XML has changed")
            tree = ET.parse(source)
            ref_files = {}
            for elem in tree.iter():
                if "file" not in elem.attrib:
                    continue
                path = Path(elem.attrib["file"])
                digest = sha(path)
                name = "assets/" + digest + path.suffix.lower()
                if name not in assets:
                    assets[name] = digest
                    archive.write(path, name, compress_type=ZIP_STORED)
                elem.set("file", "../" + name)
                ref_files[elem.get("name")] = digest
            archive.writestr(
                "layouts/" + layout + ".xml",
                ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True),
            )
            receipt.append(
                {
                    "layout": layout,
                    "original_model_sha256": sha(source),
                    "textures": ref_files,
                    "physics_checks": r["checks"],
                }
            )
        for name in [
            "robot_env.py",
            "training_task.py",
            "portable_robot_check.py",
            "train_navigation.py",
            "partial_openings.py",
        ]:
            archive.write(HERE / name, "development/workshop_realism/" + name)
        archive.write(HERE / "ROBOTICS.md", "ROBOTICS.md")
        archive.write(HERE / "materials/sources.json", "materials/sources.json")
        archive.write(HERE / "materials.py", "materials.py")
        credits = (
            (HERE / "ASSETS.md")
            .read_text()
            .replace("../../paper/evidence/ASSET-CREDITS.md", "YCB-CREDITS.md")
        )
        archive.writestr("ASSETS.md", credits)
        archive.write(PROJECT / "paper/evidence/ASSET-CREDITS.md", "YCB-CREDITS.md")
        for name in ("summary.json", "rgb-policy.npz"):
            archive.write(
                run / "robot/navigation-demo" / name, "example-policy/" + name
            )
        opening_check = read(run / "robot/partial-openings.json")
        require(
            opening_check["policy_sha256"]
            == sha(run / "robot/navigation-demo/rgb-policy.npz")
            and opening_check["code_sha256"] == sha(HERE / "partial_openings.py"),
            "The partial-opening check used different code or policy weights",
        )
        archive.write(
            run / "robot/partial-openings.json", "example-policy/partial-openings.json"
        )
        archive.write(
            run / "robot/partial-opening-design.json",
            "example-policy/partial-opening-design.json",
        )
        frames = sorted((run / "robot/partial-frames").glob("*.png"))
        require(
            len(frames) == len(opening_check["cases"]) == 20,
            "Incomplete camera observations",
        )
        for frame in frames:
            archive.write(
                frame, "example-policy/frames/" + frame.name, compress_type=ZIP_STORED
            )
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "mujoco_version": proof["results"][0]["mujoco"],
                    "layouts": receipt,
                    "assets": assets,
                    "robot_source_sha256": sha(HERE / "robot_env.py"),
                },
                indent=2,
            )
            + "\n",
        )
    with ZipFile(partial) as archive:
        require(
            archive.testzip() is None, "Training archive failed its integrity check"
        )
    partial.rename(dest)
    return {
        "models": len(receipt),
        "unique_textures": len(assets),
        "bytes": dest.stat().st_size,
        "sha256": sha(dest),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    print(json.dumps(build(args.run, args.dest), indent=2))
