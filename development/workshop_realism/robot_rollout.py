"""Exercise the sliding door, blocked/open navigation, steering, camera and reset in MuJoCo."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from development.workshop_realism.door_clearance import TRAVEL
from development.workshop_realism.release import read, sha
from development.workshop_realism.robot_env import WorkshopEnv


def distance(p, q):
    return float(np.linalg.norm(np.asarray(p) - q))


def photograph(env, out, label, camera):
    a = env.frame(camera)
    path = out / (label + ".png")
    Image.fromarray(a).save(path)
    return {
        "path": path.name,
        "sha256": sha(path),
        "size": list(a.shape),
        "std": float(np.std(a)),
    }


def audit(arr, assets, out, with_images):
    env = WorkshopEnv(arr, *assets, rgb=with_images)
    out.mkdir(parents=True, exist_ok=False)
    (out / "facility.xml").write_text(env.xml)
    photos = {}
    check = {}
    traces = []
    try:
        env.reset(target_present=False)
        center = arr["objects"]["occluder"]["poses"]["placed"]["base"][1]
        if with_images:
            photos["closed"] = photograph(env, out, "door-closed", "door_inspection")
        for name, goal in [("middle", 0.5), ("open", 1.0), ("closed", 0.0)]:
            samples = []
            contacts = []
            for _ in range(110):
                env.step(0, 0, goal, render=False)
                d = env.diagnostic()
                samples.append(d["door_m"])
                contacts.extend(d["door_contacts"])
            check[name + "_joint_target"] = abs(samples[-1] - goal * TRAVEL) < 0.008
            check[name + "_room_clear"] = len(contacts) == 0
            traces.append(
                {
                    "test": name,
                    "target_m": goal * TRAVEL,
                    "actual_m": samples[-1],
                    "min_m": min(samples),
                    "max_m": max(samples),
                    "contacts": contacts[:10],
                }
            )
            if name in ("middle", "open") and with_images:
                photos[name] = photograph(env, out, "door-" + name, "door_inspection")
        for name, door in [("blocked", 0.0), ("crossed", 1.0)]:
            env.reset(target_present=False, door_fraction=door)
            start = env.diagnostic()["robot_xyz"]
            hits = 0
            max_progress = -np.inf
            minimum_door = np.inf
            penetration = 0.0
            if with_images:
                photos[name + "_start"] = photograph(
                    env, out, name + "-start", "robot_mast"
                )
            for _ in range(120):
                env.step(0.34, 0, door, render=False)
                d = env.diagnostic()
                max_progress = max(max_progress, d["robot_xyz"][1])
                minimum_door = min(minimum_door, d["door_m"])
                for ga, gb, depth in d["door_contacts"]:
                    if "bumper" in (ga, gb):
                        hits += 1
                        penetration = min(penetration, depth)
            end = env.diagnostic()
            traces.append(
                {
                    "test": name,
                    "door_center_y_m": center,
                    "start": start,
                    "end": end["robot_xyz"],
                    "farthest_y_m": max_progress,
                    "bumper_contacts": hits,
                    "maximum_penetration_m": -penetration,
                    "site_door_position_m": end["door_m"],
                    "wheel_radians": env.observe(False)["wheel_radians"].tolist(),
                }
            )
            if not door:
                check["closed_stops_robot"] = (
                    hits > 0
                    and start[1] + 0.35 < max_progress < center - 0.10
                    and penetration > -0.012
                )
            else:
                check["open_allows_crossing"] = (
                    max_progress > center + 0.25 and hits == 0 and minimum_door > 0.95
                )
            if with_images:
                photos[name + "_end"] = photograph(
                    env, out, name + "-end", "robot_mast"
                )
        env.reset(target_present=True)
        initial = env.diagnostic()
        im0 = env.frame() if with_images else None
        for _ in range(28):
            env.step(0, 0.8, 0, render=False)
        turned = env.diagnostic()
        angle = 2 * np.arccos(
            min(1.0, abs(float(np.dot(initial["robot_quat"], turned["robot_quat"]))))
        )
        check["turns_in_place"] = (
            0.30 < angle < 1.9
            and distance(initial["robot_xyz"], turned["robot_xyz"]) < 0.12
        )
        if with_images:
            im1 = env.frame()
            photos["steering"] = photograph(env, out, "steer", "robot_mast")
            check["body_camera_changes"] = (
                float(np.abs(im0.astype(float) - im1).mean()) > 2
            )
        env.reset(target_present=True)
        again = env.diagnostic()
        check["reset_restores_pose"] = (
            distance(initial["robot_xyz"], again["robot_xyz"]) < 1e-6
            and abs(again["door_m"]) < 0.005
        )
        traces.append(
            {
                "test": "steering",
                "angle_rad": float(angle),
                "translation_m": distance(initial["robot_xyz"], turned["robot_xyz"]),
            }
        )
        return dict(
            layout=arr["base_id"],
            passed=all(check.values()),
            checks={key: bool(value) for key, value in check.items()},
            traces=traces,
            images=photos,
            physics_model_sha256=sha(out / "facility.xml"),
            mujoco=mujoco.__version__,
        )
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--tags", type=Path, required=True)
    p.add_argument("--textures", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = []
    for entry in read(args.manifest)["arrangements"]:
        item = entry["levels"]["D0"]
        spec = args.manifest.parent / item["path"]
        if sha(spec) != item["sha256"]:
            raise ValueError("Changed facility geometry")
        report = audit(
            read(spec),
            (args.models, args.tags, args.textures),
            args.out / entry["base_id"],
            entry["base_id"] in ["CORFIX", "COR01"],
        )
        (args.out / (entry["base_id"] + ".json")).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        out.append(report)
        print(entry["base_id"], report["checks"], flush=True)
    valid = all(x["passed"] for x in out)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "passed": valid,
                "manifest_sha256": sha(args.manifest),
                "env_sha256": sha(Path(__file__).with_name("robot_env.py")),
                "test_sha256": sha(Path(__file__)),
                "results": out,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if valid else 1)
