"""Fit a two-class RGB policy in simulation and execute it on reserved workshop layouts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from development.workshop_realism.robot_env import WorkshopEnv
from development.workshop_realism.training_task import DoorApproachTask

TRAIN = ("COR01", "COR02", "COR03", "COR04", "COR05", "COR06")
CHECK = ("COR07", "COR08")
POSES = [(x, yaw) for x in (-0.12, 0, 0.12) for yaw in (-5, 5)]
EVAL_POSES = [(-0.08, 3), (0.08, -3)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(rgb):
    image = (
        Image.fromarray(rgb)
        .convert("L")
        .resize((32, 18), getattr(Image, "Resampling", Image).BILINEAR)
    )
    pixels = np.asarray(image, dtype=float).ravel() / 255.0
    return (pixels - pixels.mean()) / max(float(pixels.std()), 0.02)


def demo(args):
    args.out.mkdir(parents=True, exist_ok=False)
    scenes = {}
    if args.manifest:
        manifest = json.loads(args.manifest.read_bytes())
        scenes = {x["base_id"]: x["levels"]["D0"] for x in manifest["arrangements"]}

    def environment(layout):
        if args.layouts:
            return WorkshopEnv.from_mjcf(
                args.layouts / (layout + ".xml"), imagesize=(320, 180)
            )
        spec = scenes[layout]
        path = args.manifest.parent / spec["path"]
        if sha(path) != spec["sha256"]:
            raise ValueError("Scene changed")
        a = json.loads(path.read_bytes())
        return WorkshopEnv(
            a, args.models, args.tags, args.textures, imagesize=(320, 180)
        )

    examples = []
    labels = []
    for layout in TRAIN:
        env = environment(layout)
        try:
            for opened in (False, True):
                for x, yaw in POSES:
                    obs = env.reset(
                        target_present=False,
                        door_fraction=float(opened),
                        lateral=x,
                        heading=math.radians(yaw),
                    )
                    examples.append(features(obs["rgb"]))
                    labels.append(int(opened))
        finally:
            env.close()
    values = np.array(examples)
    labels = np.array(labels)
    centroid = np.stack([values[labels == label].mean(axis=0) for label in (0, 1)])
    np.savez(args.out / "rgb-policy.npz", closed=centroid[0], opened=centroid[1])
    decisions = []
    for layout in CHECK:
        env = environment(layout)
        task = DoorApproachTask(env, float(env.model.body("blocker").pos[1]))
        try:
            for opened in (False, True):
                for idx, (x, yaw) in enumerate(EVAL_POSES):
                    obs = task.reset(
                        open_door=opened, lateral=x, heading=math.radians(yaw)
                    )
                    frame = args.out / f"{layout}-{int(opened)}-{idx}.png"
                    Image.fromarray(obs["rgb"]).save(frame)
                    ds = np.mean((centroid - features(obs["rgb"])) ** 2, axis=1)
                    go = bool(ds[1] < ds[0])
                    done = False
                    while not done:
                        _, reward, done, info = task.step(0.34, 0, stop=not go)
                    decisions.append(
                        {
                            "layout": layout,
                            "door_open": opened,
                            "lateral_m": x,
                            "heading_deg": yaw,
                            "selected_action": "go" if go else "stop",
                            "distance_to_closed": float(ds[0]),
                            "distance_to_open": float(ds[1]),
                            "reward": reward,
                            **info,
                            "starting_frame_sha256": sha(frame),
                        }
                    )
        finally:
            env.close()
    summary = {
        "training_layouts": TRAIN,
        "reserved_layouts": CHECK,
        "training_images": len(examples),
        "held_out_episodes": len(decisions),
        "successful_episodes": sum(x["reward"] == 1 for x in decisions),
        "method": "nearest class centroid on normalized 32 x 18 grayscale features, fitted only on the six training layouts; a start-frame action drives forward or stops",
        "model_sha256": sha(args.out / "rgb-policy.npz"),
        "source_sha256": sha(Path(__file__)),
        "episodes": decisions,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "Training images",
        len(examples),
        "episodes",
        len(decisions),
        "successful",
        summary["successful_episodes"],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--layouts", type=Path)
    for k in ["models", "tags", "textures"]:
        p.add_argument("--" + k, type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.manifest and not all((args.models, args.tags, args.textures)):
        p.error("Geometry generation requires --models, --tags and --textures")
    demo(args)
