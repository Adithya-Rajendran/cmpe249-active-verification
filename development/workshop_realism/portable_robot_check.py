"""Run from an extracted training archive to verify every native model and one RGB policy."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from development.workshop_realism.robot_env import WorkshopEnv
from development.workshop_realism.training_task import DoorApproachTask


def check(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    results = []
    for name, digest in manifest["assets"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Texture digest mismatch")
    weights = np.load(root / "example-policy/rgb-policy.npz")
    means = np.stack((weights["closed"], weights["opened"]))
    for r in manifest["layouts"]:
        layout = r["layout"]
        env = WorkshopEnv.from_mjcf(
            root / "layouts" / (layout + ".xml"), imagesize=(320, 180)
        )
        try:
            door_y = float(env.model.body("blocker").pos[1])
            task = DoorApproachTask(env, door_y)
            for opened in (False, True):
                obs = task.reset(
                    open_door=opened, lateral=-0.08, heading=math.radians(3)
                )
                frame = obs["rgb"]
                image = (
                    Image.fromarray(frame)
                    .convert("L")
                    .resize((32, 18), getattr(Image, "Resampling", Image).BILINEAR)
                )
                x = np.asarray(image, dtype=float).ravel()
                x /= 255.0
                x = (x - x.mean()) / max(x.std(), 0.02)
                score = np.mean((means - x) ** 2, axis=1)
                go = score[1] < score[0]
                done = False
                while not done:
                    _, reward, done, info = task.step(0.34, 0, stop=not go)
                results.append(
                    {
                        "layout": layout,
                        "door_open": opened,
                        "policy_action": "go" if go else "stop",
                        "outcome": info["outcome"],
                        "reward": reward,
                        "frame_shape": list(frame.shape),
                    }
                )
        finally:
            env.close()
    return {
        "models_verified": len(manifest["layouts"]),
        "textures_verified": len(manifest["assets"]),
        "episodes": results,
        "safe_stops_and_crossings": sum(r["reward"] == 1 for r in results),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = check(args.root)
    out = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(out)
    print(out, end="")
