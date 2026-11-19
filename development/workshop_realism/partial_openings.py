"""Test a fitted stop/go policy against the physical clearance of partly open doors."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from development.workshop_realism.robot_env import WorkshopEnv
from development.workshop_realism.train_navigation import CHECK, EVAL_POSES, features


FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def drive(env, door_y, go):
    for step in range(1, 121):
        env.step(0.34 if go else 0.0, 0.0, render=False)
        trace = env.diagnostic()
        progress = trace["robot_xyz"][1] - door_y
        if trace["door_contacts"]:
            return {"outcome": "contact", "step": step, "center_past_door_m": progress}
        if progress > 0.30:
            return {"outcome": "crossed", "step": step, "center_past_door_m": progress}
        if not go:
            return {"outcome": "stopped", "step": step, "center_past_door_m": progress}
    return {"outcome": "timeout", "step": step, "center_past_door_m": progress}


def evaluate(package):
    weights_file = package / "example-policy/rgb-policy.npz"
    with np.load(weights_file) as weights:
        centers = np.stack([weights["closed"], weights["opened"]])
    cases = []
    for layout in CHECK:
        env = WorkshopEnv.from_mjcf(
            package / "layouts" / (layout + ".xml"), imagesize=(320, 180)
        )
        try:
            door_y = float(env.model.body("blocker").pos[1])
            for fraction in FRACTIONS:
                for x, yaw in EVAL_POSES:
                    start = dict(
                        target_present=False,
                        door_fraction=fraction,
                        lateral=x,
                        heading=math.radians(yaw),
                    )
                    frame = env.reset(**start)["rgb"]
                    distances = np.mean((centers - features(frame)) ** 2, axis=1)
                    go = bool(distances[1] < distances[0])
                    policy = drive(env, door_y, go)
                    twin = env.reset(**start)["rgb"]
                    if not np.array_equal(frame, twin):
                        raise ValueError(
                            "The paired robot starts produced different observations"
                        )
                    forward = drive(env, door_y, True)
                    cases.append(
                        {
                            "layout": layout,
                            "door_fraction": fraction,
                            "lateral_m": x,
                            "heading_deg": yaw,
                            "initial_rgb_pixels_sha256": hashlib.sha256(
                                frame.tobytes()
                            ).hexdigest(),
                            "selected_action": "go" if go else "stop",
                            "policy": policy,
                            "always_forward": forward,
                        }
                    )
        finally:
            env.close()
    return {
        "policy_sha256": hashlib.sha256(weights_file.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule": "Cross 30 cm past the fixed door without touching it along this straight approach. Each stopped policy has an always-forward physical twin.",
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.package)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
