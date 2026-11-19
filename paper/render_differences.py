"""Measure matched image change and rerender variation on fully hidden calibration pairs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RESAMPLE = getattr(Image, "Resampling", Image)


def pixels(path):
    with Image.open(path) as source:
        return np.asarray(
            source.convert("RGB").resize((1200, 675), RESAMPLE.LANCZOS), dtype=np.int16
        )


def delta(a, b):
    return float(np.abs(a - b).mean())


def collect(root, engine):
    with (HERE / "evidence/arrangements.csv").open(newline="") as f:
        layouts = [r for r in csv.DictReader(f) if r["split"] == "calibration"]
    rows = []
    for r in layouts:
        folder = root / r["arrangement"] / "D0"
        for stop in ("P1", "P2", "P3", "P4", "P5"):
            with Image.open(
                folder / "masks" / f"target_placed_present_{stop}.png"
            ) as im:
                if np.any(np.array(im.convert("RGB"))):
                    continue
            base, rep = {}, {}
            for state in ("present", "removed"):
                base[state] = pixels(folder / "rgb" / f"placed_{state}_{stop}.png")
                second = pixels(folder / "rgb" / f"repeat2_placed_{state}_{stop}.png")
                rep[state] = delta(base[state], second)
            rows.append(
                {
                    "engine": engine,
                    "environment": r["environment"],
                    "arrangement": r["arrangement"],
                    "stop": stop,
                    "state_change": delta(base["present"], base["removed"]),
                    "retained_second_repeat": rep["present"],
                    "removed_second_repeat": rep["removed"],
                    "mean_repeat": (rep["present"] + rep["removed"]) / 2,
                }
            )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine", choices=("isaac", "mujoco"))
    p.add_argument("calibration_directory", type=Path)
    a = p.parse_args()
    print(json.dumps(collect(a.calibration_directory, a.engine), indent=2))
