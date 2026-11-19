"""Measure native RGB changes across both physical states before seeing verifier outputs."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from development.corridor_transfer.prepare import LAYOUTS, STOPS


def image(path, mode):
    with Image.open(path) as source:
        return np.asarray(source.convert(mode))


def compare(root, engine):
    rows = []
    for layout in LAYOUTS:
        folder = root / "operator" / engine / layout / "D6"
        for stop in STOPS:
            for placement in ("clear", "placed"):
                files = [folder / "rgb" / f"{placement}_{state}_{stop}.png" for state in ("present", "removed")]
                retained, removed = (image(p, "RGB").astype(np.int16) for p in files)
                difference = np.abs(retained - removed)
                visible = image(folder/"masks"/f"target_{placement}_present_{stop}.png", "L") > 0
                changed = np.any(difference != 0, axis=2)
                rows.append({"engine": engine, "layout": layout, "view": stop, "door": placement,
                             "native_revisit_pixels": changed.size, "visible_target_pixels": int(visible.sum()),
                             "changed_pixels": int(changed.sum()), "changed_fraction": float(changed.mean()),
                             "mean_abs_difference_8bit": float(difference.mean()),
                             "max_channel_difference_8bit": int(difference.max())})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("engine", choices=("mujoco","isaac"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = compare(args.run, args.engine)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    closed = [r for r in rows if r["door"] == "placed"]
    print(args.engine, len(rows), "native query pairs,", sum(r["changed_pixels"] == 0 for r in closed),
          "pixel-identical closed-door revisits out of", len(closed))
