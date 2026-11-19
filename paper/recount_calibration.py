"""Recompute the fixed-site RGB measurement directly from either engine's native images."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
SHAPE = (1200, 675)
NATIVE = (2304, 1296)
RESAMPLE = getattr(Image, "Resampling", Image)


def get_mask(path):
    with Image.open(path) as im:
        if im.size != NATIVE:
            raise ValueError("Unexpected native mask size")
        a = np.asarray(im)
    return np.any(a[:, :, :3] != 0, axis=-1) if a.ndim == 3 else a != 0


def scaled_mask(m):
    return (
        np.asarray(Image.fromarray(m.astype("uint8")).resize(SHAPE, RESAMPLE.NEAREST))
        != 0
    )


def get_rgb(path):
    with Image.open(path) as im:
        if im.size != NATIVE:
            raise ValueError("Unexpected native RGB size")
        return np.array(
            im.convert("RGB").resize(SHAPE, RESAMPLE.LANCZOS), dtype=np.int16
        )


def patch_for(ref, visible, box, center_offset):
    common = scaled_mask(ref) & scaled_mask(visible)
    if not common.any():
        return None
    distance = distance_transform_edt(np.pad(common, ((1, 1), (1, 1))))[1:-1, 1:-1]
    ys, xs = np.where(distance == distance.max())
    x0, y0, x1, y1 = box
    order = np.lexsort(
        (
            xs,
            ys,
            (2 * xs - (x0 + x1 - center_offset)) ** 2
            + (2 * ys - (y0 + y1 - center_offset)) ** 2,
        )
    )
    x, y = int(xs[order[0]]), int(ys[order[0]])
    if min(x - 3, y - 3) < 0 or x + 3 >= SHAPE[0] or y + 3 >= SHAPE[1]:
        return None
    select = common[y - 3 : y + 4, x - 3 : x + 4]
    if int(select.sum()) < 25:
        return None
    yy, xx = np.where(select)
    return yy + y - 3, xx + x - 3


def compute(raw, engine, convention):
    """raw points to calibration/<layout> folders for a single engine."""
    with (HERE / "evidence/arrangements.csv").open() as f:
        layouts = [r for r in csv.DictReader(f) if r["split"] == "calibration"]
    rows = []
    for layout in layouts:
        folder = raw / layout["arrangement"] / "D0"
        for stop in ("P1", "P2", "P3", "P4", "P5"):
            ref = get_mask(folder / "masks" / f"target_reference_{stop}.png")
            ys, xs = ref.nonzero()
            # Exact half-open enclosing rectangle after scale, with integer arithmetic.
            l, r = (
                xs.min() * SHAPE[0] // NATIVE[0],
                -(-(xs.max() + 1) * SHAPE[0] // NATIVE[0]),
            )
            t, b = (
                ys.min() * SHAPE[1] // NATIVE[1],
                -(-(ys.max() + 1) * SHAPE[1] // NATIVE[1]),
            )
            alone = get_mask(folder / "masks" / f"alone_{stop}.png")
            for location in ("clear", "placed"):
                present = get_mask(
                    folder / "masks" / f"target_{location}_present_{stop}.png"
                )
                index = patch_for(
                    ref, present, (l, t, r, b), int(convention == "pixel-centers")
                )
                rec = {
                    "engine": engine,
                    "coordinate_convention": convention,
                    "layout": layout["arrangement"],
                    "environment": layout["environment"],
                    "stop": stop,
                    "location": location,
                    "eligible": index is not None,
                    "fraction": int((present & alone).sum()) / int(alone.sum()),
                    "change": None,
                    "bound": None,
                    "passes": False,
                    "negative_triggers": 0,
                    "tested_negatives": 0,
                }
                if index is not None:
                    patches = {}
                    for v in ("present", "removed"):
                        for kind in ("", "repeat1_", "repeat2_", "distractor_hidden_"):
                            name = f"{kind}{location}_{v}_{stop}.png"
                            patches[v, kind] = get_rgb(folder / "rgb" / name)[index]

                    differences = {
                        (state, kind): float(np.abs(patches[state, ""] - patch).mean())
                        for (state, kind), patch in patches.items()
                        if kind
                    }
                    bound = max(
                        differences[state, "repeat1_"]
                        for state in ("present", "removed")
                    )
                    change = float(
                        np.abs(patches["present", ""] - patches["removed", ""]).mean()
                    )
                    triggers = sum(
                        contrast >= 8 and contrast - bound >= 4
                        for (_, kind), contrast in differences.items()
                        if kind in ("repeat2_", "distractor_hidden_")
                    )
                    rec.update(
                        change=change,
                        bound=bound,
                        passes=change >= 8 and change - bound >= 4,
                        negative_triggers=triggers,
                        tested_negatives=4,
                    )
                rows.append(rec)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine", choices=("isaac", "mujoco"))
    p.add_argument("calibration_directory", type=Path)
    p.add_argument(
        "--convention",
        choices=("numeric-midpoint", "pixel-centers"),
        default="numeric-midpoint",
        help="Tie break at (left+right)/2 or (left+right-1)/2 in array-index coordinates.",
    )
    a = p.parse_args()
    print(
        json.dumps(
            compute(a.calibration_directory, a.engine, a.convention),
            indent=2,
            allow_nan=False,
        )
    )
