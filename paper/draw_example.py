"""Render the published case illustration from the included, unmarked RGB inputs."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "evidence/illustration"
DEST = HERE / "figures/counterfactual-example.png"


def image_at(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def main():
    cases = json.loads((DATA / "cases.json").read_text())["rows"]
    fig, axs = plt.subplots(2, 3, figsize=(7.3, 2.82))
    fig.subplots_adjust(
        left=0.084, right=0.995, top=0.888, bottom=0.048, hspace=0.10, wspace=0.035
    )
    headings = [
        "Earlier reference",
        "Revisit · target retained",
        "Revisit · target removed",
    ]
    for col, title in enumerate(headings):
        axs[0, col].set_title(title, fontsize=9.3, pad=4, color="#172936")
    for row, case in enumerate(cases):
        prefix = case["setting"]
        for entry in case["files"]:
            src = DATA / entry["file"]
            if hashlib.sha256(src.read_bytes()).hexdigest() != entry["file_sha256"]:
                raise ValueError(f"Image file changed: {src.name}")
            if (
                hashlib.sha256(image_at(src).tobytes()).hexdigest()
                != entry["decoded_sha256"]
            ):
                raise ValueError(f"Decoded image changed: {src.name}")
        pa = image_at(DATA / f"{prefix}_present.png")
        pb = image_at(DATA / f"{prefix}_removed.png")
        diff = int(np.any(pa != pb, axis=2).sum())
        if diff != case["revisit_pixels_changed_in_full_1200x675_rgb"]:
            raise ValueError("Image equality evidence changed")
        left, top, right, bottom = case["crop_xyxy"]
        ims = [image_at(DATA / f"{prefix}-reference.png"), pa, pb]
        for col, (axis, im) in enumerate(zip(axs[row], ims)):
            axis.imshow(
                im[top:bottom, left:right], interpolation="nearest", aspect="equal"
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#bac2c2")
                spine.set_linewidth(0.6)
            if col == 0:
                x0, y0, x1, y1 = case["reference_box_xyxy"]
                axis.add_patch(
                    Rectangle(
                        (x0 - left, y0 - top),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.8,
                    )
                )
                axis.add_patch(
                    Rectangle(
                        (x0 - left, y0 - top),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="#ffde54",
                        linewidth=1.65,
                    )
                )
        name = "Clear\ntable" if prefix == "clear" else "Blocked\nshelf"
        axs[row, 0].annotate(
            name,
            (0, 0.5),
            xycoords="axes fraction",
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.4,
        )
    fig.savefig(DEST, dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
