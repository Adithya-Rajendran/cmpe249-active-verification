"""Illustrate the strongest repeatable calibration tile from Isaac Sim (PAC04, P3)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent
# Top-left native pixel from the base-pair search in hidden_pair_pixels.py.
TILE_X, TILE_Y, TILE_SIZE = 1360, 448, 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isaac_calibration", type=Path, help="Extracted isaac/calibration directory")
    args = parser.parse_args()
    folder = args.isaac_calibration / "PAC04/D0/rgb"
    states = ["present", "removed"]
    images = []
    for state in states:
        with Image.open(folder / f"placed_{state}_P3.png") as image:
            images.append(np.asarray(image.convert("RGB")))

    left, top = TILE_X - 88, TILE_Y - 46
    width, height = 192, 108
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 1.9))
    fig.subplots_adjust(left=.005, right=.995, bottom=.01, top=.83, wspace=.04)
    axes[0].imshow(images[0])
    axes[0].add_patch(Rectangle((left - .5, top - .5), width, height, fill=False,
                                edgecolor="#00efff", linewidth=1.1))
    axes[0].set_title("Isaac Sim tabletop", fontsize=10)

    for axis, image, label in zip(axes[1:], images, ["Target retained", "Target removed"]):
        tile = image[TILE_Y:TILE_Y + TILE_SIZE, TILE_X:TILE_X + TILE_SIZE]
        axis.imshow(image[top:top + height, left:left + width], interpolation="nearest")
        axis.add_patch(Rectangle((TILE_X - left - .5, TILE_Y - top - .5), TILE_SIZE,
                                 TILE_SIZE, fill=False, edgecolor="#00efff", linewidth=1.1))
        axis.set_title(f"{label} (mean {tile.mean():.1f})", fontsize=10)
    for axis in axes:
        axis.set_axis_off()
    fig.savefig(ROOT / "figures/isaac-hidden-patch.png", dpi=200, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
