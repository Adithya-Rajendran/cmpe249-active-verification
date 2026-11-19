"""Show the narrow retained-target mask that made the fresh calibration patch ineligible."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "evidence/calibration-illustration"


def main():
    info = json.loads((DATA / "illustration.json").read_text())
    for row in info["files"]:
        if (
            hashlib.sha256((DATA / (row["role"] + ".png")).read_bytes()).hexdigest()
            != row["crop_sha256"]
        ):
            raise ValueError("Cropped figure inputs changed")
    with Image.open(DATA / "mask.png") as im:
        mask = np.asarray(im.convert("L")) > 0
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("Expected a retained target in this example")
    fig, ax = plt.subplots(1, 4, figsize=(7.05, 1.58))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.08, wspace=0.035)
    for panel, role, title in zip(
        ax,
        ("reference", "retained", "removed", "mask"),
        (
            "Earlier reference",
            "Retained, panel placed",
            "Removed, panel placed",
            "Retained target mask",
        ),
    ):
        with Image.open(DATA / (role + ".png")) as source:
            img = source.convert("RGB")
            panel.imshow(
                img if role != "mask" else np.where(mask, 230, 30),
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
        if role == "reference":
            panel.add_patch(
                Rectangle(
                    (1111 - 1055, 908 - 866), 105, 67, ec="#ffdb60", lw=1.45, fill=False
                )
            )
        if role == "mask":
            panel.annotate(
                f"{len(x)} pixels",
                xy=(x.max() + 1, np.median(y)),
                xytext=(0.29, 0.80),
                textcoords="axes fraction",
                color="white",
                fontsize=7,
                ha="center",
                arrowprops={"arrowstyle": "-", "color": "white", "lw": 0.65},
            )
        panel.set_title(title, fontsize=7.8, pad=3.3)
        panel.set_xticks([])
        panel.set_yticks([])
        for spine in panel.spines.values():
            spine.set_edgecolor("#aebbbb")
            spine.set_linewidth(0.45)
    fig.savefig(ROOT / "figures/calibration-sliver.png", dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
