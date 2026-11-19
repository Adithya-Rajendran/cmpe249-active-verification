"""Select and draw one fresh matched example using the frozen display rule."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "evidence"
METHODS = ("homography", "dense", "registered")
NAMES = ("Homography", "Dense geometry", "Registered VLM")


def pairs():
    with (DATA / "queries.csv").open(newline="") as f:
        query = {r["id"]: r for r in csv.DictReader(f)}
    with (DATA / "pairs.csv").open(newline="") as f:
        return [
            (p, query[p["present_id"]], query[p["removed_id"]])
            for p in csv.DictReader(f)
        ]


def rank(entry):
    p, present, removed = entry
    certified = p["exact"] == "1"
    commits = [present[m] != "UNKNOWN" or removed[m] != "UNKNOWN" for m in METHODS]
    if certified and any(commits) and not all(commits):
        tier = 0
    elif certified and any(commits):
        tier = 1
    elif any(present[m] == "REMOVED" for m in METHODS):
        tier = 2
    elif (
        p["level"] == "D6"
        and p["placement"] == "clear"
        and len({removed[m] for m in METHODS}) > 1
    ):
        tier = 3
    elif certified:
        tier = 4
    else:
        tier = 5
    return (
        tier,
        p["level"] != "D6",
        p["engine"],
        p["arrangement"],
        p["stop"],
        p["placement"],
    )


def main(archive):
    entry = min(pairs(), key=rank)
    p, a, b = entry
    reference = tuple(json.loads(a["box"]))
    left, top, right, bottom = reference
    # Identical image coordinates for each panel. Padding accommodates a moved camera.
    crop = (
        max(0, left - 145),
        max(0, top - 80),
        min(1200, right + 145),
        min(675, bottom + 80),
    )
    digests = (
        a["reference_file_sha256"],
        a["query_file_sha256"],
        b["query_file_sha256"],
    )
    with zipfile.ZipFile(archive) as z:

        def read(digest):
            raw = z.read("rgb/" + digest + ".png")
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Digest mismatch")
            with Image.open(io.BytesIO(raw)) as source:
                return source.convert("RGB").crop(crop)

        imgs = [read(d) for d in digests]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.19, wspace=0.05)
    titles = ("Earlier reference", "Target retained", "Target removed")
    for i, (ax, im, title) in enumerate(zip(axes, imgs, titles)):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(title, fontsize=9, pad=2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.add_patch(
                Rectangle(
                    (left - crop[0], top - crop[1]),
                    right - left,
                    bottom - top,
                    fill=False,
                    edgecolor="#ffdf5a",
                    lw=1.55,
                )
            )
        for spine in ax.spines.values():
            spine.set_color("#9dabad")
            spine.set_linewidth(0.5)
    fig.text(
        0.5,
        0.04,
        f"{p['arrangement']}   ·   {p['engine']}   ·   {p['level']} / {p['stop']}   ·   {p['placement']} blocker",
        ha="center",
        fontsize=8,
    )
    fig.savefig(HERE / "figures/fresh-matched-pair.png", dpi=220, facecolor="white")
    plt.close(fig)
    record = {
        "selection_rank": rank(entry)[0],
        "pair": p,
        "target": a["target"],
        "crop_xyxy": crop,
        "input_file_sha256": digests,
        "answers": {m: {"retained": a[m], "removed": b[m]} for m in METHODS},
    }
    (DATA / "fresh-illustration.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_archive", type=Path)
    main(parser.parse_args().image_archive)
