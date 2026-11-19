"""Show the three camera treatments for the prespecified shelf example using captured RGB."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


VIEWS = [("P4", "Original view"), ("P5", "Same viewpoint,\nedge leveled"), ("P3", "Straight on")]


def make(run, output, engine):
    root = run / "operator" / engine / "PAM01"
    angles = json.loads((run / "cohort/edge-angles.json").read_text())
    angle = {r["slot"]: r["front_edge_deg"] for r in angles if
             (r["layout"], r["level"], r["camera"]) == ("PAM01", "D6", "revisit")}
    fig, axes = plt.subplots(3, 3, figsize=(10.1, 5.25))
    fig.subplots_adjust(left=.158, right=.992, bottom=.036, top=.92, wspace=.026, hspace=.085)
    for row, (slot, name) in enumerate(VIEWS):
        with Image.open(root / "D0/masks" / f"target_reference_{slot}.png") as image:
            box = image.getbbox()
            width, height = image.size
        with Image.open(root / "D6/masks" / f"blocker_placed_present_{slot}.png") as image:
            panel = image.getbbox()
        middle_x = (min(box[0], panel[0]) + max(box[2], panel[2])) / 2
        middle_y = (min(box[1], panel[1]) + max(box[3], panel[3])) / 2
        left = max(0, min(width - 760, round(middle_x - 380)))
        top = max(0, min(height - 360, round(middle_y - 180)))
        sources = [("D0", "reference"), ("D6", "placed_present"), ("D6", "placed_removed")]
        for col, (level, state) in enumerate(sources):
            axis = axes[row, col]
            with Image.open(root / level / "rgb" / f"{state}_{slot}.png") as im:
                axis.imshow(im.crop((left, top, left + 760, top + 360)), interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_linewidth(.7)
                spine.set_color("#9ba2a4")
            if col == 0:
                x, y, x2, y2 = box
                axis.add_patch(Rectangle((x-left, y-top), x2-x, y2-y, fill=False, edgecolor="#ffdf59", linewidth=1.6))
                axis.text(-.065, .52, f"{name}\n{abs(angle[slot]):.1f}°", transform=axis.transAxes,
                          ha="right", va="center", fontsize=10, linespacing=1.4)
            if row == 0:
                axis.set_title(("Earlier reference", "Revisit · box stayed", "Revisit · box removed")[col],
                               fontsize=11, pad=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=155, facecolor="white", bbox_inches="tight", pad_inches=.09)
    plt.close(fig)


def clear_errors(run, output):
    ledger = json.loads((run / "operator/private/ledger.json").read_text())
    cases = {row["id"]: row for row in ledger["rows"]}
    missed = []
    with (run / "scoring/vggt_qwen/records.jsonl").open() as stream:
        for line in stream:
            score = json.loads(line)
            row = cases[score["id"]]
            if row["stop"] == "P3" and row["state"] == "clear_present" and score["answer"] == "REMOVED":
                missed.append(row)
    if not missed:
        raise ValueError("There are no frontal clear false removals")
    fig, panels = plt.subplots(len(missed), 2, squeeze=False, figsize=(8.1, 1.7 * len(missed) + .35))
    fig.subplots_adjust(left=.12, right=.99, bottom=.04, top=.90, hspace=.11, wspace=.03)
    for axes, row in zip(panels, sorted(missed, key=lambda r: (r["engine"], r["arrangement"]))):
        files = row["files"]
        with Image.open(files["reference_mask"]["path"]) as mask:
            box = mask.getbbox()
            width, height = mask.size
        with Image.open(files["query_mask"]["path"]) as mask:
            query_box = mask.getbbox()
        cx = (min(box[0], query_box[0]) + max(box[2], query_box[2])) // 2
        cy = (min(box[1], query_box[1]) + max(box[3], query_box[3])) // 2
        left, top = max(0, min(cx - 230, width - 460)), max(0, min(cy - 100, height - 200))
        for axis, role in zip(axes, ("reference", "query")):
            with Image.open(files[role]["path"]) as im:
                axis.imshow(im.crop((left, top, left + 460, top + 200)), interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_linewidth(.6)
                spine.set_color("#9ba2a4")
        axes[0].add_patch(Rectangle((box[0]-left, box[1]-top), box[2]-box[0], box[3]-box[1],
                                    fill=False, edgecolor="#ffdf59", linewidth=1.55))
        engine = {"isaac": "Isaac Sim", "mujoco": "MuJoCo"}[row["engine"]]
        axes[0].text(-.05, .5, engine + "\n" + row["arrangement"], transform=axes[0].transAxes,
                     ha="right", va="center", fontsize=9)
    panels[0, 0].set_title("Earlier reference", fontsize=11)
    panels[0, 1].set_title("Clear revisit · marker stayed", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=.1)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--engine", choices=["isaac", "mujoco"], default="mujoco")
    parser.add_argument("--clear-errors", action="store_true")
    args = parser.parse_args()
    if args.clear_errors:
        clear_errors(args.run, args.output)
    else:
        make(args.run, args.output, args.engine)
