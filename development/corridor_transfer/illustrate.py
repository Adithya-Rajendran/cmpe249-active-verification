"""Show the same preselected corridor layouts in both simulators, with the recorded decisions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def draw_camera_comparison(inventory, cases, scores, destination):
    """Use the same first declared corridor for all three comparisons."""
    fig, axes = plt.subplots(3, 3, figsize=(11.8, 7.7), dpi=145)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.075, top=0.86, wspace=0.035, hspace=0.30)
    fig.text(0.022, 0.958, "One closed door, three camera angles", size=21, weight="bold", color="#233740")
    fig.text(0.022, 0.91, "MuJoCo · corridor COR01 · bleach bottle", size=12, color="#374953")
    views = [("P1", "Oblique"), ("P2", "Edge leveled"), ("P3", "Straight ahead")]
    states = [
        ("reference", "Earlier view"),
        ("placed_present", "Door closed · bottle kept"),
        ("placed_removed", "Door closed · bottle removed"),
    ]
    sources = []
    for row, (view, label) in enumerate(views):
        first = axes[row, 0]
        first.annotate(f"{label}\n{view}", (0, 0.5), xycoords="axes fraction", xytext=(-11, 0),
                       textcoords="offset points", ha="right", va="center", fontsize=11, color="#273b46")
        for col, (state, heading) in enumerate(states):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for edge in ax.spines.values():
                edge.set_visible(False)
            if row == 0:
                ax.set_title(heading, fontsize=11, weight="bold", loc="left", pad=8)
            kept = state == "placed_present"
            reference = state == "reference"
            result = scores["mujoco", "COR01", view, "placed_present" if reference else state]
            case = cases[result["id"]]
            role = "reference" if reference else "query"
            with Image.open(inventory.parent / case[role]) as image:
                ax.imshow(image.convert("RGB"), interpolation="lanczos")
            color = "#334955"
            if reference:
                x0, y0, x1, y1 = case["box_xyxy"]
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#ffc928", linewidth=1.8))
                caption = "Remembered bottle"
            elif result["call_type"] == "registration_gate_unknown":
                caption = "UNKNOWN · geometry check stopped the call"
            else:
                caption = "Qwen answered " + result["answer"]
                if kept and result["answer"] == "REMOVED":
                    caption += " · false removal"
                    color = "#a63220"
            ax.set_xlabel(caption, fontsize=8.6, labelpad=4, color=color)
            sources.append({"view": view, "state": state, "id": case["id"],
                            "image_sha256": case[role + "_sha256"], "caption": caption})
    fig.text(0.022, 0.025, "Within each row, the two closed-door inputs match pixel for pixel at native and model resolution.",
             size=10, color="#374953")
    fig.savefig(destination / "corridor-angle-example.png", facecolor="white")
    plt.close(fig)
    (destination / "illustrated-angles.json").write_text(json.dumps({
        "selection": "All views of COR01, the first declared layout, in the primary simulator",
        "cases": sources,
    }, indent=2) + "\n")


def draw(run, tables, destination):
    receipt = json.loads((run / "evaluation/scorer-ready.json").read_text())
    inventory = Path(receipt["cases_path"])
    cases = {r["id"]: r for r in json.loads(inventory.read_text())}
    with (tables / "decisions.csv").open(newline="") as stream:
        scores = {
            (r["engine"], r["arrangement"], r["stop"], r["state"]): r
            for r in csv.DictReader(stream)
            if r["method"] == "vggt_qwen"
        }
    selected = []
    names = {"isaac": "Isaac Sim", "mujoco": "MuJoCo"}
    choices = [("COR01", "Bleach bottle"), ("COR02", "Power drill")]
    states = [
        ("reference", "Reference"),
        ("clear_removed", "Door open · target removed"),
        ("placed_present", "Door closed · target kept"),
        ("placed_removed", "Door closed · target removed"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(14, 9.1), dpi=135)
    fig.subplots_adjust(
        left=0.115, right=0.99, bottom=0.035, top=0.894, wspace=0.026, hspace=0.26
    )
    fig.suptitle(
        "The same doorway with the target kept or removed",
        x=0.02,
        y=0.986,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#21343b",
    )
    fig.text(
        0.02,
        0.934,
        "Oblique revisit, 6 cm and 3°. Labels below the images give the registered Qwen system’s answer.",
        fontsize=11,
        color="#374953",
    )
    for index, (layout, target_name) in enumerate(choices):
        for engine_id, engine in enumerate(("isaac", "mujoco")):
            row = 2 * index + engine_id
            first_ax = axes[row, 0]
            first_ax.annotate(
                f"{target_name}\n{names[engine]}\n{layout}",
                (0, 0.5),
                xycoords="axes fraction",
                xytext=(-8, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=10,
                color="#273b46",
                linespacing=1.45,
            )
            for col, (state, title) in enumerate(states):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                result = scores[
                    engine,
                    layout,
                    "P1",
                    "clear_present" if state == "reference" else state,
                ]
                item = cases[result["id"]]
                role = "reference" if state == "reference" else "query"
                with Image.open(inventory.parent / item[role]) as im:
                    rgb = im.convert("RGB")
                ax.imshow(rgb, interpolation="lanczos")
                if row == 0:
                    ax.set_title(
                        title, fontsize=10, fontweight="bold", loc="left", pad=7
                    )
                if state == "reference":
                    x0, y0, x1, y1 = item["box_xyxy"]
                    ax.add_patch(
                        Rectangle(
                            (x0, y0),
                            x1 - x0,
                            y1 - y0,
                            fill=False,
                            edgecolor="#ffc928",
                            linewidth=1.5,
                        )
                    )
                    padding = max(7, int(max(x1 - x0, y1 - y0) * 0.18))
                    inset = ax.inset_axes([0.015, 0.55, 0.34, 0.43])
                    inset.imshow(
                        rgb.crop(
                            (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
                        )
                    )
                    inset.set_xticks([])
                    inset.set_yticks([])
                    for edge in inset.spines.values():
                        edge.set_color("#ffc928")
                        edge.set_linewidth(0.8)
                    caption = "Remembered object"
                else:
                    caption = result["answer"]
                    if result["call_type"] == "registration_gate_unknown":
                        caption += " · geometry check"
                    elif result["call_type"] == "qwen_unknown":
                        caption += " · model abstained"
                    if result["answer"] == "REMOVED" and state.endswith("present"):
                        caption += " · false removal"
                color = "#a63220" if "false removal" in caption else "#32434b"
                ax.set_xlabel(caption, fontsize=9, color=color, labelpad=3)
                selected.append(
                    {
                        "engine": engine,
                        "layout": layout,
                        "view": "P1",
                        "state": state,
                        "id": item["id"],
                        "image_sha256": item[role + "_sha256"],
                        "display": caption,
                    }
                )
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "corridor-examples.png", facecolor="white")
    plt.close(fig)
    (destination / "illustrated-cases.json").write_text(
        json.dumps(
            {
                "selection": "First fixed layout of each target identity, both engines, P1",
                "cases": selected,
            },
            indent=2,
        )
        + "\n"
    )
    draw_camera_comparison(inventory, cases, scores, destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("tables", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    draw(args.run, args.tables, args.destination)
