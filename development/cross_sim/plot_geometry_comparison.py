"""Plot paired counts for the clear-removal condition specified before scoring."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def plot(result_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    summary = json.loads((result_dir / "summary.json").read_bytes())
    source = result_dir / "arrangements.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != summary["output_sha256"][source.name]:
        raise ValueError("Changed arrangement counts")
    with source.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r["environment"], r["level"], r["state"]) ==
                ("tabletop", "D6", "clear_removed")]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8.6, 4.2))
    blue, orange = "#205d80", "#ad5a16"
    for ax, (engine, label) in zip(axes, (("isaac", "Isaac Sim"), ("mujoco", "MuJoCo"))):
        selected = sorted((r for r in rows if r["engine"] == engine), key=lambda r: r["base_id"])
        if len(selected) != 5 or any(int(r["cases"]) != 5 for r in selected):
            raise ValueError("Expected five stops for each of five tabletop arrangements")
        baseline = [int(r["baseline_correct"]) for r in selected]
        candidate = [int(r["candidate_correct"]) for r in selected]
        for y, (before, after) in enumerate(zip(baseline, candidate)):
            ax.plot([before, after], [y, y], color="#8c969a", linewidth=2, zorder=2)
        ax.scatter(baseline, range(5), s=105, facecolor="white", edgecolor=blue, linewidth=2.2, zorder=3)
        ax.scatter(candidate, range(5), s=38, color=orange, marker="D", zorder=4)
        ax.set_title(f"{label}    {sum(baseline)}/25 → {sum(candidate)}/25", loc="left", fontsize=12)
        ax.set_xlim(-0.35, 5.35)
        ax.set_ylim(4.55, -0.55)
        ax.set_xticks(range(6))
        ax.set_yticks(range(5))
        ax.set_yticklabels([r["base_id"] for r in selected], fontsize=10)
        ax.tick_params(axis="both", length=0, pad=5)
        ax.grid(axis="x", color="#dce1e4", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.77, wspace=0.18)
    fig.suptitle("Correct clear removals after a 6 cm and 3° camera shift", x=0.04,
                 y=0.96, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.04, 0.865, "Each row is one tabletop arrangement viewed from five camera stops.", fontsize=10)
    handles = [Line2D([], [], marker="o", markersize=9, markerfacecolor="white", markeredgecolor=blue,
                      markeredgewidth=2, linestyle="none", label="200-pixel baseline"),
               Line2D([], [], marker="D", markersize=6, color=orange, linestyle="none", label="RGB geometry")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.result_dir, args.output)
