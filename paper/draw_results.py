"""Draw primary resolution by environment and target from each delivered query."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
SPECS = [
    ("homography", "Homography", "#354852", "o", -0.17),
    ("dense", "Dense geometry", "#008378", "s", 0.0),
    ("registered", "Registered VLM", "#965400", "^", 0.17),
]


def main():
    with (ROOT / "evidence/queries.csv").open(newline="") as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r["level"] == "D6" and r["state"] == "clear_removed"]
    envs = ("shelf", "tabletop")
    objects = ("gelatin_box", "large_marker")
    factors = [(env, obj) for env in envs for obj in objects]
    labels = [
        "Shelf · gelatin box",
        "Shelf · marker",
        "Table · gelatin box",
        "Table · marker",
    ]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.15, 2.65))
    fig.subplots_adjust(left=0.17, right=0.974, top=0.83, bottom=0.24, wspace=0.14)
    for ax, engine, name in zip(axes, ("isaac", "mujoco"), ("Isaac Sim", "MuJoCo")):
        ax.set_title(name, loc="left", fontweight="bold", fontsize=10)
        for y, (env, obj) in enumerate(factors):
            group = [
                r
                for r in rows
                if (r["engine"], r["environment"], r["target"]) == (engine, env, obj)
            ]
            if len(group) != 20:
                raise ValueError(
                    f"Expected twenty queries in {engine}, {env}, {obj}; found {len(group)}"
                )
            values = [
                sum(r[method] == "REMOVED" for r in group) for method, *_ in SPECS
            ]
            ax.plot(
                [min(values), max(values)], [y, y], lw=1.2, color="#aab7b8", zorder=1
            )
            for x, (_, _, color, shape, offset) in zip(values, SPECS):
                ax.scatter(
                    x,
                    y + offset,
                    s=36,
                    marker=shape,
                    linewidth=0.9,
                    edgecolor=color,
                    facecolor="white" if shape == "o" else color,
                    zorder=3,
                )
        ax.set_yticks(range(4), labels)
        ax.set_xlim(-0.5, 20.5)
        ax.set_xticks(range(0, 21, 5))
        ax.set_ylim(3.45, -0.48)
        ax.tick_params(axis="both", length=0, labelsize=8)
        ax.grid(axis="x", color="#dfe5e6", linewidth=0.65, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=shape,
            markeredgecolor=color,
            markerfacecolor="white" if shape == "o" else color,
            markersize=5.8,
            label=name,
        )
        for _, name, color, shape, _ in SPECS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.56, 0.015),
        ncol=3,
        frameon=False,
        fontsize=8.3,
        handletextpad=0.2,
        columnspacing=1.4,
    )
    fig.savefig(ROOT / "figures/primary-by-target.png", dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
