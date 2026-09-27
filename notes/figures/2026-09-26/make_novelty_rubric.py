"""Render the novelty-check rubric matrix used in notes/2026-09-26-novelty-check.md.

The values are the literature classifications from docs/NOVELTY_CHECK.md
(2 = yes, 1 = partial, 0 = no / not described). They are judgments from reading
the papers, not measurements.

Run from the repo root:
    uv run --no-project --with matplotlib python notes/figures/2026-09-26/make_novelty_rubric.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

OUT = Path(__file__).with_name("novelty_rubric.png")

COLUMNS = [
    "(a) view choice\nfor one object",
    "(b) stopping /\nabstention",
    "(c) occluded\nvs removed",
    "(d) learned vs\ngeometric, matched",
    "(e) false-removal\nerrors reported",
]

# (label, in_survey, [a, b, c, d, e]); order = planned study, then ranked threats.
ROWS = [
    ("This study (planned)", None, [2, 2, 2, 2, 2]),
    ("PInVerify 2026 (sim)", False, [2, 2, 1, 1, 1]),
    ("PBD-AG 2026 (sim)", False, [1, 1, 2, 0, 2]),
    ("Li et al. 2021 (sim)", False, [2, 2, 2, 0, 0]),
    ("INSPECT 2026 (real imgs)", False, [2, 2, 1, 1, 1]),
    ("Atanasov et al. 2013", False, [2, 2, 1, 0, 1]),
    ("[10] Bogenberger 2025", True, [0, 1, 1, 0, 1]),
    ("[7] DynaMem 2024", True, [0, 1, 1, 0, 1]),
    ("VAP-TAMP 2026", False, [2, 2, 1, 1, 0]),
    ("[11] Terashima 2023", True, [1, 0, 0, 0, 0]),
    ("[12] SceneDiff 2025", True, [0, 0, 2, 0, 1]),
]

# Reference palette (dataviz skill): sequential blue, ordinal steps 250 and 550;
# validated with validate_palette.js --ordinal (all checks pass on #fcfcfb).
SURFACE = "#fcfcfb"
PARTIAL = "#86b6ef"
YES = "#1c5cab"
RULE = "#d9d8d3"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"

FILL = {2: YES, 1: PARTIAL, 0: SURFACE}
TEXT = {2: "Yes", 1: "Partial", 0: "No"}
TEXT_COLOR = {2: "#ffffff", 1: INK, 0: INK_MUTED}


def main() -> None:
    n_rows, n_cols = len(ROWS), len(COLUMNS)
    fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    gap = 0.06  # surface gap between cells
    for r, (_, _, vals) in enumerate(ROWS):
        y = n_rows - 1 - r
        for c, v in enumerate(vals):
            ax.add_patch(
                Rectangle(
                    (c + gap / 2, y + gap / 2),
                    1 - gap,
                    1 - gap,
                    facecolor=FILL[v],
                    edgecolor=RULE if v == 0 else FILL[v],
                    linewidth=0.8,
                )
            )
            ax.text(c + 0.5, y + 0.5, TEXT[v], ha="center", va="center",
                    fontsize=8.5, color=TEXT_COLOR[v])

    # Separator under the planned-study row.
    ax.plot([0, n_cols], [n_rows - 1, n_rows - 1], color=INK_2, linewidth=1.0)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([c + 0.5 for c in range(n_cols)])
    ax.set_xticklabels(COLUMNS, fontsize=8.5, color=INK_2)
    ax.xaxis.tick_top()
    ax.set_yticks([n_rows - 1 - r + 0.5 for r in range(n_rows)])
    labels = []
    for label, in_survey, _ in ROWS:
        if in_survey is False:
            label += "  *"
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    fig.suptitle(
        "No work found has all five properties of the planned comparison",
        x=0.02, y=0.985, ha="left", fontsize=12, color=INK, fontweight="bold",
    )
    fig.text(
        0.02, 0.935,
        "Literature classification from docs/NOVELTY_CHECK.md (search date 2026-09-26). "
        "Judgments from reading papers, not measurements.\n"
        "Rows below the line are ordered by closeness. * = not in the proposal's survey. Years are first arXiv versions.",
        ha="left", va="top", fontsize=8, color=INK_2,
    )
    fig.legend(
        handles=[
            Patch(facecolor=YES, edgecolor=YES, label="Yes"),
            Patch(facecolor=PARTIAL, edgecolor=PARTIAL, label="Partial"),
            Patch(facecolor=SURFACE, edgecolor=RULE, label="No / not described"),
        ],
        loc="lower left", bbox_to_anchor=(0.02, 0.005), ncol=3, frameon=False,
        fontsize=8.5, labelcolor=INK_2,
    )
    fig.subplots_adjust(left=0.30, right=0.98, top=0.78, bottom=0.07)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
