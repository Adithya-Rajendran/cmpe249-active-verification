"""Count decisions by the retained twin's visibility and draw the descriptive figure."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
BANDS = ("0%", "(0%, 10%)", "[10%, 50%)", "[50%, 90%)", "[90%, 99.95%)", "≥99.95%")
METHODS = {"homography": "Homography", "dense": "Dense geometry", "registered": "Qwen (diagnostic)"}
STATES = ("PRESENT", "REMOVED")
COLORS = ("#087969", "#a94125", "#dee4e6")


def band_for(retained):
    visible = int(retained["visible_isolated_overlap_pixels"])
    isolated = int(retained["isolated_pixels"])
    if isolated <= 0 or not 0 <= visible <= isolated:
        raise ValueError(f"Invalid native silhouette for {retained['id']}")
    if visible == 0:
        return BANDS[0]
    for name, limit in zip(BANDS[1:-1], (1000, 5000, 9000, 9995)):
        if visible * 10000 < limit * isolated:
            return name
    return BANDS[-1]


def count_answers():
    with (EVIDENCE / "queries.csv").open(newline="") as src:
        queries = list(csv.DictReader(src))
    by_id = {row["id"]: row for row in queries}
    if len(by_id) != len(queries):
        raise ValueError("Duplicate query identifier")
    counts = defaultdict(Counter)
    layouts = defaultdict(set)
    used = set()
    with (EVIDENCE / "pairs.csv").open(newline="") as src:
        for pair in csv.DictReader(src):
            band = band_for(by_id[pair["present_id"]])
            for field, truth in zip(("present_id", "removed_id"), STATES):
                row = by_id[pair[field]]
                if row["id"] in used or row["truth"] != truth or row["engine"] != pair["engine"]:
                    raise ValueError(f"Inconsistent pair member {row['id']}")
                used.add(row["id"])
                for engine in (pair["engine"], "pooled"):
                    layouts[engine, band].add(row["arrangement"])
                    for method in METHODS:
                        decision = row[method]
                        if decision not in (*STATES, "UNKNOWN"):
                            raise ValueError(f"Unexpected decision {decision}")
                        counts[engine, band, truth, method][decision] += 1
    if used != by_id.keys():
        raise ValueError("Some queries have no matched twin")
    result = []
    for engine in ("isaac", "mujoco", "pooled"):
        for band in BANDS:
            for truth in STATES:
                for method in METHODS:
                    answers = counts[engine, band, truth, method]
                    result.append(dict(engine=engine, visible_percent_band=band, truth=truth,
                                       verifier=method, queries=sum(answers.values()),
                                       layouts=len(layouts[engine, band]), present=answers["PRESENT"],
                                       removed=answers["REMOVED"], unknown=answers["UNKNOWN"]))
    return result


def draw(rows):
    pooled = {(r["visible_percent_band"], r["truth"], r["verifier"]): r
              for r in rows if r["engine"] == "pooled"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.235, right=0.967, top=0.86, bottom=0.20, wspace=0.20, hspace=0.43)
    for i, truth in enumerate(STATES):
        other = "removed" if truth == "PRESENT" else "present"
        for j, (method, title) in enumerate(METHODS.items()):
            ax = axes[i, j]
            for y, band in enumerate(BANDS):
                row = pooled[band, truth, method]
                left = 0
                for field, color in zip((truth.lower(), other, "unknown"), COLORS):
                    width = 100 * row[field] / row["queries"] if row["queries"] else 0
                    ax.barh(y, width, left=left, height=0.73, color=color, linewidth=0)
                    left += width
            if i == 0:
                ax.set_title(title, fontweight="bold", color="#172b35", fontsize=9)
            ax.set_xlim(0, 100)
            ax.set_ylim(len(BANDS) - 0.5, -0.5)
            ax.set_xticks([0, 50, 100])
            ax.tick_params(length=0, labelsize=7.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
        labels = [f"{band}    {pooled[band, truth, 'dense']['queries']:3d}" for band in BANDS]
        axes[i, 0].set_yticks(range(len(BANDS)), labels)
        y = axes[i, 0].get_position().y1
        fig.text(0.008, y + 0.012, ("Target retained", "Target removed")[i],
                 fontsize=8.7, color="#172b35", weight="bold")
    fig.text(0.6, 0.103, "Share of queries (%)", ha="center", fontsize=8, color="#172b35")
    fig.legend(handles=[Patch(facecolor=c, label=s) for c, s in zip(COLORS, ("Correct", "Wrong", "Unknown"))],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.savefig(ROOT / "figures/answers-by-visibility.png", dpi=250, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    rows = count_answers()
    with (EVIDENCE / "answers-by-visibility.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    draw(rows)
