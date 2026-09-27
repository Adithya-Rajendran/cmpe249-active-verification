"""Tables and small figures for the 2026-09-26 perception smoke-test notebook entry.

Reads the outputs of smoke_test.py and match_check.py and writes
  <run>/report/*.md|csv                  tables quoted in the notebook entry
  <figs>/perception_overlays.jpg         detector overlays, selected frames x 2 models
  <figs>/perception_conditions.png       target-class confidence per condition x model (heatmap)
  <figs>/perception_defocus.png          focus sweep: mug sharpness, detector confidence, match similarity
  <figs>/match_similarity.png            positives vs background crops; occlusion curves
  <figs>/match_crops.jpg                 example crops with their similarities

Usage (from the repo root):
  uv run python perception/report_smoke.py --run runs/2026-09-26-perception-smoke \
      --manifest perception/manifests/2026-09-26_smoke.json --figs notes/figures/2026-09-26
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, apply_ops, load_entry, load_manifest  # noqa: E402
from smoke_test import draw_overlay, get_font  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# Reference palette (dataviz skill, light mode; validated: slots 1-4 pass adjacent CVD/normal checks,
# aqua/yellow are below 3:1 contrast, so every series is also direct-labelled or in a table).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#9a9893"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MODELS = ["yolov8s-worldv2", "yoloe-26s-seg", "yolo11s", "yolo26s"]
MODEL_LABEL = {"yolov8s-worldv2": "YOLO-World v2 S\n(open vocab)", "yoloe-26s-seg": "YOLOE-26 S\n(open vocab)",
               "yolo11s": "YOLO11 S\n(COCO)", "yolo26s": "YOLO26 S\n(COCO)"}
ENC = ["dinov2-small", "clip-vit-b32"]
ENC_LABEL = {"dinov2-small": "DINOv2-S", "clip-vit-b32": "CLIP B/32"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load_detect(run):
    return [json.loads(line) for line in open(run / "detect" / "detections.jsonl")]


# ----------------------------------------------------------------------------- tables


ROWS = [  # (section, label, image ids or prefix*)
    ("real", "probe 11:28, full frame (mug cut off, AF-defocused, rolled)", ["probe_full"]),
    ("real", "probe 11:28, ROI crop around the mug", ["probe_crop"]),
    ("real", "11:44, 17 lux, lens 6.0 D, upright", ["sweep_best_upright"]),
    ("real", "  same frame, upside down (native sensor)", ["sweep_best_native"]),
    ("real", "  same frame, ROI crop", ["sweep_best_upright_crop"]),
    ("real", "11:41, 52 lux, AF on background (median, n=18)", ["af_*"]),
    ("real", "12:05 pilot, 4 lux, locked lens 6.0 D (median, n=6)",
     ["pilot_PILOT00*", "pilot_calibration"]),
    ("real", "  pilot reference t01, ROI crop", ["pilot_ref_t01_crop"]),
    ("real", "12:16, 4 lux, lens 6.0 D, auto exposure", ["now_lens6_autoexp"]),
    ("real", "12:16, 4 lux, full auto (AF chose 3.0 D)", ["now_fullauto"]),
    ("syn", "pilot ref t01, unmodified", ["pilot_PILOT00_P1_reference_pilot_ambient_t01"]),
    ("syn", "  left 25% of mug cut by frame edge", ["syn_trunc_left_25"]),
    ("syn", "  left 50% cut", ["syn_trunc_left_50"]),
    ("syn", "  left 75% cut", ["syn_trunc_left_75"]),
    ("syn", "  flat dark block over left 25% of mug", ["syn_occl_left_25"]),
    ("syn", "  block over 50%", ["syn_occl_left_50"]),
    ("syn", "  block over 75%", ["syn_occl_left_75"]),
    ("syn", "  Gaussian blur sigma 4 px", ["syn_blur_4"]),
    ("syn", "  blur sigma 8 px", ["syn_blur_8"]),
    ("syn", "  blur sigma 16 px", ["syn_blur_16"]),
    ("syn", "  blur sigma 32 px", ["syn_blur_32"]),
    ("syn", "  rolled 15 deg", ["syn_roll_15"]),
    ("syn", "  rolled 30 deg", ["syn_roll_30"]),
    ("syn", "  rolled 45 deg", ["syn_roll_45"]),
    ("syn", "  rolled 90 deg", ["syn_roll_90"]),
    ("syn", "  rotated 180 deg", ["syn_roll_180"]),
    ("syn", "  x2 digital gain (luma 64 -> ~128)", ["syn_pilot_gain_2x"]),
    ("syn", "17-lux frame x0.5 gain (luma 125 -> ~62)", ["syn_sweep_gain_0.5x"]),
    ("syn", "52-lux AF frame x0.5 gain (luma 131 -> ~65)", ["syn_af_gain_0.5x"]),
]


def match_ids(pattern, ids):
    if pattern.endswith("*"):
        return [i for i in ids if i.startswith(pattern[:-1])]
    return [pattern] if pattern in ids else []


def condition_table(recs):
    by = {(r["image_id"], r["model"]): r for r in recs}
    ids = sorted({r["image_id"] for r in recs})
    table = []
    for sec, label, pats in ROWS:
        sel = [i for p in pats for i in match_ids(p, ids)]
        row = {"section": sec, "label": label, "n": len(sel), "ids": sel}
        for m in MODELS:
            v = [by[(i, m)]["hit_conf"] for i in sel]
            row[m] = statistics.median(v)
            row[m + "_range"] = (min(v), max(v))
            row[m + "_any"] = statistics.median([by[(i, m)]["any_on_mug_conf"] for i in sel])
            row[m + "_top"] = by[(sel[0], m)]["top_on_mug"] if len(sel) == 1 else None
        table.append(row)
    return table


def write_tables(run, recs, match):
    rep = run / "report"
    rep.mkdir(exist_ok=True)
    tab = condition_table(recs)
    lines = ["| Condition | n | " + " | ".join(m for m in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for r in tab:
        cells = []
        for m in MODELS:
            c = f"{r[m]:.2f}"
            if r["n"] > 1:
                c += f" ({r[m + '_range'][0]:.2f}–{r[m + '_range'][1]:.2f})"
            elif r[m + "_top"] and r[m + "_top"].split()[0] not in ("cup", "mug"):
                c += f" [top box on mug: {r[m + '_top']}]"
            cells.append(c)
        lines.append(f"| {r['label'].strip()} | {r['n']} | " + " | ".join(cells) + " |")
    (rep / "conditions.md").write_text("\n".join(lines) + "\n")

    # detection counts over the real, full frames with the whole mug in view
    ids = sorted({r["image_id"] for r in recs})
    light_ok = {}
    for row in csv.DictReader(open(REPO / "data/raw/2026-09-26_camera-check/analysis/focus_sweep_run2.csv")):
        light_ok[f"sweep2_lp_{float(row['lp_req']):04.1f}"] = row["light_ok"] == "True"
    sets = [
        ("pilot + 12:16 frames, 4 lux (lens 6.0 D except one AF frame)", [i for i in ids if i.startswith("pilot_PILOT00") or i == "pilot_calibration" or i.startswith("now_")]),
        ("focus sweep 0-13.5 D, 17 lux (light-drop steps excluded)", [i for i in ids if i.startswith("sweep2_") and light_ok.get(i)]),
        ("  of which in the mug's in-focus band 5.0-7.5 D", [i for i in ids if i.startswith("sweep2_") and light_ok.get(i) and 5.0 <= float(i.split("_lp_")[1]) <= 7.5]),
        ("AF stills, 52 lux (mug defocused)", [i for i in ids if i.startswith("af_")]),
    ]
    by = {(r["image_id"], r["model"]): r for r in recs}
    rl = ["| Frames (whole mug in view, unedited) | n | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    counts = {}
    for lab, sel in sets:
        cells = []
        for m in MODELS:
            h25 = sum(by[(i, m)]["hit_conf"] >= 0.25 for i in sel)
            h10 = sum(by[(i, m)]["hit_conf"] >= 0.10 for i in sel)
            box = sum(by[(i, m)]["any_on_mug_conf"] >= 0.10 for i in sel)
            fp = sum(by[(i, m)]["target_fp_conf"] >= 0.25 for i in sel)
            cells.append(f"{h25} / {h10} / {box} / {fp}")
            counts[f"{lab}|{m}"] = {"n": len(sel), "hit_ge_0.25": h25, "hit_ge_0.10": h10, "any_class_box_on_mug_ge_0.10": box, "fp_target_ge_0.25": fp}
        rl.append(f"| {lab} | {len(sel)} | " + " | ".join(cells) + " |")
    (rep / "detection_counts.md").write_text("\n".join(rl) + "\n\nCell = frames with a target-class box on the mug at conf >= 0.25 / at >= 0.10 / "
                                            "frames with a box of any class on the mug at >= 0.10 / frames with a target-class box off the mug (IoU < 0.1) at >= 0.25.\n")
    (rep / "detection_counts.json").write_text(json.dumps(counts, indent=1))

    # latency
    lat = ["| Model | median of per-image medians (ms) | min | max | n images |", "|---|---|---|---|---|"]
    for m in MODELS:
        v = [r["wall_ms_median"] for r in recs if r["model"] == m]
        lat.append(f"| {m} | {statistics.median(v):.2f} | {min(v):.2f} | {max(v):.2f} | {len(v)} |")
    for e in ENC:
        L = match[e]["latency_ms_single_crop"]
        lat.append(f"| {e} (one 224 px crop, prep + forward) | {L['median']:.2f} | {L['min']:.2f} | – | {L['n']} runs |")
    (rep / "latency.md").write_text("\n".join(lat) + "\n")

    # class used on the mug by YOLO-World: cup vs mug prompt
    cm = {"cup": 0, "mug": 0}
    mug_best = []
    for r in recs:
        if r["model"] == "yolov8s-worldv2" and r["hit_cls"]:
            cm[r["hit_cls"]] += 1
        if r["model"] == "yolov8s-worldv2":
            mug_best.append(max([d["conf"] for d in r["dets"] if d["cls"] == "mug" and (d["iou_gt"] or 0) >= 0.5], default=0.0))
    (rep / "world_cup_vs_mug.json").write_text(json.dumps({"best_hit_class_counts": cm, "n_images": len(mug_best),
                                                           "max_mug_prompt_conf_on_mug": max(mug_best),
                                                           "images_with_mug_prompt_ge_0.10_on_mug": sum(v >= 0.10 for v in mug_best)}, indent=1))

    # occluder prompts (book / bottle / box) on the real, distinct full frames: no such object is in the
    # scene, so any score is spurious. Excludes ROI crops and the two sweep_best copies of sweep2_lp_06.0.
    real = {"pilot", "now", "sweep_best", "sweep_series", "af_bright", "probe"}
    occ = {}
    for m in ("yolov8s-worldv2", "yoloe-26s-seg"):
        sel = [r for r in recs if r["model"] == m and r["group"] in real and not r["image_id"].endswith("_crop")
               and r["image_id"] not in ("sweep_best_native", "sweep_best_upright")]
        confs = [d["conf"] for r in sel for d in r["dets"] if d["cls"] in ("book", "bottle", "box")]
        occ[m] = {"n_frames": len(sel), "max_conf": max(confs, default=0.0),
                  "frames_with_occluder_prompt_ge_0.25": sum(any(d["conf"] >= 0.25 and d["cls"] in ("book", "bottle", "box")
                                                                 for d in r["dets"]) for r in sel)}
    (rep / "occluder_prompts.json").write_text(json.dumps(occ, indent=1))

    # matching table
    ml = ["| Crop set | n | DINOv2-S min / median | CLIP B/32 min / median |", "|---|---|---|---|"]
    names = {"pilot": "pilot frames, same sitting (P1, 4 lux)", "now": "12:16 captures (10 min later)",
             "sweep_best": "17-lux frame at 6.0 D (upright; upside down)", "sweep_series": "focus sweep 0-15 D, 17 lux",
             "af_bright": "52-lux AF frames (mug defocused)", "probe": "11:28 probe (other pose, cut off, defocused)"}
    for g, lab in names.items():
        a, b = match[ENC[0]]["positive"][g], match[ENC[1]]["positive"][g]
        ml.append(f"| mug, {lab} | {a['n']} | {a['min']:.3f} / {a['median']:.3f} | {b['min']:.3f} / {b['median']:.3f} |")
    a, b = match[ENC[0]]["negative_all"], match[ENC[1]]["negative_all"]
    ml.append(f"| background crops, same frames (all groups) | {a['n']} | median {a['median']:.3f}, **max {a['max']:.3f}** | median {b['median']:.3f}, **max {b['max']:.3f}** |")
    (rep / "match_real.md").write_text("\n".join(ml) + "\n")
    al = ["| Augmentation of the reference crop | DINOv2-S | CLIP B/32 |", "|---|---|---|"]
    for k in match[ENC[0]]["augment"]:
        al.append(f"| {k} | {match[ENC[0]]['augment'][k]:.3f} | {match[ENC[1]]['augment'][k]:.3f} |")
    al.append(f"| *max background crop (for comparison)* | *{match[ENC[0]]['negative_all']['max']:.3f}* | *{match[ENC[1]]['negative_all']['max']:.3f}* |")
    (rep / "match_augment.md").write_text("\n".join(al) + "\n")
    return tab


# ----------------------------------------------------------------------------- figures


def fig_conditions(tab, path):
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad(SURFACE)
    rows, labels, heads = [], [], []
    for sec, head in (("real", "Real frames (same mug)"), ("syn", "Synthetic edits of pilot reference t01")):
        heads.append(len(rows))
        rows.append([np.nan] * len(MODELS))
        labels.append(head)
        for r in tab:
            if r["section"] == sec:
                rows.append([r[m] for m in MODELS])
                labels.append(r["label"].strip() + (f"  (median, n={r['n']})" if r["n"] > 1 and "median" not in r["label"] else ""))
    M = np.ma.masked_invalid(np.array(rows, dtype=float))
    fig = plt.figure(figsize=(9.6, 10.2))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.43, 0.06, 0.55, 0.86])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if v >= 0.5 else INK, fontweight="bold" if v >= 0.25 else "normal")
    ax.set_xticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS], fontsize=8.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), labels, fontsize=8.5, color=INK)
    for k in heads:
        ax.get_yticklabels()[k].set_fontweight("bold")
    for j in range(1, len(MODELS)):
        ax.axvline(j - 0.5, color=SURFACE, lw=3)
    for i in range(1, len(rows)):
        ax.axhline(i - 0.5, color=SURFACE, lw=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.012, "Cell = highest confidence of a target-class box (open vocab: cup or mug; COCO: cup) with IoU >= 0.5 against the "
             "hand-annotated mug box; 0.00 = none above 0.01.\nBold = at or above ultralytics' default 0.25 threshold. "
             "One mug, one scene, 81 images: not an accuracy estimate.", fontsize=7.5, color=INK2)
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def fig_defocus(run, recs, match_rows, path):
    by = {(r["image_id"], r["model"]): r for r in recs}
    sweep = sorted({r["image_id"] for r in recs if r["image_id"].startswith("sweep2_lp_")})
    lp = [float(i.split("_lp_")[1]) for i in sweep]
    sharp = {}
    f = REPO / "data/raw/2026-09-26_camera-check/analysis/focus_sweep_run2.csv"
    for row in csv.DictReader(open(f)):
        sharp[float(row["lp_req"])] = (float(row["mug_pattern_lapvar_norm"]), row["light_ok"] == "True")
    smax = max(v[0] for v in sharp.values() if v[1])
    fig, axs = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.6, 1.6]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axs:
        style(ax)
        ax.axvspan(13.75, 15.25, color=GRID, lw=0)
        ax.axvline(6.17, color=INK2, lw=0.8, ls=":")
    ax = axs[0]
    xs = sorted(x for x in sharp if sharp[x][1])
    ax.plot(xs, [sharp[x][0] / smax for x in xs], color=INK2, lw=1.6, marker="o", ms=3)
    ax.set_ylabel("mug sharpness\n/ max", fontsize=8, color=INK)
    ax.set_ylim(0, 1.15)
    ax.text(6.35, 0.12, "mug sharpness peak 6.17 D\n(camera-check fit)", fontsize=7, color=INK2, va="bottom")
    ax.text(14.5, 0.35, "light\ndrop\n(frames\n~5x darker)", fontsize=6.5, color=INK2, ha="center")
    ax = axs[1]
    markers = ["o", "s", "^", "D"]
    for k, m in enumerate(MODELS):
        y = [by[(i, m)]["hit_conf"] for i in sweep]
        ax.plot(lp, y, color=SLOTS[k], lw=1.6, marker=markers[k], ms=4, label=MODEL_LABEL[m].replace("\n", " "))
    ax.axhline(0.25, color=INK2, lw=0.8, ls="--")
    ax.text(0.1, 0.26, "0.25 default threshold", fontsize=7, color=INK2, va="bottom")
    ax.set_ylabel("target-class confidence\non the mug", fontsize=8, color=INK)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
    ax = axs[2]
    for k, e in enumerate(ENC):
        pts = sorted((float(r["frame"].split("_lp_")[1]), float(r["sim"])) for r in match_rows
                     if r["encoder"] == e and r["set"] == "positive" and r["frame"].startswith("sweep2_lp_"))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SLOTS[k], lw=1.6, marker=markers[k], ms=4,
                label=f"{ENC_LABEL[e]}: mug crop vs reference")
        negs = [float(r["sim"]) for r in match_rows if r["encoder"] == e and r["set"] == "negative"]
        negmax = max(negs)
        ax.axhline(negmax, color=SLOTS[k], lw=1.0, ls="--")
        ax.text(15.35, negmax, f"max of {len(negs)}\nbackground\ncrops, {ENC_LABEL[e]}\n({negmax:.2f})", fontsize=6.5, color=INK2, va="center")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_ylabel("cosine similarity to\nreference mug crop", fontsize=8, color=INK)
    ax.set_xlabel("manual lens position (dioptres); exposure and white balance locked, ~17 lux", fontsize=8, color=INK)
    ax.set_xlim(-0.3, 17)
    ax.set_ylim(0, 1.02)
    axs[0].set_title("Focus sweep run 2 (31 frames, rotated upright): detection and matching vs defocus",
                     fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def fig_match(match, match_rows, path):
    groups = ["pilot", "now", "sweep_best", "sweep_series", "af_bright", "probe"]
    glab = {"pilot": "pilot, same sitting", "now": "12:16, 10 min later", "sweep_best": "17 lux, 6.0 D",
            "sweep_series": "focus sweep 0-15 D", "af_bright": "52 lux, AF defocus", "probe": "11:28 probe pose"}
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.4), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for row, e in enumerate(ENC):
        ax = axs[row, 0]
        style(ax)
        ax.grid(True, axis="y", color=SURFACE)
        for k, g in enumerate(groups):
            neg = [float(r["sim"]) for r in match_rows if r["encoder"] == e and r["set"] == "negative" and r["group"] == g]
            pos = [float(r["sim"]) for r in match_rows if r["encoder"] == e and r["set"] == "positive" and r["group"] == g]
            ax.scatter(neg, k + 0.18 + rng.uniform(-0.08, 0.08, len(neg)), s=12, facecolors="none", edgecolors=MUTED, lw=0.8,
                       label="background crop, same frames" if k == 0 else None)
            ax.scatter(pos, k - 0.18 + rng.uniform(-0.08, 0.08, len(pos)), s=16, color=SLOTS[0], lw=0,
                       label="mug crop" if k == 0 else None)
        negmax = match[e]["negative_all"]["max"]
        ax.axvline(negmax, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(range(len(groups)), [glab[g] for g in groups], fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(min(-0.05, match[e]["negative_all"]["min"] - 0.05), 1.02)
        ax.set_title(f"{ENC_LABEL[e]}: similarity to the reference mug crop", fontsize=9, color=INK, loc="left")
        ax.set_ylim(len(groups) - 0.5, -0.9)
        ax.text(negmax, -0.62, f" max background {negmax:.2f}", fontsize=7, color=INK2, va="center")
        if row == 1:  # CLIP's negatives start at ~0.42, so the left of this panel is empty
            ax.legend(fontsize=7.5, frameon=False, loc="center left")
        ax.set_xlabel("cosine similarity", fontsize=8, color=INK)
        ax = axs[row, 1]
        style(ax)
        fr = [0, 0.25, 0.5, 0.75]
        styles = [("band_flat", "flat dark band from left"), ("band_texture", "background-patch band"),
                  ("centre_texture", "background patch, centred")]
        markers = ["o", "s", "^"]
        for k, (st, lab) in enumerate(styles):
            y = [1.0] + [match[e]["augment"][f"occl_{st}_{int(f * 100)}"] for f in fr[1:]]
            ax.plot(fr, y, color=SLOTS[k], lw=1.6, marker=markers[k], ms=5, label=lab)
        ax.axhline(negmax, color=INK2, lw=0.8, ls="--")
        ax.text(0.0, negmax + 0.01, "max background crop", fontsize=7, color=INK2, va="bottom")
        ax.set_ylim(min(0.15, negmax - 0.1), 1.03)
        ax.set_xticks(fr, ["0", "25%", "50%", "75%"])
        ax.set_xlabel("share of the reference crop covered", fontsize=8, color=INK)
        ax.set_title(f"{ENC_LABEL[e]}: pasted occluder", fontsize=9, color=INK, loc="left")
        if row == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.text(0.01, 0.005, "One mug, one reference crop (pilot ref t01). Positives from the same pose are near-duplicates; "
             "no second cup exists, so instance discrimination is untested.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def fig_overlays(man, recs, path, tile_w=560):
    ids = ["probe_full", "probe_crop", "pilot_PILOT00_P1_reference_pilot_ambient_t01", "af_cont_normal_1",
           "syn_trunc_left_50", "syn_roll_30"]
    labels = {"probe_full": "11:28 probe (as captured)", "probe_crop": "11:28 probe, ROI crop",
              "pilot_PILOT00_P1_reference_pilot_ambient_t01": "12:05 pilot ref, 4 lux, sharp",
              "af_cont_normal_1": "11:41, 52 lux, mug defocused", "syn_trunc_left_50": "pilot ref, left 50% of mug cut",
              "syn_roll_30": "pilot ref, rolled 30 deg"}
    models = ["yolov8s-worldv2", "yolo11s"]
    ent = {e["id"]: e for e in man["images"]}
    by = {(r["image_id"], r["model"]): r for r in recs}
    tmp = path.parent / "_tiles"
    tiles = []
    for i in ids:
        img, gt = load_entry(ent[i])
        row = []
        for m in models:
            r = by[(i, m)]
            targets = ["cup"] if m.startswith("yolo1") or m.startswith("yolo2") else ["cup", "mug"]
            p = tmp / f"{m}_{i}.jpg"
            draw_overlay(img, r["dets"], gt, targets, f"{m} | {labels[i]} | hit {r['hit_conf']:.2f}", p, max_w=tile_w, top=4)
            row.append(Image.open(p))
        tiles.append(row)
    rh = [max(t.height for t in row) for row in tiles]
    head = 26
    W, H = tile_w * len(models) + 6 * (len(models) - 1), head + sum(rh) + 6 * (len(ids) - 1)
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(canvas)
    font = get_font(16)
    for j, m in enumerate(models):
        d.text((j * (tile_w + 6) + 4, 4), {"yolov8s-worldv2": "YOLO-World v2 S, prompts: cup, mug, book, bottle, box",
                                           "yolo11s": "YOLO11 S, COCO-80 (control)"}[m], fill=(11, 11, 11), font=font)
    y = head
    for k, row in enumerate(tiles):
        for j, t in enumerate(row):
            canvas.paste(t, (j * (tile_w + 6), y))
        y += rh[k] + 6
    canvas.save(path, quality=78, optimize=True)
    for p in tmp.glob("*.jpg"):
        p.unlink()
    tmp.rmdir()


def fig_crops(man, match_rows, path, h=170):
    from match_check import augmentations, crop  # reuse the exact augmentation code

    ent = {e["id"]: e for e in man["images"]}
    sims = {(r["encoder"], r["frame"], r["variant"]): float(r["sim"]) for r in match_rows}
    mref = man["match_reference"]
    ref_img, ref_box = apply_ops(Image.open(REPO / mref["path"]).convert("RGB"), mref.get("ops", []), mref["box"])
    ref_crop = crop(ref_img, ref_box)
    patch = crop(ref_img, [80, 300, 80 + (ref_box[2] - ref_box[0]), 300 + (ref_box[3] - ref_box[1])])
    aug = {n: c for n, _, _, c in augmentations(ref_crop, patch)}
    items = [(ref_crop, "reference crop (pilot ref t01)", None)]
    for fid, lab in [("now_fullauto", "12:16 full auto"), ("af_cont_normal_1", "52 lux, AF defocus"),
                     ("sweep2_lp_00.0", "sweep, lens 0.0 D"), ("probe_full", "11:28 probe pose")]:
        img, gt = load_entry(ent[fid])
        items.append((crop(img, gt), lab, (fid, "hand_box")))
    for v, lab in [("occl_band_texture_50", "patch band 50%"), ("occl_centre_texture_75", "centred patch 75%"),
                   ("blur_s16", "blur sigma 16 px")]:
        items.append((aug[v], lab, ("reference_t01", v)))
    negs = [r for r in match_rows if r["encoder"] == ENC[0] and r["set"] == "negative"]
    top = max(negs, key=lambda r: float(r["sim"]))
    img, _ = load_entry(ent[top["frame"]])
    items.append((crop(img, json.loads(top["box"])), "most similar background", (top["frame"], top["variant"])))
    font = get_font(13)
    tiles = []
    for im, lab, key in items:
        t = im.resize((round(im.width * h / im.height), h))
        if key is None:
            cap = f"{lab}\nself-similarity 1.00"
        else:
            cap = f"{lab}\nDINOv2 {sims[(ENC[0], *key)]:.2f} | CLIP {sims[(ENC[1], *key)]:.2f}"
        c = Image.new("RGB", (max(t.width, 200), h + 40), (252, 252, 251))
        c.paste(t, (0, 0))
        ImageDraw.Draw(c).multiline_text((2, h + 3), cap, fill=(11, 11, 11), font=font, spacing=2)
        tiles.append(c)
    rows = [tiles[:4], tiles[4:]]
    W = max(sum(t.width for t in r) + 6 * (len(r) - 1) for r in rows)
    out = Image.new("RGB", (W, 2 * (h + 40) + 6), (252, 252, 251))
    for k, r in enumerate(rows):
        x = 0
        for t in r:
            out.paste(t, (x, k * (h + 46)))
            x += t.width + 6
    out.save(path, quality=80, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--figs", required=True)
    a = ap.parse_args()
    run, figs = Path(a.run), Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    man = load_manifest(a.manifest)
    recs = load_detect(run)
    match = json.loads((run / "match" / "summary.json").read_text())
    match_rows = list(csv.DictReader(open(run / "match" / "similarities.csv")))
    tab = write_tables(run, recs, match)
    fig_conditions(tab, figs / "perception_conditions.png")
    fig_defocus(run, recs, match_rows, figs / "perception_defocus.png")
    fig_match(match, match_rows, figs / "match_similarity.png")
    fig_overlays(man, recs, figs / "perception_overlays.jpg")
    fig_crops(man, match_rows, figs / "match_crops.jpg")
    for p in ["perception_conditions.png", "perception_defocus.png", "match_similarity.png", "perception_overlays.jpg", "match_crops.jpg"]:
        print(f"{figs / p}: {(figs / p).stat().st_size / 1024:.0f} KB")
    for p in sorted((run / "report").glob("*")):
        print(f"--- {p}\n{p.read_text()}")


if __name__ == "__main__":
    main()
