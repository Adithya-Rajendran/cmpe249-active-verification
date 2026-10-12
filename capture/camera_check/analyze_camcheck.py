"""Analysis for the 2026-09-26 Pi camera check (runs on the GB10, not the Pi).

Run in the project's uv environment (numpy, pillow, matplotlib, opencv), e.g.
    uv run --no-sync python capture/camera_check/analyze_camcheck.py focus --data data/raw/2026-09-26_camera-check
(`drift` needs OpenCV for capture/ecc.py; the other subcommands still run in the old ephemeral env
 `uv run --no-project --with numpy --with pillow --with matplotlib`.)

Subcommands
    focus    sharpness (variance of Laplacian) vs manual lens position, per region of interest
    repeat   frame-to-frame differences and metadata drift, full-auto vs locked controls
    drift    locked frames spaced in time (ambient-light drift); image shifts by ECC translation
             (capture/ecc.py, sigma 1.5). --estimator phase selects the deprecated phase_shift.
    fov      footprint of the datasheet field of view at given distances / heights / tilts
    previews small downscaled previews for the lab notebook

All regions of interest are in NATIVE sensor orientation (the camera is currently mounted
upside down, so previews are rotated 180 degrees for readability; the analysis is not).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image

# (x0, y0, x1, y1) in native 2304x1296 pixel coordinates, picked by eye from a gridded
# frame (sweep_run2/lp_05.0.jpg). Descriptions refer to the 180-degree-rotated view.
ROIS = {
    "mug_pattern": (512, 384, 896, 768),     # dark brush strokes on the near mug
    "puck": (1200, 520, 1540, 740),          # round black device + desk-mat edge behind the mug
    "speaker_grille": (1330, 800, 1650, 1180),  # striped grille of the monitor / speaker base
    "shelf_far": (1700, 820, 2300, 1030),    # figurines on the far shelf
    "wall_left": (0, 660, 110, 1280),        # textured wall beside the mug (native left edge)
    "whole_frame": (0, 0, 2304, 1296),
}

# Categorical slots from the dataviz reference palette (light mode), fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def luma(path: str) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def laplacian(g: np.ndarray) -> np.ndarray:
    return g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4.0 * g[1:-1, 1:-1]


def roi_scores(g: np.ndarray) -> dict:
    out = {}
    for name, (x0, y0, x1, y1) in ROIS.items():
        sub = g[y0:y1, x0:x1]
        lv = float(laplacian(sub).var())
        m = float(sub.mean())
        out[name] = {"lapvar": lv, "mean": m, "lapvar_norm": lv / (m * m) * 1e4}
    return out


def meta(path_json: str) -> dict:
    with open(path_json) as f:
        return json.load(f)


def lp_from_name(p: str) -> float:
    return float(re.search(r"lp_([0-9.]+)\.jpg$", p).group(1))


def peak_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Sub-step peak: parabola through log(score) at the argmax and its two neighbours."""
    i = int(np.argmax(y))
    res = {"argmax_x": float(x[i]), "argmax_y": float(y[i])}
    if 0 < i < len(x) - 1 and np.all(y[i - 1:i + 2] > 0):
        xs, ls = x[i - 1:i + 2], np.log(y[i - 1:i + 2])
        a, b, _ = np.polyfit(xs, ls, 2)
        if a < 0:
            res["fit_x"] = float(-b / (2 * a))
    # Band where score >= 80% of peak (linear interpolation on the sampled curve).
    thr = 0.8 * y[i]
    lo = x[0]
    for j in range(i, 0, -1):
        if y[j - 1] < thr:
            lo = x[j - 1] + (thr - y[j - 1]) / (y[j] - y[j - 1]) * (x[j] - x[j - 1])
            break
    hi = x[-1]
    for j in range(i, len(x) - 1):
        if y[j + 1] < thr:
            hi = x[j] + (y[j] - thr) / (y[j] - y[j + 1]) * (x[j + 1] - x[j])
            break
    res["band80"] = [float(lo), float(hi)]
    res["edge_limited"] = bool(i == 0 or i == len(x) - 1)
    return res


def dist_cm(d: float) -> float:
    return float("inf") if d <= 0 else 100.0 / d


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- focus
def cmd_focus(a):
    runs = {}
    for run in a.runs:
        files = sorted(glob.glob(os.path.join(a.data, run, "lp_*.jpg")), key=lp_from_name)
        rows = []
        for f in files:
            m = meta(f[:-4] + ".json")
            g = luma(f)
            sc = roi_scores(g)
            rows.append({"file": os.path.relpath(f, a.data), "lp_req": lp_from_name(f),
                         "lp_meta": m.get("LensPosition"), "fom": m.get("FocusFoM"),
                         "lux": m.get("Lux"), "exposure_us": m["ExposureTime"],
                         "again": m["AnalogueGain"], "scores": sc})
        runs[run] = rows
    # Lighting guard: flag frames whose whole-frame mean luma differs >5% from the run median.
    report = {"rois": ROIS, "runs": {}}
    for run, rows in runs.items():
        med = float(np.median([r["scores"]["whole_frame"]["mean"] for r in rows]))
        for r in rows:
            r["light_ok"] = abs(r["scores"]["whole_frame"]["mean"] / med - 1) <= 0.05
        good = [r for r in rows if r["light_ok"]]
        x = np.array([r["lp_req"] for r in good])
        peaks = {}
        for roi in ROIS:
            for metric in ("lapvar", "lapvar_norm"):
                y = np.array([r["scores"][roi][metric] for r in good])
                pk = peak_fit(x, y)
                fx = pk.get("fit_x", pk["argmax_x"])
                pk["distance_cm_at_peak"] = dist_cm(fx)
                pk["distance_cm_band80"] = [dist_cm(pk["band80"][1]), dist_cm(pk["band80"][0])]
                peaks[f"{roi}:{metric}"] = pk
        report["runs"][run] = {
            "n_frames": len(rows), "n_light_ok": len(good),
            "excluded_light": [r["lp_req"] for r in rows if not r["light_ok"]],
            "lp_meta_max_abs_err": max(abs((r["lp_meta"] or 0) - r["lp_req"]) for r in rows),
            "peaks": peaks,
        }
        with open(os.path.join(a.out, f"focus_{run}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lp_req", "lp_meta", "fom", "lux", "light_ok"] +
                       [f"{roi}_{m}" for roi in ROIS for m in ("lapvar", "lapvar_norm", "mean")])
            for r in rows:
                w.writerow([r["lp_req"], r["lp_meta"], r["fom"], r["lux"], r["light_ok"]] +
                           [round(r["scores"][roi][m], 4) for roi in ROIS
                            for m in ("lapvar", "lapvar_norm", "mean")])
    with open(os.path.join(a.out, "focus_summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    for run, rep in report["runs"].items():
        print(f"== {run}: {rep['n_light_ok']}/{rep['n_frames']} frames light-ok; "
              f"excluded {rep['excluded_light']}; max |LensPosition-requested| {rep['lp_meta_max_abs_err']:.4f}")
        for k, pk in rep["peaks"].items():
            print(f"  {k:28s} argmax {pk['argmax_x']:5.2f}  fit {pk.get('fit_x', float('nan')):5.2f}  "
                  f"band80 {pk['band80'][0]:5.2f}-{pk['band80'][1]:5.2f} D  -> "
                  f"{pk['distance_cm_at_peak']:.1f} cm [{pk['distance_cm_band80'][0]:.1f}, "
                  f"{pk['distance_cm_band80'][1]:.1f}]{'  EDGE' if pk['edge_limited'] else ''}")
    if a.fig:
        plot_focus(a, runs, report)


def plot_focus(a, runs, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    af = json.load(open(a.af_json)) if a.af_json else {}
    rois = ["mug_pattern", "puck", "speaker_grille", "shelf_far", "whole_frame"]
    labels = {"mug_pattern": "mug pattern (near)", "puck": "black puck (mid)",
              "speaker_grille": "speaker grille (mid)", "shelf_far": "far shelf",
              "whole_frame": "whole frame"}
    fig, ax = plt.subplots(figsize=(9.6, 3.9), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for run, ls, mk in zip(a.runs[:1], ["-"], ["o"]):
        rows = [r for r in runs[run] if r["light_ok"]]
        x = np.array([r["lp_req"] for r in rows])
        for k, roi in enumerate(rois):
            y = np.array([r["scores"][roi]["lapvar_norm"] for r in rows])
            y = y / y.max()
            ax.plot(x, y, ls, color=SLOTS[k], lw=1.6 if run == a.runs[0] else 1.0,
                    marker=mk, ms=3.5 if run == a.runs[0] else 3,
                    alpha=1.0 if run == a.runs[0] else 0.8,
                    label=labels[roi] if run == a.runs[0] else None)
    ymax = 1.08
    for lo, hi, name in af.get("bands", []):
        ax.axvspan(lo, hi, color=INK2, alpha=0.15, lw=0, label=name)
    for k, (lp, name) in enumerate(af.get("lines", [])):
        ax.axvline(lp, color=INK2, lw=1.0, ls=["-", ":"][k % 2], label=name)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.3, 14.0)
    ax.set_xlabel("manual lens position (dioptres; nominal distance = 100/D cm)", fontsize=9, color=INK)
    ax.set_ylabel("sharpness / run max\n(Laplacian var ÷ mean²)", fontsize=9, color=INK)
    secx = ax.secondary_xaxis("top", functions=(lambda d: d, lambda d: d))
    ticks = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    secx.set_xticks(ticks)
    secx.set_xticklabels([f"{100 / t:.0f}" for t in ticks], fontsize=7, color=INK2)
    secx.set_xlabel("nominal focus distance (cm)", fontsize=8, color=INK2)
    secx.spines["top"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(a.title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(a.fig, dpi=110, facecolor=SURFACE)
    print("wrote", a.fig, os.path.getsize(a.fig), "bytes")


# ---------------------------------------------------------------------------- repeat
def seq_stats(files):
    ims = [rgb(f) for f in files]
    ms = [meta(f[:-4] + ".json") for f in files]
    first = ims[0]
    cons, vs_first, frac10 = [], [], []
    for i in range(1, len(ims)):
        d = np.abs(ims[i] - ims[i - 1])
        cons.append(float(d.mean()))
        d0 = np.abs(ims[i] - first)
        vs_first.append(float(d0.mean()))
        frac10.append(float((d0.max(axis=2) > 10).mean()))
    means = np.array([im.reshape(-1, 3).mean(0) for im in ims])

    # Noise-suppressed difference: 8x8 block means (averages 64 px, so ~8x less sensor noise)
    # isolate systematic changes (focus, white balance, exposure, light) from per-pixel noise.
    def blk(im, b=8):
        h, w = (im.shape[0] // b) * b, (im.shape[1] // b) * b
        return im[:h, :w].reshape(h // b, b, w // b, b, 3).mean(axis=(1, 3))
    bl = [blk(im) for im in ims]
    blk_vs_first = [float(np.abs(bl[i] - bl[0]).mean()) for i in range(1, len(bl))]
    blk_cons = [float(np.abs(bl[i] - bl[i - 1]).mean()) for i in range(1, len(bl))]
    roi_vs_first = {}
    for name, (x0, y0, x1, y1) in ROIS.items():
        if name == "whole_frame":
            continue
        roi_vs_first[name] = [float(np.abs(ims[i][y0:y1, x0:x1] - first[y0:y1, x0:x1]).mean())
                              for i in range(1, len(ims))]
    keys = ["ExposureTime", "AnalogueGain", "DigitalGain", "LensPosition", "Lux", "ColourTemperature", "FocusFoM"]
    md = {}
    for k in keys:
        v = [m.get(k) for m in ms]
        if all(x is not None for x in v):
            md[k] = {"values": v, "min": min(v), "max": max(v),
                     "range": max(v) - min(v), "rel_range": (max(v) - min(v)) / (abs(np.mean(v)) or 1)}
    cg = np.array([m["ColourGains"] for m in ms])
    md["ColourGains"] = {"values": cg.tolist(), "range_r": float(np.ptp(cg[:, 0])), "range_b": float(np.ptp(cg[:, 1]))}
    ccm = np.array([m["ColourCorrectionMatrix"] for m in ms])
    md["ColourCorrectionMatrix_max_abs_range"] = float(np.ptp(ccm, axis=0).max())
    return {
        "files": [os.path.basename(f) for f in files],
        "mad_consecutive": cons, "mad_consecutive_mean": float(np.mean(cons)), "mad_consecutive_max": float(np.max(cons)),
        "mad_vs_first": vs_first, "mad_vs_first_max": float(np.max(vs_first)),
        "frac_px_gt10_vs_first": frac10, "frac_px_gt10_vs_first_max": float(np.max(frac10)),
        "blockmean8_mad_vs_first": blk_vs_first, "blockmean8_mad_vs_first_max": float(np.max(blk_vs_first)),
        "blockmean8_mad_consecutive": blk_cons,
        "roi_mad_vs_first": roi_vs_first,
        "roi_mad_vs_first_max": {k: float(np.max(v)) for k, v in roi_vs_first.items()},
        "channel_means": means.tolist(), "channel_mean_range_rgb": np.ptp(means, axis=0).tolist(),
        "channel_mean_std_rgb": means.std(axis=0, ddof=1).tolist(),
        "metadata": md,
    }, ims


def cmd_repeat(a):
    out = {}
    stacks = {}
    for tag in a.tags:
        files = sorted(glob.glob(os.path.join(a.data, a.dir, f"{tag}_*.jpg")))
        st, ims = seq_stats(files)
        out[tag] = st
        stacks[tag] = ims
        print(f"== {tag}: n={len(files)}")
        print(f"  mean |diff| consecutive: mean {st['mad_consecutive_mean']:.3f}  max {st['mad_consecutive_max']:.3f}  (8-bit levels)")
        print(f"  mean |diff| vs first:    max {st['mad_vs_first_max']:.3f};  frac px >10 levels vs first: max {st['frac_px_gt10_vs_first_max']:.4f}")
        print(f"  8x8 block-mean |diff| vs first: {[round(x, 3) for x in st['blockmean8_mad_vs_first']]}")
        print(f"  ROI mean |diff| vs first (max): { {k: round(v, 3) for k, v in st['roi_mad_vs_first_max'].items()} }")
        print(f"  channel-mean range R,G,B: {[round(x, 3) for x in st['channel_mean_range_rgb']]}  std: {[round(x, 3) for x in st['channel_mean_std_rgb']]}")
        for k, v in st["metadata"].items():
            if isinstance(v, dict) and "range" in v:
                print(f"  {k:18s} min {v['min']:.4f} max {v['max']:.4f} range {v['range']:.4f}")
        cgm = st["metadata"]["ColourGains"]
        print(f"  ColourGains range r {cgm['range_r']:.4f} b {cgm['range_b']:.4f}; CCM max abs range {st['metadata']['ColourCorrectionMatrix_max_abs_range']:.4f}")
    with open(os.path.join(a.out, f"{a.name}_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    if a.fig:
        plot_repeat(a, out, stacks)


def plot_repeat(a, out, stacks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = a.tags
    fig = plt.figure(figsize=(8.0, 5.2), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.45, wspace=0.3)
    # Row 1: |frame_last - frame_first| heatmaps, same scale, rotated 180 for readability.
    vmax = 12
    for k, tag in enumerate(tags):
        ax = fig.add_subplot(gs[0, k])
        ims = stacks[tag]
        d = np.abs(ims[-1] - ims[0]).mean(axis=2)[::-4, ::-4]
        im = ax.imshow(d, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        st = out[tag]
        ax.set_title(f"{a.labels[k]}: |last − first|, mean {st['mad_vs_first'][-1]:.2f}",
                     fontsize=8.5, color=INK, loc="left")
    cax = fig.add_axes([0.92, 0.57, 0.012, 0.3])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.set_label("8-bit levels", fontsize=7, color=INK2)
    # Row 2 left: consecutive-frame mean abs diff per pair.
    ax = fig.add_subplot(gs[1, 0])
    style_axes(ax)
    for k, tag in enumerate(tags):
        v = out[tag]["mad_consecutive"]
        ax.plot(range(2, len(v) + 2), v, "-o", color=SLOTS[k], lw=1.6, ms=4,
                label=f"{a.labels[k]}: per pixel, vs previous frame")
        v = out[tag]["blockmean8_mad_vs_first"]
        ax.plot(range(2, len(v) + 2), v, "--s", color=SLOTS[k], lw=1.2, ms=3.5,
                label=f"{a.labels[k]}: 8×8 block means, vs frame 1")
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("frame index", fontsize=8.5, color=INK)
    ax.set_ylabel("mean |diff| (8-bit levels)", fontsize=8.5, color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    # Row 2 right: lens position per frame.
    ax = fig.add_subplot(gs[1, 1])
    style_axes(ax)
    for k, tag in enumerate(tags):
        v = out[tag]["metadata"]["LensPosition"]["values"]
        ax.plot(range(1, len(v) + 1), v, "-o", color=SLOTS[k], lw=1.6, ms=4, label=a.labels[k])
    ax.set_xlabel("frame index", fontsize=8.5, color=INK)
    ax.set_ylabel("LensPosition (dioptres)", fontsize=8.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle(a.title, fontsize=10, color=INK, x=0.02, ha="left")
    fig.savefig(a.fig, dpi=100, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", a.fig, os.path.getsize(a.fig), "bytes")


# ----------------------------------------------------------------------------- drift
def ecc_shift(a: np.ndarray, b: np.ndarray):
    """Translation of b relative to a (px): (dx, dy, cc) from capture/ecc.py ecc_translation
    (intensity mode, Gaussian sigma 1.5, 24 px border excluded). Inputs are copied by ecc.py."""
    cap = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if cap not in sys.path:
        sys.path.insert(0, cap)
    import ecc  # capture/ecc.py; needs OpenCV (project uv env)
    r = ecc.ecc_translation(a, b, sigma=1.5, mode="intensity")
    return float(r.dx), float(r.dy), float(r.cc)


def phase_shift(a: np.ndarray, b: np.ndarray):
    """DEPRECATED - kept only as a comparison (data/raw/2026-09-26_case-check/scripts/
    shift_recheck_camcheck.py imports it). Unstable at sub-pixel scale (case check section 2.11);
    use ecc_shift. Translation of b relative to a (px) by phase correlation with parabolic sub-pixel peak."""
    wy = np.hanning(a.shape[0])[:, None]
    wx = np.hanning(a.shape[1])[None, :]
    A = np.fft.fft2((a - a.mean()) * wy * wx)
    B = np.fft.fft2((b - b.mean()) * wy * wx)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    i, j = np.unravel_index(np.argmax(r), r.shape)

    def sub(c, m):
        l, cc, rr = c[(m - 1) % len(c)], c[m], c[(m + 1) % len(c)]
        return m + 0.5 * (l - rr) / (l - 2 * cc + rr)
    dy, dx = sub(r[:, j], i), sub(r[i, :], j)
    if dy > a.shape[0] / 2:
        dy -= a.shape[0]
    if dx > a.shape[1] / 2:
        dx -= a.shape[1]
    return float(-dx), float(-dy), float(r.max())


def cmd_drift(a):
    files = sorted(glob.glob(os.path.join(a.data, a.dir, "locked_*.jpg")))
    st, ims = seq_stats(files)
    times = []
    log = open(os.path.join(a.data, "commands.log")).read().splitlines()
    for f in files:
        rel = os.path.relpath(f, a.data)
        for line in log:
            if f"-o {rel} " in line:
                times.append(line.split()[0])
    st["log_times"] = times
    shift = ecc_shift if a.estimator == "ecc" else phase_shift
    st["shift_estimator"] = ("ecc_translation (capture/ecc.py, intensity, sigma 1.5)" if a.estimator == "ecc"
                             else "phase_shift (DEPRECATED)")
    g0 = luma(files[0])
    st["shift_vs_first_px"] = [shift(g0, luma(f))[:2] for f in files[1:]]
    for ref in a.shift_refs:
        fs = sorted(glob.glob(os.path.join(a.data, ref)))
        st.setdefault("shift_last_vs_first_other", {})[ref] = shift(luma(fs[0]), luma(fs[-1]))[:2]
    with open(os.path.join(a.out, "drift_summary.json"), "w") as fo:
        json.dump(st, fo, indent=2)
    print("shift vs first (dx, dy px):", [(round(x, 2), round(y, 2)) for x, y in st["shift_vs_first_px"]])
    print("other sequences, last vs first:", st.get("shift_last_vs_first_other"))
    if a.fig:
        plot_drift(a, st, times)
    print(f"n={len(files)} times {times[:1]} .. {times[-1:]}")
    print("mad vs first:", [round(x, 2) for x in st["mad_vs_first"]])
    print("lux:", [round(x, 2) for x in st["metadata"]["Lux"]["values"]])
    print("channel means:", [[round(c, 1) for c in m] for m in st["channel_means"]])


def plot_drift(a, st, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import datetime as dt
    t = [dt.datetime.fromisoformat(x) for x in times]
    mins = np.array([(x - t[0]).total_seconds() / 60 for x in t])
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = axs[0]
    style_axes(ax)
    ax.plot(mins[1:], st["mad_vs_first"], "-o", color=SLOTS[0], lw=1.6, ms=4, label="per pixel")
    ax.plot(mins[1:], st["blockmean8_mad_vs_first"], "-s", color=SLOTS[1], lw=1.6, ms=3.5, label="8×8 block means")
    ax.set_ylim(0, None)
    ax.set_title("mean |frame − frame 1| (8-bit)", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axs[1]
    style_axes(ax)
    sh = np.array(st["shift_vs_first_px"])
    ax.plot(mins[1:], np.hypot(sh[:, 0], sh[:, 1]), "-o", color=SLOTS[0], lw=1.6, ms=4)
    ax.set_ylim(0, None)
    ax.set_title("image shift vs frame 1 (px, phase corr.)", fontsize=8.5, color=INK, loc="left")
    ax = axs[2]
    style_axes(ax)
    lux = np.array(st["metadata"]["Lux"]["values"])
    gm = np.array(st["channel_means"])[:, 1]
    ax.plot(mins, (lux / lux[0] - 1) * 100, "-o", color=SLOTS[0], lw=1.6, ms=4, label="rpicam Lux estimate")
    ax.plot(mins, (gm / gm[0] - 1) * 100, "-s", color=SLOTS[1], lw=1.6, ms=3.5, label="green-channel mean")
    ax.set_title("change vs frame 1 (%)", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axs:
        ax.set_xlabel("minutes since first frame", fontsize=8, color=INK)
    fig.suptitle(f"Locked controls, {len(times)} stills ~1 min apart ({times[0][11:16]}–{times[-1][11:16]})",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.fig, dpi=100, facecolor=SURFACE)
    print("wrote", a.fig, os.path.getsize(a.fig), "bytes")


# ------------------------------------------------------------------------------- fov
def footprint_flat(d, hfov, vfov):
    return 2 * d * math.tan(math.radians(hfov / 2)), 2 * d * math.tan(math.radians(vfov / 2))


def footprint_tilted(h, depression_deg, hfov, vfov):
    """Camera at height h above the table, optical axis pitched down by depression_deg.
    Returns near/far ground distances (from the point below the camera) of the vertical FOV
    edges, and the ground width of the horizontal FOV at the near edge, at the axis and at the far edge."""
    t = math.radians(depression_deg)
    v = math.radians(vfov / 2)
    th = math.tan(math.radians(hfov / 2))
    near_ang, far_ang = t + v, t - v
    near = h / math.tan(near_ang)
    far = h / math.tan(far_ang) if far_ang > 0 else float("inf")
    axis = h / math.tan(t)

    def width_at(ang):  # slant range along the ray's vertical plane projected onto optical axis
        if ang <= 0:
            return float("inf")
        rng = h / math.sin(ang)
        return 2 * rng * math.cos(ang - t) * th
    return {"near": near, "axis": axis, "far": far, "slant_axis": h / math.sin(t),
            "w_near": width_at(near_ang), "w_axis": width_at(t), "w_far": width_at(far_ang)}


def cmd_fov(a):
    hf, vf = a.hfov, a.vfov
    print(f"datasheet FOV {hf} x {vf} deg; pinhole from f=4.74 mm, 6.45 x 3.63 mm sensor: "
          f"{2 * math.degrees(math.atan(3.225 / 4.74)):.1f} x {2 * math.degrees(math.atan(1.815 / 4.74)):.1f} deg")
    res = {"hfov": hf, "vfov": vf, "fronto_parallel": {}, "tilted": []}
    for d in (30, 45, 60):
        w, h = footprint_flat(d, hf, vf)
        res["fronto_parallel"][d] = [w, h]
        print(f"  plane at {d} cm: {w:.1f} x {h:.1f} cm")
    for h, dep in [(25, 30), (30, 35), (32, 40), (35, 40), (38, 40), (42, 40), (35, 45), (40, 50)]:
        r = footprint_tilted(h, dep, hf, vf)
        r.update({"height": h, "depression": dep})
        res["tilted"].append(r)
        print(f"  h={h} cm, tilt {dep} deg down: axis hits table {r['axis']:.1f} cm out (slant {r['slant_axis']:.1f} cm); "
              f"visible ground {r['near']:.1f}..{r['far']:.1f} cm; width near {r['w_near']:.1f}, "
              f"axis {r['w_axis']:.1f}, far {r['w_far']:.1f} cm")
    with open(os.path.join(a.out, "fov_summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    if a.fig:
        plot_fov(a)


def plot_fov(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    R, H, dep = a.radius, a.height, a.tilt
    hf, vf = a.hfov, a.vfov
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0), dpi=100,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    # Top view: workspace rectangle, 5 positions on an arc, FOV wedge at P3.
    style_axes(ax1)
    ax1.grid(False)
    from matplotlib.patches import Circle
    wd = a.ws_d
    ax1.add_patch(Circle((0, 0), wd / 2, fill=False, ec=INK, lw=1.2))
    ax1.text(0, wd / 2 + 2, f"workspace Ø{wd:.0f} cm", fontsize=7.5, color=INK, ha="center", va="bottom")
    angles = np.linspace(-a.arc / 2, a.arc / 2, 5)
    for k, ang in enumerate(angles):
        th = math.radians(-90 + ang)
        x, y = R * math.cos(th), R * math.sin(th)
        ax1.plot(x, y, "o", color=SLOTS[0], ms=7, mec=SURFACE, mew=1.5)
        ax1.text(x * 1.12, y * 1.12, f"P{k + 1}", fontsize=8, color=INK, ha="center", va="center")
        if k == 2:
            # ground footprint of the FOV from P3 (trapezoid)
            fp = footprint_tilted(H, dep, hf, vf)
            near, far = fp["near"], min(fp["far"], R + 60)
            wn, wf = fp["w_near"], fp["w_near"] * far / near if fp["far"] == float("inf") else fp["w_far"]
            pts = [(-wn / 2, y + near), (wn / 2, y + near), (wf / 2, y + far), (-wf / 2, y + far)]
            ax1.add_patch(Polygon(pts, closed=True, fc=SLOTS[0], alpha=0.10, ec=SLOTS[0], lw=0.8))
    arc = np.linspace(math.radians(-90 - a.arc / 2 - 5), math.radians(-90 + a.arc / 2 + 5), 100)
    ax1.plot(R * np.cos(arc), R * np.sin(arc), color=INK2, lw=0.8)
    ax1.plot(0, 0, "+", color=INK, ms=8)
    ax1.set_aspect("equal")
    ax1.set_xlim(-R - 16, R + 16)
    ax1.set_ylim(-R - 10, R + 35)
    ax1.set_xlabel("cm", fontsize=8, color=INK2)
    ax1.set_title(f"Top view: 5 stops on a {a.arc:.0f}° arc, r = {R:.0f} cm\n(shaded: P3 ground footprint)",
                  fontsize=8.5, color=INK, loc="left")
    # Side view from P3.
    style_axes(ax2)
    ax2.grid(False)
    ax2.axhline(0, color=INK, lw=1.0)
    ax2.plot(0, H, "o", color=SLOTS[0], ms=7)
    t = math.radians(dep)
    for ang, lw in [(dep, 1.2), (dep + vf / 2, 0.8), (dep - vf / 2, 0.8)]:
        r = math.radians(ang)
        L = H / math.sin(r) if ang > 0 else 120
        L = min(L, 110)
        ax2.plot([0, L * math.cos(r)], [H, H - L * math.sin(r)], color=SLOTS[0] if lw > 1 else INK2, lw=lw)
    ax2.add_patch(Rectangle((R - wd / 2, 0), wd, 1.2, color=INK2))
    for xo in (R - wd / 2 + 4, R + wd / 2 - 4):
        ax2.plot([xo - 4, xo - 4, xo + 4, xo + 4], [0, 12, 12, 0], color=SLOTS[1], lw=1.4)

    ax2.text(2, H + 3, f"camera {H:.0f} cm up, {dep:.0f}° down", fontsize=7.5, color=INK)
    ax2.set_aspect("equal")
    ax2.set_xlim(-5, R + wd / 2 + 25)
    ax2.set_ylim(-10, H + 12)
    ax2.text(R, -4.5, f"workspace Ø{wd:.0f} cm, 12 cm objects at its near/far edge", fontsize=7,
             color=INK2, ha="center", va="top")
    ax2.set_xlabel("cm from camera (horizontal)", fontsize=8, color=INK2)
    ax2.set_title("Side view from P3 (optical axis + vertical FOV)", fontsize=8.5, color=INK, loc="left")
    fig.suptitle("PROPOSED rig geometry (recommendation, not a measurement)", fontsize=9.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.fig, dpi=100, facecolor=SURFACE)
    print("wrote", a.fig, os.path.getsize(a.fig), "bytes")


# -------------------------------------------------------------------------- previews
def cmd_previews(a):
    from PIL import ImageDraw
    for spec in a.items:
        src, dst, *opt = spec.split(":")
        im = Image.open(os.path.join(a.data, src)).convert("RGB")
        if "rois" in opt:
            d = ImageDraw.Draw(im)
            for k, (name, (x0, y0, x1, y1)) in enumerate(ROIS.items()):
                if name == "whole_frame":
                    continue
                d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=SLOTS[k], width=8)
        im = im.rotate(180)
        if "rois" in opt:
            d = ImageDraw.Draw(im)
            for k, (name, (x0, y0, x1, y1)) in enumerate(ROIS.items()):
                if name == "whole_frame":
                    continue
                rx0, ry0 = 2304 - x1, 1296 - y1
                d.rectangle([rx0, max(ry0 - 44, 0), rx0 + 26 * len(name), max(ry0 - 2, 40)], fill=(255, 255, 255))
                d.text((rx0 + 6, max(ry0 - 40, 2)), name, fill=(0, 0, 0), font_size=36)
        im.thumbnail((a.width, a.width))
        im.save(dst, quality=82, optimize=True)
        print("wrote", dst, im.size, os.path.getsize(dst), "bytes")


def cmd_sheet(a):
    """Contact sheet of frames (rotated 180 for readability) with a caption strip per tile."""
    from PIL import ImageDraw
    tiles = []
    for spec in a.items:
        src, label = spec.split("=", 1) if "=" in spec else (spec, os.path.basename(spec))
        im = Image.open(os.path.join(a.data, src)).convert("RGB").rotate(180)
        im.thumbnail((a.width, a.width))
        cap = Image.new("RGB", (im.width, 26), (252, 252, 251))
        ImageDraw.Draw(cap).text((6, 5), label, fill=(11, 11, 11), font_size=15)
        t = Image.new("RGB", (im.width, im.height + 26), (252, 252, 251))
        t.paste(cap, (0, 0)); t.paste(im, (0, 26))
        tiles.append(t)
    cols = a.cols
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * 6, rows * th + (rows - 1) * 6), (252, 252, 251))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (tw + 6), (i // cols) * (th + 6)))
    sheet.save(a.out_file, quality=80, optimize=True)
    print("wrote", a.out_file, sheet.size, os.path.getsize(a.out_file), "bytes")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("focus")
    f.add_argument("--data", required=True)
    f.add_argument("--runs", nargs="+", default=["sweep_run2"])
    f.add_argument("--out", required=True)
    f.add_argument("--fig")
    f.add_argument("--af-json", help="json {label: lens_position} of AF results to mark")
    f.add_argument("--title", default="Manual focus sweep")
    r = sub.add_parser("repeat")
    r.add_argument("--data", required=True)
    r.add_argument("--dir", default="rep")
    r.add_argument("--tags", nargs="+", default=["auto", "locked"])
    r.add_argument("--labels", nargs="+", default=["full auto", "locked"])
    r.add_argument("--name", default="repeat")
    r.add_argument("--out", required=True)
    r.add_argument("--fig")
    r.add_argument("--title", default="Repeatability")
    d = sub.add_parser("drift")
    d.add_argument("--data", required=True)
    d.add_argument("--dir", default="drift")
    d.add_argument("--out", required=True)
    d.add_argument("--fig")
    d.add_argument("--shift-refs", nargs="*", default=["rep/locked_*.jpg", "rep/auto_*.jpg"])
    d.add_argument("--estimator", choices=["ecc", "phase"], default="ecc",
                   help="image-shift estimator: ecc (default, capture/ecc.py) or phase (deprecated phase_shift)")
    v = sub.add_parser("fov")
    v.add_argument("--hfov", type=float, default=66.0)
    v.add_argument("--vfov", type=float, default=41.0)
    v.add_argument("--out", required=True)
    v.add_argument("--fig")
    v.add_argument("--radius", type=float, default=38.0)
    v.add_argument("--height", type=float, default=32.0)
    v.add_argument("--tilt", type=float, default=40.0)
    v.add_argument("--arc", type=float, default=120.0)
    v.add_argument("--ws-d", type=float, default=40.0, help="workspace diameter (cm)")
    pv = sub.add_parser("previews")
    pv.add_argument("--data", required=True)
    pv.add_argument("--width", type=int, default=960)
    pv.add_argument("items", nargs="+", help="src_rel:dst_path[:rois]")
    sh = sub.add_parser("sheet")
    sh.add_argument("--data", required=True)
    sh.add_argument("--out-file", required=True)
    sh.add_argument("--cols", type=int, default=2)
    sh.add_argument("--width", type=int, default=480)
    sh.add_argument("items", nargs="+", help="src_rel[=caption]")
    a = p.parse_args(argv)
    if hasattr(a, "out"):
        os.makedirs(a.out, exist_ok=True)
    {"focus": cmd_focus, "repeat": cmd_repeat, "drift": cmd_drift, "fov": cmd_fov,
     "previews": cmd_previews, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
