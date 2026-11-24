"""Measure hidden Isaac Sim pairs from the two version 1 native image archives.

Run with native-isaac-evaluation.zip and native-calibration.zip, in that order.
The script needs NumPy, Pillow and the paper's evidence CSV files.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

EVIDENCE = Path(__file__).resolve().parent / "evidence"
NATIVE = (2304, 1296)
DELIVERED = (1200, 675)
TILE = 16
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def rows(filename):
    with (EVIDENCE / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def image(archive, path, mode):
    with Image.open(io.BytesIO(archive.read(path))) as src:
        if src.size != NATIVE:
            raise ValueError(f"Unexpected image size in {path}")
        return src.convert(mode)


def mask(archive, folder, kind, stop):
    return np.asarray(image(archive, f"{folder}/masks/{kind}_{stop}.png", "L")) != 0


def frame(archive, folder, kind, stop):
    return image(archive, f"{folder}/rgb/{kind}_{stop}.png", "RGB")


def compare_stage(archive, folder, prefixes):
    """Check all archived transform, camera, geometry and visibility fields of the placed twins."""
    record = json.loads(archive.read(f"{folder}/observed.json"))
    states = record["usd_per_state"]
    previous = None
    for prefix in prefixes:
        retained, removed = (states[prefix + "placed_" + s] for s in ("present", "removed"))
        if previous is not None and retained != previous:
            raise ValueError(f"Recorded retained scene differs across repeats: {folder}")
        previous = retained
        if retained.keys() != removed.keys():
            raise ValueError(f"Scene membership differs: {folder}")
        for name, data in retained.items():
            expected = dict(data)
            if name == "/World/Objects/target" or name.startswith("/World/Objects/target/"):
                if data["visibility"] != "inherited":
                    raise ValueError(f"Unexpected target visibility: {folder}/{name}")
                expected["visibility"] = "invisible"
            if expected != removed[name]:
                raise ValueError(f"Other scene changes at {folder}/{name}")
    return record["cameras"]


def tile_means(rgb):
    width, height = NATIVE
    # Sum the three 8-bit channels before averaging over the native pixel grid.
    value = np.asarray(rgb).sum(axis=2, dtype=np.int32)
    return value.reshape(height // TILE, TILE, width // TILE, TILE).mean(axis=(1, 3)) / 3


def peak(difference):
    return np.unravel_index(np.abs(difference).argmax(), difference.shape)


def diff(retained, removed):
    return np.abs(np.asarray(removed, dtype=np.int16) - np.asarray(retained, dtype=np.int16))


def main_pairs(archive):
    query = {
        (r["arrangement"], r["level"], r["stop"], r["state"]): r
        for r in rows("queries.csv") if r["engine"] == "isaac"
    }
    records, parked = [], []
    checked_folders = set()
    for pair in rows("isaac-input-pairs.csv"):
        if pair["placement"] != "placed":
            continue
        layout, level, stop = (pair[k] for k in ("arrangement", "level", "stop"))
        folder = f"isaac/evaluation/{layout}/{level}"
        if folder not in checked_folders:
            cameras = compare_stage(archive, folder, [""])
            if level == "D0" and cameras["nominal"] != cameras["revisit"]:
                raise ValueError(f"Unexpected nominal camera shift: {folder}")
            checked_folders.add(folder)
        if level == "D0":
            # Same physical scene, target retained, blocker parked, matched camera.
            reference, present = (frame(archive, folder, state, stop) for state in ("reference", "clear_present"))
            parked.append(diff(reference, present).mean())
        exposed = mask(archive, folder, "target_placed_present", stop)
        if np.count_nonzero(exposed) != int(pair["target_native_pixels"]):
            raise ValueError(f"Retained target mask disagrees with pair table: {folder}/{stop}")
        if mask(archive, folder, "target_placed_removed", stop).any():
            raise ValueError(f"Nonempty removed target mask: {folder}/{stop}")
        if exposed.any():
            continue
        alone = mask(archive, folder, "alone", stop)
        blocker = mask(archive, folder, "blocker_placed_present", stop)
        other_blocker = mask(archive, folder, "blocker_placed_removed", stop)
        if not alone.any() or (alone & ~blocker).any():
            raise ValueError(f"Retained blocker does not cover the isolated target: {folder}/{stop}")
        native, delivered = [], []
        for state in ("placed_present", "placed_removed"):
            rgb = frame(archive, folder, state, stop)
            resized = rgb.resize(DELIVERED, RESAMPLE)
            q = query[layout, level, stop, state]
            for im, key in ((rgb, "query_native_pixels_sha256"), (resized, "query_pixels_sha256")):
                if hashlib.sha256(im.tobytes()).hexdigest() != q[key]:
                    raise ValueError(f"Image differs from published {key}: {folder}/{state}_{stop}")
            native.append(rgb)
            delivered.append(resized)
        delta = diff(*native)
        changed = delta.any(axis=2)
        tile_delta = tile_means(native[1]) - tile_means(native[0])
        y, x = peak(tile_delta)
        window = np.s_[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE]
        records.append({
            "key": f"{layout}/{level}/{stop}",
            "changed": changed.sum(),
            "changed_pct": 100 * changed.mean(),
            "native_mad": delta.mean(),
            "delivered_mad": diff(*delivered).mean(),
            "outside_pct": 100 * (changed & ~alone).sum() / changed.sum(),
            "strong_pct": 100 * (delta.max(axis=2) >= 8).mean(),
            "blocker_xor": (blocker != other_blocker).sum(),
            "peak_x": x * TILE,
            "peak_y": y * TILE,
            "peak_delta": tile_delta[y, x],
            "peak_blocker_pixels": blocker[window].sum(),
        })
    print(f"Main hidden views {len(records)}, layout/level scene records checked {len(checked_folders)}")
    print("Both visible-target masks are empty in each hidden pair. The retained blocker covers the full isolated silhouette.")
    print("Only target visibility differs in the recorded object, shelf and camera fields of the placed twins.")
    print("All native and resized hidden revisit images match the published decoded RGB hashes.")
    print("Whole-image mean absolute difference (MAD) averages all pixels and RGB channels on the 0 to 255 scale.")
    print("Summary columns: minimum, median, maximum over pairs")
    for label, key in [
        ("Native pixels with any channel different", "changed"),
        ("Native pixels with any channel different (%)", "changed_pct"),
        ("Native mean absolute RGB channel difference", "native_mad"),
        ("Delivered mean absolute RGB channel difference", "delivered_mad"),
        ("Changed native pixels outside isolated target silhouette (%)", "outside_pct"),
        ("Native pixels differing by >=8 in any channel (%)", "strong_pct"),
        ("Native pixels differing in blocker-mask membership", "blocker_xor"),
    ]:
        values = [r[key] for r in records]
        print(f"{label}: {min(values):.6f} {np.median(values):.6f} {max(values):.6f}")
    print(f"Parked D0 same-state reference/revisit native RGB difference ({len(parked)} pairs): "
          f"{min(parked):.6f} {np.median(parked):.6f} {max(parked):.6f}")
    print("Native per-pair results. The peak is the grid tile with the largest absolute signed RGB mean.")
    print("key changed_pixels native_MAD delivered_MAD outside_silhouette_pct >=8_pct "
          "blocker_xor_px peak_x peak_y peak_removed_minus_present peak_blocker_pixels")
    for r in records:
        print("{key} {changed} {native_mad:.6f} {delivered_mad:.6f} {outside_pct:.6f} "
              "{strong_pct:.6f} {blocker_xor} {peak_x} {peak_y} {peak_delta:.6f} {peak_blocker_pixels}".format(**r))


def calibration(archive):
    print("\nSeparate fixed-camera calibration, native 16 x 16 tiles, zero-based top-left (x,y).")
    print("Select the largest absolute tile contrast on the base pair alone. Reuse its location on both repeats.")
    print("For this descriptive check, require the base sign in both repeats and a minimum absolute contrast")
    print("across the three pairs larger than the largest within-state range of their three tile means.")
    print("key x y blocker_px_max isolated_px present_base,r1,r2 removed_base,r1,r2 "
          "removed_minus_present_base,r1,r2 largest_within_state_range passes")
    prefixes = ["", "repeat1_", "repeat2_"]
    observed = []
    for layout in rows("arrangements.csv"):
        if layout["split"] != "calibration":
            continue
        folder = "isaac/calibration/" + layout["arrangement"] + "/D0"
        compare_stage(archive, folder, prefixes)
        for stop in ("P1", "P2", "P3", "P4", "P5"):
            if mask(archive, folder, "target_placed_present", stop).any():
                continue
            means = []
            blocker_counts = []
            for prefix in prefixes:
                values = []
                for state in ("present", "removed"):
                    kind = f"{prefix}placed_{state}"
                    if mask(archive, folder, "target_" + kind, stop).any():
                        raise ValueError(f"Target exposed in {folder}/{kind}_{stop}")
                    values.append(tile_means(frame(archive, folder, kind, stop)))
                if not prefix:
                    y, x = peak(values[1] - values[0])
                    window = np.s_[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE]
                for state in ("present", "removed"):
                    blocker = mask(archive, folder, f"blocker_{prefix}placed_{state}", stop)
                    blocker_counts.append(blocker[window].sum())
                means.append([v[y, x] for v in values])
            present, removed = np.asarray(means).T
            cross = removed - present
            within = max(np.ptp(present), np.ptp(removed))
            passed = np.all(np.sign(cross) == np.sign(cross[0])) and min(np.abs(cross)) > within
            observed.append(passed)
            vectors = [",".join(f"{v:.3f}" for v in series) for series in (present, removed, cross)]
            print(f"{layout['arrangement']}/{stop} {x * TILE} {y * TILE} {max(blocker_counts)} "
                  f"{mask(archive, folder, 'alone', stop)[window].sum()} {' '.join(vectors)} {within:.3f} {int(passed)}")
    print(f"Calibration views meeting the retrospective consistency check: {sum(observed)}/{len(observed)}")
    print("The two fresh repeat cycles used the same order of interventions, with retained before removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_isaac_evaluation_zip", type=Path)
    parser.add_argument("native_calibration_zip", type=Path)
    args = parser.parse_args()
    with ZipFile(args.native_isaac_evaluation_zip) as archive:
        main_pairs(archive)
    with ZipFile(args.native_calibration_zip) as archive:
        calibration(archive)
