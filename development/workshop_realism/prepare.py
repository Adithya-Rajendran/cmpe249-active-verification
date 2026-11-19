"""Check native interventions, cameras and exact input pairs, then release only anonymous images."""

import argparse
import csv
import hashlib
import os
import secrets
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from development.paper_audit.inputs_build import box_from_mask, clean
from development.workshop_realism.release import (
    HERE,
    OLD,
    read,
    require,
    sha,
    validate_design,
    write_once,
)


def csv_file(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pixels(path, mode):
    with Image.open(path) as img:
        if img.mode != mode:
            raise ValueError(f"Expected {mode} in {path}")
        return np.asarray(img).copy()


def pixhash(array):
    return hashlib.sha256(
        str(array.shape).encode() + b"\0" + array.tobytes()
    ).hexdigest()


def receipt(folder, engine, item, release, lock):
    name = "completion.json" if engine == "isaac" else "complete.json"
    proof = read(folder / name)
    outputs = proof["outputs"]
    native_files = {
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.is_file() and p.name != name
    }
    require(
        native_files == set(outputs),
        "Native completion inventory mismatch in " + str(folder),
    )
    for rel, digest in outputs.items():
        path = folder / rel
        require(
            path.resolve().is_relative_to(folder.resolve()) and sha(path) == digest,
            "Native source changed: " + str(path),
        )
    if engine == "mujoco":
        b = proof["identity"]
        require(
            b["arrangement_sha256"] == item["sha256"]
            and b["workshop_renderer_sha256"] == release["source_files"]["capture.py"]
            and b["declaration_sha256"] == lock["declaration_sha256"]
            and b["manifest_sha256"] == release["cohort_manifest_sha256"],
            "MuJoCo renderer or design binding differs",
        )
    else:
        observed = read(folder / "observed.json")
        binding_path = folder.parents[2] / "isaac-binding.json"
        binding = read(binding_path)
        require(
            observed["arrangement_sha256"] == item["sha256"]
            and observed["binding_sha256"]
            == proof["binding_sha256"]
            == sha(binding_path),
            "Isaac scene binding differs",
        )
        require(
            binding["workshop_renderer_sha256"] == release["source_files"]["capture.py"]
            and binding["qualification_declaration_sha256"]
            == lock["declaration_sha256"]
            and binding["manifest_sha256"] == release["cohort_manifest_sha256"],
            "Isaac renderer or design binding differs",
        )
    return proof


def verify_observed(native, plain, arr, engine):
    out = []
    for group, spec_key in [("nominal", "cameras"), ("revisit", "revisit_cameras")]:
        require(
            set(native["cameras"][group]) == set(arr["stops"]),
            "Wrong native camera set",
        )
        for expected in arr["scene"][spec_key]:
            v = expected["name"]
            actual = native["cameras"][group][v]
            for key in (
                "position",
                "forward",
                "up",
                "right",
                "f_px",
                "width",
                "height",
                "principal_point_px",
            ):
                tol = 0.01 if key in ("f_px", "principal_point_px") else 0.000005
                require(
                    np.allclose(expected[key], actual[key], atol=tol, rtol=0),
                    "Native camera mismatch: " + key,
                )
                if v in ("P1", "P2", "P3"):
                    require(
                        np.allclose(
                            plain["cameras"][group][v][key],
                            actual[key],
                            atol=tol,
                            rtol=0,
                        ),
                        "Different actual camera in matched comparison",
                    )
            out.append(
                {
                    "engine": engine,
                    "layout": arr["base_id"],
                    "level": arr["level"],
                    "camera": group,
                    "view": v,
                    "matched_old": v in ("P1", "P2", "P3"),
                    **{
                        k: actual[k]
                        for k in ("position", "forward", "up", "right", "f_px")
                    },
                }
            )
    for s in ("clear_present", "clear_removed", "placed_present", "placed_removed"):
        present = s.endswith("_present")
        door = "placed" if s.startswith("placed") else "parked"
        if engine == "mujoco":
            st = native["states"][s]
            visible = st["target_enabled"]
            door_pos = st["blocker"]["position"]
        else:
            st = native["usd_per_state"][s]
            visible = st["/World/Objects/target"]["visibility"] != "invisible"
            door_pos = st["/World/Objects/occluder"]["world_from_local_row"][3][:3]
        require(
            visible == present
            and np.allclose(
                door_pos, arr["objects"]["occluder"]["poses"][door]["base"], atol=1e-5
            ),
            "Measured physical state differs",
        )
    return out


def prepare(run):
    run = run.resolve()
    release = read(run / "design-release.json")
    lock = read(run / "endpoint-lock.json")
    decl = read(HERE / lock.get("declaration_file", "declaration.json"))
    cohort = read(run / "cohort/manifest.json")
    for key, value in validate_design(run).items():
        require(release[key] == value, "Design binding changed")
    require(
        sha(run / "fixture-review.json") == release["fixture_review_sha256"],
        "Different approved fixture",
    )
    packet = run / "scoring/fresh-packet"
    audit = run / "qualification"
    require(not packet.exists() and not audit.exists(), "Packet already built")
    packet.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run) as spooling:
        staging = Path(spooling)
        staged_packet = staging / "fresh-packet"
        staged_packet.mkdir()
        staged_audit = staging / "qualification"
        staged_audit.mkdir()
        cache = {}
        public = []
        private = []
        pairs = []
        poses = []
        markers = {}

        @lru_cache(maxsize=800)
        def rawhash(p):
            return sha(p)

        def export(source, dest):
            source_hash = rawhash(source)
            if source_hash in cache:
                prev, encoded, dhash = cache[source_hash]
                os.link(prev, dest)
                return encoded, dhash
            clean(source, dest)
            with Image.open(dest) as im:
                require(
                    im.size == (1200, 675) and im.mode == "RGB" and not im.info,
                    "Incorrect anonymous RGB export",
                )
                decoded = pixhash(np.asarray(im))
            encoded = sha(dest)
            cache[source_hash] = (dest, encoded, decoded)
            return encoded, decoded

        for engine in decl["engines"]:
            for entry in cohort["arrangements"]:
                layout = entry["base_id"]
                base = run / "operator" / engine / layout
                old = OLD / "operator" / engine / layout
                for level, item in entry["levels"].items():
                    folder = base / level
                    receipt(folder, engine, item, release, lock)
                    markers[str(folder.relative_to(run))] = sha(
                        folder
                        / ("completion.json" if engine == "isaac" else "complete.json")
                    )
                    arr = read(run / "cohort" / item["path"])
                    poses.extend(
                        verify_observed(
                            read(folder / "observed.json"),
                            read(old / level / "observed.json"),
                            arr,
                            engine,
                        )
                    )
                for view in decl["stops"]:
                    nominal = base / "D0"
                    moved = base / "D6"
                    ref = nominal / "rgb" / f"reference_{view}.png"
                    refmask = nominal / "masks" / f"target_reference_{view}.png"
                    box = box_from_mask(refmask)
                    alone_path = moved / "masks" / f"alone_{view}.png"
                    alone = pixels(alone_path, "L") > 0
                    require(alone.any(), "Empty isolated target")
                    rawnative = pixels(ref, "RGB")
                    require(
                        rawnative.shape == (1296, 2304, 3), "Reference shape changed"
                    )
                    ref_dig = pixhash(rawnative)
                    for placement in ("clear", "placed"):
                        opposite = []
                        rgb = []
                        for truth in ("present", "removed"):
                            state = placement + "_" + truth
                            qry = moved / "rgb" / f"{state}_{view}.png"
                            maskpath = moved / "masks" / f"target_{state}_{view}.png"
                            occ = moved / "masks" / f"blocker_{state}_{view}.png"
                            mask = pixels(maskpath, "L") > 0
                            require(
                                mask.shape == alone.shape
                                and (truth == "present" or not mask.any()),
                                "Unexpected target mask",
                            )
                            uid = secrets.token_hex(12)
                            rn = uid + "-reference.png"
                            qn = uid + "-query.png"
                            rh, rd = export(ref, staged_packet / rn)
                            qh, qd = export(qry, staged_packet / qn)
                            img = pixels(qry, "RGB")
                            require(
                                img.shape == rawnative.shape, "Query dimensions changed"
                            )
                            rgb.append(img)
                            record = {
                                "id": uid,
                                "reference": rn,
                                "query": qn,
                                "box_xyxy": box,
                                "reference_sha256": rh,
                                "query_sha256": qh,
                            }
                            public.append(record)
                            priv = {
                                "id": uid,
                                "engine": engine,
                                "layout": layout,
                                "target": arr["target_id"],
                                "view": view,
                                "state": state,
                                "visible": int(mask.sum()),
                                "isolated": int(alone.sum()),
                                "overlap": int(np.count_nonzero(mask & alone)),
                                "native_reference": ref_dig,
                                "native_query": pixhash(img),
                                "delivered_reference": rd,
                                "delivered_query": qd,
                                "box_xyxy": box,
                                "sources": {
                                    k: {
                                        "path": str(path.relative_to(run)),
                                        "sha256": rawhash(path),
                                    }
                                    for k, path in [
                                        ("reference", ref),
                                        ("reference_mask", refmask),
                                        ("query", qry),
                                        ("query_mask", maskpath),
                                        ("blocker_mask", occ),
                                        ("alone_mask", alone_path),
                                    ]
                                },
                            }
                            private.append(priv)
                            opposite.append(priv)
                        kept, gone = opposite
                        native = all(
                            kept[k] == gone[k]
                            for k in ("native_reference", "native_query")
                        )
                        delivered = all(
                            kept[k] == gone[k]
                            for k in (
                                "delivered_reference",
                                "delivered_query",
                                "box_xyxy",
                            )
                        )
                        difference = np.abs(rgb[0].astype("int16") - rgb[1])
                        changed = int(np.count_nonzero(np.any(difference, axis=2)))
                        pairs.append(
                            {
                                "engine": engine,
                                "layout": layout,
                                "target": arr["target_id"],
                                "view": view,
                                "door": placement,
                                "present_id": kept["id"],
                                "removed_id": gone["id"],
                                "native_exact": native,
                                "delivered_exact": delivered,
                                "exact": native and delivered,
                                "target_pixels": kept["visible"],
                                "isolated_pixels": kept["isolated"],
                                "visible_fraction": kept["overlap"] / kept["isolated"],
                                "changed_native_pixels": changed,
                                "mean_absolute_channel_difference": float(
                                    difference.mean()
                                ),
                            }
                        )
        require(
            len(public) == len({c["id"] for c in public}) == 320
            and len(pairs) == 160
            and len(poses) == 320,
            "Incomplete dimensions",
        )
        secrets.SystemRandom().shuffle(public)
        write_once(staged_packet / "cases.json", public)
        write_once(staged_audit / "private-ledger.json", private)
        csv_file(staged_audit / "input-pairs.csv", pairs)
        csv_file(staged_audit / "measured-cameras.csv", poses)
        write_once(
            staged_audit / "scorer-ready.json",
            {
                "pass": True,
                "queries": len(public),
                "pairs": len(pairs),
                "camera_transforms": len(poses),
                "cases_sha256": sha(staged_packet / "cases.json"),
                "private_ledger_sha256": sha(staged_audit / "private-ledger.json"),
                "pairs_sha256": sha(staged_audit / "input-pairs.csv"),
                "cameras_sha256": sha(staged_audit / "measured-cameras.csv"),
                "design_release_sha256": sha(run / "design-release.json"),
                "task_sha256": lock["original_task_sha256"],
                "completion_hashes": markers,
            },
        )
        staged_packet.rename(packet)
        staged_audit.rename(audit)
    print("Released 320 anonymous cases and checked 160 opposite-state inputs.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    prepare(a.run)
