"""Writes the tabletop generator configs (tabletop wave 2A).

    python data/raw/2026-11-04_tabletop_scan/make_tabletop_config.py base
        -> data/raw/2026-11-04_tabletop_scan/base_config.json: the config the board-height scan runs on (committed WITH the
           scan plan, before the scan). Board heights are placeholders (null): the scan sets every board family's height_m
           to the scanned height (and IND28's width_m to the scanned width). slots.target_assignment lists every target for
           every class (provisional; the scan and the class matrix decide).
    python data/raw/2026-11-04_tabletop_scan/make_tabletop_config.py final --decision D.json --matrix M.json --design G.json
        -> configs/sim/arrange_tabletop_v1.json: the base config with the scan's H_T (and IND28 width, ycb_box choices), the
           class matrix's target_assignment / excluded, and the design cases' gap / shift from the margin scan.

Everything not decided by the scan is copied from configs/sim/arrange_shelf_v3.json (objects, rest poses, rules, retry
maximum, revisit ranges) or set here from the design proposal / DECISIONS 2026-11-04 (comments next to each value).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SHELF = REPO / "configs" / "sim" / "arrange_shelf_v3.json"
BASE_OUT = HERE / "base_config.json"
FINAL_OUT = REPO / "configs" / "sim" / "arrange_tabletop_v1.json"
TARGETS = [4, 9, 14, 21, 6]
CLASSES = ["IND28", "M16-L", "M16-R", "S8", "white_box", "ycb_box"]
NAMES = {4: "tomato_soup_can", 9: "potted_meat_can", 14: "mug", 21: "foam_brick", 6: "tuna_fish_can"}


def _reserve(start: int, n: int) -> list[int]:
    return list(range(start, start + n))


def base() -> dict:
    s = json.loads(SHELF.read_text())
    sm16 = s["occluder"]["families"]["M16"]
    c = {
        "schema": s["schema"],
        "name": "tabletop_v1",
        "status": ("DEV config, 2026-11-04 (tabletop environment; researcher's decisions of 2026-11-04, "
                   "progress/drafts/2026-11-04-tabletop-design.md). NOT FROZEN. Board height H_T, the IND28 width, the ycb_box "
                   "choices and the per-class target lists come from the board-height scan and class matrix of "
                   "data/raw/2026-11-04_tabletop_scan/ (plan committed before the run)."),
        "environment": "tabletop",
        "geometry": "configs/sim/tabletop_geometry.json",
        "positions": "configs/sim/positions_tabletop_A.json",
        "seeds": copy.deepcopy(s["seeds"]),
        "ids": {"dev_prefix": "TD", "design_prefix": "TDES",
                "note": ("TD + 4-digit dev seed actually used (TD5003 = seed 5003; a slot replaced from the reserve list is named "
                         "after the reserve seed); TDES0..TDES3 = the tabletop design cases (seeds design-4..design-7). Disjoint "
                         "from the shelf's SD / DES; both match verify's ^[A-Z][A-Z0-9]{1,15}$.")},
        "objects": copy.deepcopy(s["objects"]),
        "rest_poses": copy.deepcopy(s["rest_poses"]),
        "clearances_m": {"table_edge": 0.02, "between_objects": 0.01, "camera_depth": 0.05,
                         "note": ("table_edge: every footprint (target, occluders placed and parked, distractors) at least this "
                                  "far inside the table's edges (design proposal section 3: 2 cm from the edge); between_objects as "
                                  "the shelf; camera_depth: every hull point at least this far in front of every stop's camera "
                                  "(sim/arrange.py in_front_of_cameras). No ceiling.")},
        "target": {
            "objects": list(TARGETS),
            "max_height_m": s["target"]["max_height_m"],
            "max_height_note": "the shelf's limit kept for parity (the five targets are <= 10.2 cm)",
            "region_m": {"x": [-0.15, 0.15], "y": [-0.05, 0.15]},
            "region_note": ("base-centre sampling box (design proposal section 3); a draw is kept only if the target is on the "
                            "table and every stop sees it whole: every hull point projects frame_margin_px inside the frame, "
                            "pinhole and with the fitted k1 (no opening test on the table)"),
            "frame_margin_px": 40,
            "tries": 2000,
            "objects_note": "the shelf's five YCB-V targets (design proposal section 3: enables the secondary comparison on common targets)",
        },
        "distractors": {
            "count": [2, 5], "p_inner": 0.5, "inner_radius_m": 0.20,
            "outer_region_m": {"x": [-0.58, 0.58], "y": [-0.28, 0.48]},
            "outer_min_stops_in_frame": 1,
            "objects": list(range(1, 22)),
            "tries_each": 200,
            "rule": ("2-5 distinct upright models, never the target's or the occluder's; per try 'inner' with probability p_inner "
                     "(base centre uniform over the 0.20 m workspace disc) or 'outer' (base centre uniform over outer_region_m, "
                     "outside the disc and inside the frame at >= outer_min_stops_in_frame stops: on the seen table); on the table "
                     "with table_edge clearance, footprint between_objects clear of the target, the placed occluder and earlier "
                     "distractors, and never occluding the target from any stop (projected hulls overlap only if the distractor is "
                     "wholly behind the target)"),
        },
        "occluder": {
            "p3_coverage_min": s["occluder"]["p3_coverage_min"],
            "coverage_note": s["occluder"]["coverage_note"],
            "min_clear_stops": s["occluder"]["min_clear_stops"],
            "clear_coverage_max": s["occluder"]["clear_coverage_max"],
            "stop_rule_note": s["occluder"]["stop_rule_note"],
            "families": {
                "S8": {"kind": "board", "width_m": 0.08, "thickness_m": 0.005, "height_m": None,
                       "gap_m": [0.01, 0.3], "shift_m": [-0.015, 0.015],
                       "note": "8 cm x H_T board, 0.5 cm thick (the physical S8's width and thickness); symmetric shift about the start-stop sight line (shelf values)"},
                "M16": {"kind": "board", "width_m": 0.16, "thickness_m": 0.005, "height_m": None,
                        "gap_m": [0.01, 0.3], "side_shift_m": list(sm16["side_shift_m"]), "tries": sm16["tries"],
                        "side_rule": {k: v for k, v in sm16["side_rule"].items()},
                        "note": "16 cm x H_T board; the shelf_v3 side rule and ranges (SIDE R = displaced toward +x, hides P3 and P4: CCHHC; L hides P2 and P3: CHHCC)"},
                "IND28": {"kind": "board", "width_m": 0.28, "thickness_m": 0.005, "height_m": None,
                          "gap_m": [0.01, 0.04], "shift_m": [-0.02, 0.02], "predeclared_indistinguishable": True,
                          "note": "28 cm x H_T board (a wider tabletop variant only if the scan's rule picks one); must cover the target at every stop"},
                "white_box": copy.deepcopy(s["occluder"]["families"]["white_box"]),
                "ycb_box": {"kind": "ycbv", "choices": [[2, "upright"], [3, "upright"], [16, "upright"], [7, "edge"]],
                            "gap_m": [0.005, 0.3], "shift_m": [-0.03, 0.03],
                            "note": ("design proposal section 4: upright cracker box (7.2 x 16.4 x 21.3 cm), upright sugar box "
                                     "(5.0 x 9.4 x 17.6), upright wood block (10.4 x 10.4 x 20.6), pudding box on edge (13.8 x 3.9 x 12.9); "
                                     "the scan decides which are kept")},
            },
            "tries_per_family": s["occluder"]["tries_per_family"],
            "parked": {
                "region_m": {"x": [-0.58, 0.58], "y": [-0.28, 0.48]},
                "tries": 600,
                "target_window_context": 1.0,
                "min_stops_in_frame": 1,
                "shadow": {"light_grid_n": 5, "target_step_m": 0.01, "target_levels": 6},
                "rule": ("the occluder at yaw 0, base centre uniform over region_m: on the table, footprint clear of the target and "
                         "distractors, base centre inside the frame at >= min_stops_in_frame stops, outside the target's context window "
                         "at every stop, hiding no object from any stop (as the shelf), and free of the parked-shadow nuisance "
                         "(sim/arrange.py parked_reason): (i) initial screen: its hull projected from each light's centre onto z = 0 "
                         "must not overlap the target window's table footprint at any stop; (ii) ray check: no segment from a "
                         "light_grid_n x light_grid_n grid over each rect light that is on in some preset to the target's full-height "
                         "samples (prism over its hull outline, every target_step_m, at target_levels heights, plus a target_step_m "
                         "grid on its top) meets the occluder's box (analytic slab test). Shadows of the occluder MOVING between "
                         "phases are legitimate and kept (decision 2026-11-04 item 6): this is nuisance control only."),
            },
            "gap_note": s["occluder"]["gap_note"].replace("front lip", "table's near edge (table_edge)"),
        },
        "slots": {
            "note": s["slots"]["note"],
            "retry_max": s["slots"]["retry_max"],
            "dev": {"permutation_seed": 20261107, "primary_seed_start": 5000,
                    "quota": {"S8": 12, "M16": {"L": 9, "R": 9}, "IND28": 6, "white_box": 12, "ycb_box": 12},
                    "reserve": {"S8": _reserve(6000, 12), "M16": _reserve(6100, 18), "IND28": _reserve(6200, 6),
                                "white_box": _reserve(6300, 12), "ycb_box": _reserve(6400, 12)},
                    "target_seed": 20261109},
            "test": {"permutation_seed": 20261108, "primary_seed_start": 110000,
                     "quota": {"S8": 20, "M16": {"L": 15, "R": 15}, "IND28": 10, "white_box": 20, "ycb_box": 20},
                     "reserve": {"S8": _reserve(111000, 20), "M16": _reserve(111100, 30), "IND28": _reserve(111200, 10),
                                 "white_box": _reserve(111300, 20), "ycb_box": _reserve(111400, 20)},
                     "target_seed": 20261110},
            "target_assignment": {"rule": s["slots"]["target_assignment"]["rule"],
                                  "classes": {k: list(TARGETS) for k in CLASSES},
                                  "model_ids": {str(t): NAMES[t] for t in TARGETS},
                                  "excluded": {}, "excluded_note": "provisional (base config): every target in every class"},
        },
        "revisit": {k: copy.deepcopy(v) for k, v in s["revisit"].items() if k != "note"},
        "materials": {"white_box": {"rgba": list(s["materials"]["white_box"]["rgba"]),
                                    "note": "white (decision 2026-11-04 item 4: the white box stays white; the shelf's white box colour)"}},
        "design_cases": None,
        "scene": {"tags": False,
                  "note": "no fiducials (decision 2026-11-03, kept for the tabletop)",
                  "frame_check": ("Revisit and nominal frame check: the workspace outline (sim/tabletop.py workspace_outline, 24 points on "
                                  "the 0.20 m circle at z = 0 and z = 0.12) and the target's projected hull inside the frame with margin >= "
                                  "revisit.min_margin_px (0), fitted k1 (revisit.distort)")},
    }
    c["revisit"]["stream_salt"] = 2094307861        # new stream salt for the tabletop (shelf: 1383428169)
    c["revisit"]["note"] = ("The shelf_v3 revisit protocol (ranges, min margin, k1, max_redraws) with a new stream_salt; the entropy is "
                            "salted by environment (sim/arrange.py revisit_entropy: [ENTROPY_SALT['tabletop'], seed]) so equal seeds in "
                            "the two environments never share draws. Frame check: scene.frame_check.")
    return c


DESIGN_CASES = [  # (seed, id, family, side, intended pattern): design proposal section 4, DECISIONS 2026-11-04
    ("design-4", "TDES0", "S8", None, "CCHCC"),
    ("design-5", "TDES1", "M16", "R", "CCHHC"),
    ("design-6", "TDES2", "M16", "L", "CHHCC"),
    ("design-7", "TDES3", "IND28", None, "HHHHH"),
]
PARK_ORIGIN = (-0.40, 0.35)       # design parking: the accepted grid spot nearest the table's back-left area


def set_boards(c: dict, height: float, ind28_width: float) -> None:
    for f in ("S8", "M16", "IND28"):
        c["occluder"]["families"][f]["height_m"] = float(height)
    c["occluder"]["families"]["IND28"]["width_m"] = float(ind28_width)


def design_block(c: dict) -> dict:
    """The tabletop design cases: margin-maximising (gap, shift) per case (sim/arrange.py design_margin_scan) and one fixed
    parking spot valid for all four boards (parked_reason)."""
    sys.path.insert(0, str(REPO))
    from sim import arrange as A

    dc = {"note": ("Tabletop design cases TDES0-TDES3 (design proposal section 4): the measured study mug of configs/rig_geometry.json "
                   "(profile + handle plate) at the workspace centre, handle toward P3, no distractors; the family's board at H_T "
                   "(IND28 at its width) at the gap / shift that maximises the smallest per-stop pixel margin of the intended "
                   "pattern over the margin_scan grid (sim/arrange.py design_margin_scan: inset of the mug's projected hull in the "
                   "board's at H stops, separation at C stops); S8 CCHCC, M16-R CCHHC, M16-L CHHCC, IND28 HHHHH (predeclared "
                   "indistinguishable). Parked at one fixed spot for all four (parked: the first spot of a 5 cm grid over the "
                   f"parking region, nearest {PARK_ORIGIN}, that parked_reason accepts for every case). Fixed, predeclared dev "
                   "arrangements outside the 60 slots."),
          "mug_geometry": "configs/rig_geometry.json", "mug_x": 0.0, "mug_y": 0.0, "handle": "toward", "target_name": "study_mug",
          "margin_scan": {"gap_m": [0.005, 0.25, 0.005], "shift_m": [-0.15, 0.15, 0.005]},
          "parked": None, "cases": []}
    c = copy.deepcopy(c)
    c["design_cases"] = dc
    ctx = A.load_context(c)
    mug, _ = A.design_mug(ctx)
    for seed, aid, fam, side, pat in DESIGN_CASES:
        best = A.design_margin_scan(ctx, fam, pat, mug)
        case = {"seed": seed, "arrangement_id": aid, "family": fam, "m16_side": side, "intended_pattern": pat,
                "gap_m": best["gap_m"], "shift_m": best["shift_m"], "margin_px": best["margin_px"],
                "stop_margins_px": best["stop_margins_px"], "scan_grid_n": best["grid"]["n"]}
        if fam == "IND28":
            case["predeclared_indistinguishable"] = True
        dc["cases"].append(case)
        print(f"{aid} {fam}{'-' + side if side else ''} {pat}: gap {best['gap_m']} shift {best['shift_m']} margin {best['margin_px']} px "
              f"{best['stop_margins_px']}", flush=True)
    lo_x, hi_x = c["occluder"]["parked"]["region_m"]["x"]
    lo_y, hi_y = c["occluder"]["parked"]["region_m"]["y"]
    import numpy as np
    spots = [(round(float(x), 3), round(float(y), 3)) for x in np.arange(-0.55, 0.551, 0.05) for y in np.arange(-0.25, 0.451, 0.05)
             if lo_x <= x <= hi_x and lo_y <= y <= hi_y]
    spots.sort(key=lambda p: (round((p[0] - PARK_ORIGIN[0]) ** 2 + (p[1] - PARK_ORIGIN[1]) ** 2, 9), p))
    ss = A.shadow_setup(ctx, mug)
    for x, y in spots:
        ok = True
        for case in dc["cases"]:
            placed = A.design_board(ctx, case["family"], mug, case["gap_m"], case["shift_m"])
            if A.parked_reason(ctx, A.Body(placed.size, (x, y, 0.0), 0.0), mug, [], ss) is not None:
                ok = False
                break
        if ok:
            dc["parked"] = {"x": x, "y": y, "note": f"first accepted spot of the 5 cm grid nearest {PARK_ORIGIN} (checked by design_arrangement)"}
            break
    if dc["parked"] is None:
        raise SystemExit("no design parking spot valid for all four boards")
    print("design parked", dc["parked"], flush=True)
    return dc


def final(a) -> int:
    """configs/sim/arrange_tabletop_v1.json from the scan decision and the class matrix."""
    dec = json.loads(Path(a.decision).read_text())
    mx = json.loads(Path(a.matrix).read_text())
    if dec["H_T_m"] is None:
        raise SystemExit(f"refused: the scan's outcome is '{dec['outcome']}', no H_T to write")
    H, W = float(dec["H_T_m"]), float(dec["ind28_width_m"])
    kept = [[int(k.split(":")[0]), k.split(":")[1]] for k, v in dec["ycb_box"].items() if v["kept"]]
    ov = mx.get("overrides", {})
    want = {**{f"occluder.families.{f}.height_m": H for f in ("S8", "M16", "IND28")},
            "occluder.families.IND28.width_m": W, "occluder.families.ycb_box.choices": kept}
    if ov != want:
        raise SystemExit(f"refused: the class matrix was run with {ov}, not the decision's {want}")
    c = base()
    c["status"] = ("DEV config, 2026-11-04 (tabletop environment; researcher's decisions of 2026-11-04, "
                   "progress/drafts/2026-11-04-tabletop-design.md). NOT FROZEN. H_T, the IND28 width, the ycb_box choices and the "
                   "per-class target lists are the results of data/raw/2026-11-04_tabletop_scan/ (scan_plan.json committed before "
                   "the scan; results.json, decision.json, matrix.json).")
    set_boards(c, H, W)
    fam = c["occluder"]["families"]
    hn = (f"H_T = {H:g} m: {dec['outcome']} (data/raw/2026-11-04_tabletop_scan/decision.json; plan scan_plan.json sha256 "
          f"{dec['plan_sha256']})")
    for f in ("S8", "M16", "IND28"):
        fam[f]["height_note"] = hn
    if W > 0.28:
        fam["IND28"]["variant"] = (f"TABLETOP VARIANT: {W:g} m wide x {H:g} m tall x 0.005 m thick (the physical IND28 is 0.28 m "
                                   "wide); the narrowest scanned width meeting the scan's rule (b) at H_T")
    fam["ycb_box"]["choices"] = kept
    fam["ycb_box"]["choices_note"] = ("kept by the scan (a choice is kept if it admits >= 1 target in 100 tries): "
                                      + ", ".join(f"{k} {'kept' if v['kept'] else 'dropped'} (admits {v['admitted']})"
                                                  for k, v in dec["ycb_box"].items()))
    n = int(mx["tries_per_cell"])
    classes, excluded = {}, {}
    for r in mx["rows"]:
        if r["ok"] >= 1:
            classes.setdefault(r["class"], []).append(int(r["target"]))
        else:
            excluded.setdefault(r["class"], {})[str(r["target"])] = (
                f"{NAMES[int(r['target'])]}: no valid placement found in {n} tries (0 of {n}; fails at {r['fail_reasons']})")
    order = {t: i for i, t in enumerate(TARGETS)}
    ta = c["slots"]["target_assignment"]
    ta["classes"] = {k: sorted(v, key=order.get) for k, v in sorted(classes.items())}
    ta["excluded"] = excluded
    ta["excluded_note"] = (f"class matrix at H_T (data/raw/2026-11-04_tabletop_scan/matrix.json: {n} real generator tries per (class, "
                           f"target), probe seeds {mx['probe_seeds'][0]}-{mx['probe_seeds'][1]}); zero in N tries is not a proof "
                           "that no placement exists")
    missing = [k for k in CLASSES if not ta["classes"].get(k)]
    if missing:
        raise SystemExit(f"refused: classes with no admitted target: {missing}")
    c["design_cases"] = design_block(c)
    Path(a.out).write_text(json.dumps(c, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("base")
    fx = sub.add_parser("fixture", help="tests/fixtures/tabletop/arrange_tabletop_fixture.json: base + boards at 0.20 m + design cases (TESTS ONLY)")
    fx.add_argument("--out", default=str(REPO / "tests" / "fixtures" / "tabletop" / "arrange_tabletop_fixture.json"))
    fi = sub.add_parser("final")
    fi.add_argument("--decision", required=True)
    fi.add_argument("--matrix", required=True)
    fi.add_argument("--out", default=str(FINAL_OUT))
    a = ap.parse_args(argv)
    if a.cmd == "base":
        c = base()
        BASE_OUT.write_text(json.dumps(c, indent=1) + "\n")
        print(f"wrote {BASE_OUT}")
        return 0
    if a.cmd == "fixture":
        c = base()
        c["name"] = "tabletop_test_fixture"
        c["status"] = ("TEST FIXTURE ONLY (tests/test_sim_arrange_tabletop.py): the tabletop base config with every board at 0.20 m (IND28 0.44 m wide) and "
                       "design cases from the margin scan. Not a study config; the study config is configs/sim/arrange_tabletop_v1.json.")
        set_boards(c, 0.20, 0.44)
        c["design_cases"] = design_block(c)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(c, indent=1) + "\n")
        print(f"wrote {a.out}")
        return 0
    return final(a)


if __name__ == "__main__":
    sys.exit(main())
