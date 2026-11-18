"""Analytic (geometry-only, no rendering) check of the tabletop design cases TDES0-3 of a config: per stop, the convex-hull
coverage of the design mug by the placed board and the pattern margin (px; H stops: how far the mug's projected hull lies
inside the board's, C stops: how far apart they are) at the NOMINAL cameras and at the case's fixed REVISIT cameras
(sim/arrange.py revisit_cameras, the poses the renderer uses for the query phases). The rendered instance masks stay the
authority for the visibility labels.

    python data/raw/2026-11-04_tabletop_scan/design_check.py configs/sim/arrange_tabletop_v1.json OUT.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim import arrange as A  # noqa: E402


def margins(cams, mug, board, pattern):
    out = []
    for cam, ch in zip(cams, pattern):
        t, o = A.projected_hull(cam, mug.shape_points()), A.projected_hull(cam, board.shape_points())
        out.append(A._inset(t, o) if ch == "H" else A._sat_gap(t, o))
    return out


def main(cfg_path: str, out: str) -> int:
    ctx = A.load_context(cfg_path)
    dc = ctx.config["design_cases"]
    mug, _ = A.design_mug(ctx)
    res = {"config": {"path": cfg_path, "sha256": A.sha256_file(A._repo_path(cfg_path))},
           "generator_sha256": A.sha256_file(A.THIS_FILE),
           "scope": ("geometry only: nominal pinhole convex hulls through the nominal cameras and through each case's fixed revisit "
                     "cameras; no rendering; rendered masks remain authoritative"), "cases": {}}
    for case in dc["cases"]:
        board = A.design_board(ctx, case["family"], mug, float(case["gap_m"]), float(case["shift_m"]))
        rec, rcams = A.revisit_cameras(ctx, ctx.rig, case["seed"], mug.shape_points())
        pat = case["intended_pattern"]
        names = [c.name for c in ctx.cams]
        nom_cov = [A.coverage(c, mug, board) for c in ctx.cams]
        rev_cov = [A.coverage(c, mug, board) for c in rcams]
        nom_m, rev_m = margins(ctx.cams, mug, board, pat), margins(rcams, mug, board, pat)
        h = [i for i, ch in enumerate(pat) if ch == "H"]
        c = [i for i, ch in enumerate(pat) if ch == "C"]
        res["cases"][case["arrangement_id"]] = {
            "family": case["family"] + (f"-{case['m16_side']}" if case["m16_side"] else ""), "intended_pattern": pat,
            "gap_m": case["gap_m"], "shift_m": case["shift_m"], "config_margin_px": case["margin_px"],
            "nominal": {n: {"coverage": round(v, 4), "margin_px": round(m, 3)} for n, v, m in zip(names, nom_cov, nom_m)},
            "revisit": {n: {"coverage": round(v, 4), "margin_px": round(m, 3), "redraws": rec["stops"][n]["n_rejected"]}
                        for n, v, m in zip(names, rev_cov, rev_m)},
            "revisit_min_coverage_at_H": round(min(rev_cov[i] for i in h), 4) if h else None,
            "revisit_max_coverage_at_C": round(max(rev_cov[i] for i in c), 4) if c else None,
            "revisit_pattern": "".join("H" if v >= 0.95 else ("C" if v <= 0.10 else "p") for v in rev_cov),
        }
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["cases"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
