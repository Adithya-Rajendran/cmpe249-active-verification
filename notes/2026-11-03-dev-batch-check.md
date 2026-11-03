# Small dev batch check, checkpoint 2 (2026-11-03)

**Simulation results only.** This checks the checkpoint-2 generator and renderer on 10 arrangements before the full dev batch is rendered. Nothing here is a result of the study.

## What was rendered

- **Set:** `/mnt/bench-data/datasets/cmpe249-active-verification/sim/shelf_dev_v1_check/` (called `ROOT` below). `shelf_dev_v0` was not touched.
- **Arrangements:**
  - The first slot of each class in the dev slot list: SD1000 (M16-L, slot 0), SD1001 (IND28, slot 1), SD1002 (S8, slot 2), SD1005 (ycb_box, slot 5), SD1007 (M16-R, slot 7), SD1010 (white_box, slot 10).
  - The four design cases DES0 to DES3.
- **Code:** commit `de7770b` on branch `cp2-sim` (`sim/arrange.py` sha256 `03b48f0f...`, `sim/isaac_batch.py` `4052922b...`). The M16 near-opposite-stop limit was tightened from 0.5 to 0.10 after a first render (see "One-sided M16"); that first render was deleted and everything here is from the second.
- **Run:** one Isaac process on the GB10, `ROOT/render_report_20261103T000139.json` (Isaac Sim 6.2.0.0, `rt_subframes` 16, 100 updates on fresh render products before each capture). Exit code 0. Log on the GB10: `~/cmpe249-sim/run/shelf_dev_v1_check.log`.
- **Checks:** `python -m sim.check_batch --root ROOT` wrote `ROOT/check/check.json` (all numbers below) and `check_stdout.txt`. Contact sheets are `ROOT/<ARR>/contact_L1.jpg`, `contact_L2.jpg` and `crops_L1.jpg`. `python -m sim.episodes` built the manifests, jobs and annotations in the same folder without complaint.

## Render time

| | Seconds |
|---|---|
| Per arrangement (`render.json` `total_s`, 10 values) | min 124.46, median 128.55, max 130.60 |
| Whole run, boot to exit (10 arrangements) | 1300 s, i.e. 130 s per arrangement |
| Boot (`render_report` `boot_s`) | 9.3 |

Each arrangement has 8 captures of about 12.3 s, the two target-alone passes (4.6 s together) and a settle of about 24 s (SD1000: `settle_s` 23.79); `render.json` lists every step. This matches the 120.9 to 126.4 s of the old `shelf_dev_v0` (median 124.4 s) within about 4 s. The extra alone pass and the revisit cameras cost little. At 130 s each the 60 dev arrangements need about 2.2 h, and 100 test arrangements about 3.6 h. Inference and retries are not included.

## Revisit poses

Source: `ROOT/check/check.json` `summary.revisit_delta`, from the 50 stop draws of the 10 arrangements (one per arrangement and stop; also in each `arrangement.json` under `revisit`, in `labels.json`, and in `set.json`).

| Delta | Min | Max | Range |
|---|---|---|---|
| dx, dy, dz (mm) | -9.99 (dz) | +9.94 (dy) | +-10 |
| dyaw, dpitch, droll (deg) | -0.499 (dpitch) | +0.496 (dyaw) | +-0.5 |

- **Redraws:** 6 of the 50 stops needed a redraw, 14 draws were rejected in all: 10 at P1 and 4 at P5, none at P2 to P4. P1 and P5 are the stops with the narrow pitch windows.
- **Frame margin:** every used pose has a minimum margin of at least 0.79 px (analytic, fitted k1; SD1001 P1). Margins at the narrow stops are 0.79 to 17.1 px at P1 and 2.7 to 20.5 px at P5 (per arrangement in `check.json`).
- **Rendering has no k1:** the rendered tag corners are at least 7.5 px from the frame edge in every query capture (median 59.7 px), against at least 20.2 px in the references.

## Tags and registration

- **Tag detection:** tags 0 to 3 were found in all 400 captures (100 nominal reference captures, 300 query captures). No extra marker ids appeared in this render. (The first render had one extra id 17 in a capture, which my first version of the check counted as a miss.)
- **Registration of every query onto its L1 reference** (`capture/tags.py align`, the call `perception/evidence.py` makes): 300 of 300 queries registered, all with 4 tags and none dropped.

| Quantity (300 views) | Min | Median | Max |
|---|---|---|---|
| In-fit RMS (px) | 0.15 | 0.25 | 0.47 |
| Leave-one-tag-out error (px) | 0.82 | 2.55 | 6.82 |
| Target mask IoU after warping the visible-phase mask through H, against the reference mask | 0.882 | 0.951 | 0.993 |
| The same IoU without registration | 0.401 | 0.691 | 0.928 |
| Centroid offset of the warped mask (px) | 0.22 | 1.71 | 4.18 |

The cut-offs of `capture/tags.py` (RMS 10, leave-one-out 30) are far from these values. The target is 5 to 30 cm behind the tag plane, so a few px of offset at the target remain after registration (centroid offset up to 4.2 px).

## Do visible-phase queries differ from their references?

Yes. Each visible capture is compared with the reference at the same stop and lighting (100 pairs, grey levels, `check.json` `visible_vs_reference`).

| | Pixel-identical | Mean absolute difference (grey) | Pixels differing by more than 8 grey (median) |
|---|---|---|---|
| v1, revisit poses | 0 of 100 | min 1.26, median 4.56, max 11.05 | 8.2 % |
| v0, same pose (`ROOT/check/v0_baseline.json`, 100 pairs of `shelf_dev_v0`) | 0 of 100 | min 0.09, median 0.11, max 0.57 | 0.004 % |

Two renders of an unchanged scene are never identical either (render noise, mean difference 0.1 grey), but the revisit pose makes the difference about 40 times larger. After registering the visible capture onto the reference, inside the tag rectangle, the mean difference is 0.58 to 2.28 grey (median 1.21): parallax of the shelf interior and the target offset remain.

## Patterns and visible fractions

Patterns are P1..P5 from the rendered occluded captures at the revisit poses (H <= 0.05, C >= 0.90, else p). "Analytic" is the generator's convex-hull coverage at the nominal poses. Values are from `check.json` `arrangements[].labels`.

| Arrangement | Class | Target | Occluded visible fraction P1 / P2 / P3 / P4 / P5 | Pattern | Diagnostic views (visible / occluded / removed) |
|---|---|---|---|---|---|
| DES0 | S8 | study mug | 1.00 / 1.00 / 0.00 / 1.00 / 1.00 | CCHCC | 5 / 4 / 4 |
| DES1 | M16 (R) | study mug | 1.00 / 0.997 / 0.00 / 0.00 / 1.00 | CCHHC | 5 / 3 / 3 |
| DES2 | M16 (L) | study mug | 1.00 / 0.003 / 0.00 / 1.00 / 1.00 | CHHCC | 5 / 3 / 3 |
| DES3 | IND28 | study mug | 0 at every stop | HHHHH | 5 / 0 / 0 |
| SD1000 | M16-L | tuna can | 1.00 / 0.14 / 0.00 / 0.91 / 1.00 | CpHCC | 5 / 3 / 3 |
| SD1007 | M16-R | foam brick | 1.00 / 1.00 / 0.00 / 0.00 / 1.00 | CCHHC | 5 / 3 / 3 |
| SD1001 | IND28 | foam brick | 0 at every stop | HHHHH | 5 / 0 / 0 |
| SD1002 | S8 | meat can | 1.00 / 0.38 / 0.045 / 0.81 / 1.00 | CpHpC | 5 / 2 / 2 |
| SD1005 | ycb_box | foam brick | 1.00 / 0.69 / 0.00 / 0.35 / 1.00 | CpHpC | 5 / 2 / 2 |
| SD1010 | white_box | tomato can | 1.00 / 0.62 / 0.00 / 0.55 / 1.00 | CpHpC | 5 / 2 / 2 |

- **Reference and visible patterns** are CCCCC in all 10 arrangements.
- **Rendered coverage matches the generator's:** 1 minus the rendered visible fraction is within 0.03 of the analytic coverage at every stop of all 10 arrangements, and within 0.021 for the six random ones (for example SD1000 P2: analytic 0.853, rendered 0.864).
- **Removed views:** the occluder's share of the visible-phase region (`region_obstruction.removed`) follows the same numbers (SD1007: 0, 0, 1.00, 1.00, 0 for P1 to P5). Eight removed views are partly obstructed (0.02 to 0.9). The only one near the 10 % cut-off is SD1000 P4 at 0.088 (diagnostic); the next lowest is 0.194.
- **S8, ycb_box and white_box** still give the centred CpHpC pattern, because the generator requires only 2 clear stops for them. Only M16 is one-sided by design.

## One-sided M16

- **Both generated M16 arrangements are one-sided:** SD1000 (L) has non-C stops only at P2 and P3, and SD1007 (R) only at P3 and P4 (`rendered_side` in `check.json`: L and R). The far stops P1 and P5 are C. DES1 (R) and DES2 (L) reproduce the design patterns CCHHC and CHHCC at the revisit poses.
- **First version:** the generator allowed the nearest opposite stop to be covered up to 0.5. The first render gave SD1000 = CpHpC and SD1007 = CpHHC (both sides touched, P4 at 0.78 and P2 at 0.62 visible). The limit is now 0.10, so that stop renders as a C view. Only the 0.10 version is in the committed config.
- **Side rule coverage:** across the 60 dev slots (generation only, `ROOT/check/dev60_generation/`), all 18 M16 arrangements satisfy the rule on the analytic coverage: the far opposite stop (P1 for R, P5 for L) has coverage 0 and the near opposite stop at most 0.09.

## Generation log for the whole dev list (generation only, no rendering)

`python -m sim.arrange --split dev --slots all --design` took 5 min 34 s on the Jetson, while another job used it (`ROOT/check/dev60_generation/time.txt`).

| Class | Slots | With retries | Failed tries | Replaced from reserve |
|---|---|---|---|---|
| S8 | 12 | 7 | 11 | 0 |
| M16-L | 9 | 8 | 34 | 0 |
| M16-R | 9 | 9 | 77 | 1 (slot 57: 21 failed tries, replaced by reserve seed 2100) |
| IND28 | 6 | 3 | 12 | 0 |
| white_box | 12 | 4 | 6 | 0 |
| ycb_box | 12 | 7 | 25 | 0 |

- **Totals:** 165 failed tries and 1 replacement in the 60 slots, no class ever changed (`generation_log.json`).
- **Failed tries by reason:** M16 111, ycb_box 25, S8 11, IND28 12 (8 parked, 4 occluder), white_box 6, all in the slot's own family.
- **M16 targets:** only the foam brick and the tuna can occur (18 of 18). A one-sided board that must also hide the next stop needs a large gap, and the 12 cm board then cannot cover the taller targets (tomato can, meat can, mug) from the neighbouring stop.
- **The 6 check slots:** 28 failed tries, no replacement (`ROOT/generation_log.json`).

## Not checked

- Full-batch throughput with inference and retries.
- The tabletop environment.
- Evidence scores on these views (no evidence run was made).
