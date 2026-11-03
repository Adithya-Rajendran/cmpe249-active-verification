# Taller simulation M16, per-slot targets and a render check (2026-11-03)

**Simulation results only.** This records the checkpoint-2 review fixes to `sim/`, the `shelf_v2` generator config, and a small
render check on the GB10. Nothing here is a result of the study. `shelf_v1` and `shelf_dev_v1_check` were not touched.

Evidence lives in `/mnt/bench-data/datasets/cmpe249-active-verification/sim/` (called `SIM` below):
`shelf_v2_geometry/` (feasibility scan and matrices, dev 60 generation, test slot list) and `shelf_dev_v2_check/` (the render check).
Code: branch `cp2-fix-sim`, `sim/arrange.py` sha256 `63603aff...`, `sim/isaac_batch.py` `217ea28b...`, config `shelf_v2`
(canonical-JSON sha256 `bc0e58be...`).

## 1. M16 height: no height admitted every target in the tested search

The student's decision (2026-11-03) was a taller M16 that can hide every target in the list, with the 16 cm width, the one-sided
placement and the side rule kept. `python -m sim.feasibility scan` asks, for each target, side and board height, whether **any**
placement exists under the generator's own rules (target seen whole from every stop; board inside #3's walls and under the
ceiling clearance; P3 hidden >= 0.95; the adjacent same-side stop hidden >= 0.95; the near opposite stop <= 0.10 and the far one
clear; the stops beyond the adjacent one on the board's side <= 0.10; at least 2 clear stops). 60 target placements per
target and side (the same for every height), board gap swept in 1 cm and lateral shift in 0.5 cm steps. The ceiling clearance was
removed for the scan so that heights above #3's limit could be tried. Evidence: `SIM/shelf_v2_geometry/height_scan.json` and
`height_scan.out` (417.6 s). Numbers are feasible (placement, gap, shift) combinations out of the 60 placements.

| Target | Side | 0.10 | 0.11 | 0.12 | 0.13 | 0.14 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|---|---|---|---|---|
| tuna can | L | 59 | 93 | 93 | 93 | 93 | 93 | 93 | 93 |
| tuna can | R | 55 | 74 | 74 | 74 | 74 | 74 | 74 | 74 |
| foam brick | L | 0 | 56 | 173 | 179 | 179 | 179 | 179 | 179 |
| foam brick | R | 0 | 94 | 183 | 184 | 184 | 184 | 184 | 184 |
| mug | L | 0 | 0 | 0 | 0 | 2 | 2 | 2 | 2 |
| mug | R | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| potted meat can | L, R | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tomato soup can | L, R | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **Minimum height that makes every target feasible: none found in the tested search** (zero in N tries or on a grid is not a proof; the mug has two left-side combinations from 0.14 m). The tomato soup can and the potted meat can have no
  placement on either side at any height up to 0.25 m, and the mug has one only on side L (2 combinations, from 0.14 m).
- **The height is not the binding constraint; the side rule and the 2-clear-stop rule are.** A 16 cm board that hides the target at P3 and at the next stop, but leaves the near opposite stop <= 0.10 and one more stop clear, needs the board far
  in front of the target, which needs the target at the back of #3, but a target that deep is not seen whole from the outer
  stops (with the target region restricted to base-centre y 0.22-0.27, `place_target` found no placement for the tomato soup can
  in 600 draws: `SIM/shelf_v2_geometry/deep_probe_tomato.json`). Relaxing rules on the three tall targets at 0.18 m (`SIM/shelf_v2_geometry/relax3.json`, 40 placements,
  combinations; this was run with the first shelf_v2 rule set, before `same_far_max`): the baseline gives tomato 0 / 0,
  meat can 0 / 0, mug 1 / 0 (L / R); no limit on the near opposite stop gives 0 / 0, 10 / 1, 9 / 4; one clear stop instead of
  two gives 0 / 0, 0 / 1, 2 / 4; **both together give 409 / 490, 326 / 399, 465 / 615**; a wider board (20 or 24 cm) gives 0
  everywhere. Both relaxations would make the near opposite stop a partial `p` view, which breaks the intended CHHCC / CCHHC.
- **Chosen height: 0.15 m** (`occluder.families.M16.height_m`), the tallest the ceiling clearance admits (clear height 0.165
  - clearance 0.015 = 0.150). Margins: 0.04 m above the height at which the tuna can's count stops changing (0.11), 0.02 m above
  the foam brick's (0.13); 0 m to the ceiling-clearance limit; 1.5 cm of real clearance remains below the 16.5 cm ceiling. It
  buys nothing for the targets that M16 can serve (0.12 m would admit the same ones); it is a revised simulation variant and
  **no longer matches the physical 10-10 board (12 cm)**, which the README, the config status and `height_note` say. The design
  cases DES1 and DES2 keep the physical boards.

## 2. Side rule: adjacent stop hidden, and the stops beyond it covered <= 0.10

- `adjacent_min` is 0.95 (v1: 0.5), so the adjacent same-side stop is an H view. On the generated dev 60 the adjacent stop's
  analytic coverage is >= 0.956 on all 9 L boards (min 0.956) and all 9 R boards (min 0.997).
- **Added beyond the brief:** `same_far_max` 0.10. With `adjacent_min` alone, the first shelf_v2 dev 60
  (`SIM/shelf_v2_geometry/superseded_adjacent_only/dev60`) had 5 of 18 boards (1 of 9 L, 4 of 9 R) covering the far same-side stop (P1 for L, P5 for R) by 0.03 to
  0.25, and the rendered SD1000 (M16-L) was pHHCC (`SIM/shelf_dev_v2_check_superseded_adjacent_only/`), not CHHCC. The stops
  beyond the adjacent one are now limited to 0.10 (they render as C). Final dev 60: far same-side coverage <= 0.081 (L 0, R
  0.081), near opposite stop <= 0.075 (L) / 0.096 (R), far opposite stop 0 on both sides. It is one optional key; remove it from
  `side_rule` to get the brief's rule exactly. This is a decision for the coordinator (see the report).
- **L and R are symmetric** in the rule, in the generated boards (all 18 have `height_m` 0.15, gaps 0.129-0.177 m) and in the
  render check: SD1000 (L) renders CHHCC and SD1007 (R) CCHHC, DES1/DES2 CCHHC/CHHCC.

## 3. Per-slot target assignment, and what is feasible

`slots.target_assignment` (config) gives every slot its target. Within each slot class the targets come from a seeded balanced
list (`assign_targets`: n slots over k models give each n // k and n % k models one more, those with the lowest running total
over the classes already done), so the target is no longer a function of the retries. A failed try retries the same (family,
side, target) and then the family's reserve seeds; the target never changes. The M16 sides are separate classes.

Feasibility matrix: real generator tries (`python -m sim.feasibility matrix`), attempt 0, 24 probe seeds (3000-3023) per cell;
`SIM/shelf_v2_geometry/matrix_24.json`. Cells with 0 successes were repeated with 100 tries (`matrix_100_tall.json`).

| Class | tomato soup can | potted meat can | mug | foam brick | tuna can |
|---|---|---|---|---|---|
| S8 | 0/100 | 41/100 | 40/100 | 20/24 | 23/24 |
| M16-L | 0/100 | 0/100 | 0/100 | 7/24 | 8/24 |
| M16-R | 0/100 | 0/100 | 0/100 | 7/24 | 8/24 |
| IND28 | 2/24 | 10/24 | 12/24 | 20/24 | 19/24 |
| white box | 14/24 | 11/24 | 10/24 | 14/24 | 18/24 |
| YCB box | 0/100 | 0/100 | 0/100 | 11/24 | 14/24 |

Every failure of a 0-success cell is at the occluder stage (IND28 fails at the occluder or at parking). Pairs with 0
placements are **left out** of the class's list with the reason in the config (`slots.target_assignment.excluded`); IND28 with
the tomato soup can (2/24) is kept.

**Dev 60 generated with shelf_v2** (`SIM/shelf_v2_geometry/dev60/`, `dev60_tables.txt`; generation only, 143 s on the Jetson):

| Class | tomato soup can | potted meat can | mug | foam brick | tuna can | Total |
|---|---|---|---|---|---|---|
| S8 | 0 | 3 | 3 | 3 | 3 | 12 |
| M16-L | 0 | 0 | 0 | 4 | 5 | 9 |
| M16-R | 0 | 0 | 0 | 5 | 4 | 9 |
| IND28 | 2 | 1 | 1 | 1 | 1 | 6 |
| white box | 3 | 2 | 3 | 2 | 2 | 12 |
| YCB box | 0 | 0 | 0 | 6 | 6 | 12 |
| **All** | 5 | 6 | 7 | 21 | 21 | 60 |

Against shelf_v1 (M16 only tuna can 11 and foam brick 7, YCB box only foam brick 4 and tuna can 8, S8 never the mug or the
tomato can), S8 now gets the mug and the meat can, white box and IND28 every model; M16 and the YCB box are still limited to the
two short models, because no valid placement was found for the others in the tested generator search (open question 1).

**Retries and replacements, dev 60** (`dev60/index/generation_log.json`): 77 failed tries, 0 replacements, no reserve seed used.

| Class | Slots | With retries | Failed tries |
|---|---|---|---|
| S8 | 12 | 2 | 7 (meat can 6, mug 1) |
| M16-L | 9 | 5 | 10 (tuna can 5, foam brick 5) |
| M16-R | 9 | 6 | 11 (tuna can 8, foam brick 3) |
| IND28 | 6 | 3 | 33 (tomato soup can 29, mug 4) |
| white box | 12 | 5 | 8 |
| YCB box | 12 | 6 | 8 |

**Test-100 slot list targets** (list only; no test arrangement exists; `SIM/shelf_v2_geometry/test_slot_list_targets.json`):

| Class | tomato soup can | potted meat can | mug | foam brick | tuna can | Total |
|---|---|---|---|---|---|---|
| S8 | 0 | 5 | 5 | 5 | 5 | 20 |
| M16-L | 0 | 0 | 0 | 8 | 7 | 15 |
| M16-R | 0 | 0 | 0 | 7 | 8 | 15 |
| IND28 | 2 | 2 | 2 | 2 | 2 | 10 |
| white box | 4 | 4 | 4 | 4 | 4 | 20 |
| YCB box | 0 | 0 | 0 | 10 | 10 | 20 |
| **All** | 6 | 11 | 11 | 36 | 36 | 100 |

## 4. Render check (GB10)

Twelve arrangements, not the full 60: the first slot of each class (SD1000 M16-L, SD1001 IND28, SD1002 S8, SD1005 YCB box,
SD1007 M16-R, SD1010 white box), two extra slots with taller targets (SD1014 white box with the tomato soup can, SD1022 S8 with
the mug) and DES0-DES3. **No M16 slot has a taller target** (no valid placement was found for any), so the first M16-L and M16-R slots were
rendered as they are (both foam brick). One Isaac process, exit code 0, 1565 s from start to exit, Isaac Sim 6.2.0.0,
`rt_subframes` 16, 100 updates on fresh render products per capture, boot 9.45 s
(`SIM/shelf_dev_v2_check/render_report_20261103T033622.json`; log `~/cmpe249-sim/run/shelf_dev_v2_check.log` on the GB10).
`python -m sim.episodes --contact-sheets` and `python -m sim.check_batch` ran without complaint
(`SIM/shelf_dev_v2_check/check/check.json`, `check_stdout.txt`, `episodes_stdout.txt`, contact sheets `<ARR>/contact_L1.jpg`,
`contact_L2.jpg`, `crops_L1.jpg`).

- **Render seconds:** per arrangement min 127.36, median 129.1, max 130.91, mean 129.04 (`render.json total_s`, 12 values); about
  130 s each from boot to exit. Same as the v1 check (124.5-130.6 s).
- **Tags and registration:** tags 0 to 3 found in all 480 captures (120 nominal, 360 query), no extra ids. All 360 query views
  registered onto their L1 reference with 4 tags: in-fit RMS min 0.149, median 0.255, max 0.583 px; leave-one-tag-out min
  0.958, median 2.716, max 7.112 px; target mask IoU after warping 0.884 / 0.957 / 0.990 (min / median / max), centroid
  offset 0.26 / 1.75 / 4.09 px. Rendered tag corners are at least 7.50 px from the frame edge in the query captures (median
  59.7) and 20.23 px in the references.
- **Revisit deltas** (60 stop draws): dx -9.07 to +7.92 mm, dy -9.90 to +9.94 mm, dz -9.99 to +9.22 mm; dyaw -0.485 to
  +0.496 deg, dpitch -0.500 to +0.496 deg, droll -0.458 to +0.496 deg, all inside +-10 mm / +-0.5 deg. 8 of the 60 stops needed a
  redraw (18 rejected draws).
- **Visible-phase queries differ from their references** (120 pairs): none pixel-identical, mean absolute difference 1.16 /
  4.61 / 10.9 grey (min / median / max).
- **Label sanity:** no arrangement flagged; the lowest visible-phase / target-alone pixel ratio is 1.0 in all 12.
- **M16 patterns and sidedness** (rendered occluded views, H <= 0.05, C >= 0.90):

| Arrangement | Side | Target | Visible fraction P1 / P2 / P3 / P4 / P5 | Pattern | Rendered side |
|---|---|---|---|---|---|
| SD1000 | L | foam brick | 1.00 / 0.00 / 0.00 / 0.913 / 1.00 | CHHCC | L |
| SD1007 | R | foam brick | 1.00 / 0.945 / 0.00 / 0.00 / 1.00 | CCHHC | R |
| DES1 | R | study mug | 1.00 / 0.997 / 0.00 / 0.00 / 1.00 | CCHHC | R |
| DES2 | L | study mug | 1.00 / 0.003 / 0.00 / 1.00 / 1.00 | CHHCC | L |

  Both generated boards are the intended pattern, L and R are symmetric, and the adjacent stop is H on both (v1 gave CpHCC for
  SD1000). The near opposite stops are close to the C cut-off (SD1000 P4 0.913, SD1007 P2 0.945), as the rule's 0.10 allows.
  **There is no rendered M16 pattern for a tall target, because no valid placement was found for one.**
- **Other classes** (occluded pattern): IND28 HHHHH (SD1001 potted meat can, DES3); S8 CpppC (SD1002 meat can), CpHpC (SD1022
  mug); YCB box CpHCC (SD1005); white box CpHpC (SD1010 tuna can, SD1014 tomato soup can). Only M16 is one-sided by design.
- **Coverage match, rendered against analytic:** 1 minus the rendered visible fraction against the generator's analytic hull
  coverage differs by at most 0.019 at any stop of the 8 random arrangements (SD1022 P3: rendered 0.981 hidden, analytic 0.962);
  SD1000 0.011, SD1007 0.001; the design cases at most 0.003.
- **Diagnostic views** (visible / occluded / removed): M16 5 / 3 / 3; IND28 5 / 0 / 0; S8, YCB box, white box 5 / 2-3 / 2-3.
  SD1000 P4 removed: the occluder covers 0.087 of the region (diagnostic, next to the 0.10 cut-off).

## 5. Review fixes (A.1-A.11) and two later findings

Tests: 573 run, all pass, 10 skipped (the same 10 as before; baseline at the start was 533 run, 10 skipped).

| # | Fix | Test (checked by mutation) |
|---|---|---|
| 1 | `--skip-existing` also compares isaac_batch / isaac_rig sha256, the lighting set, the render settings hash (resolution, renderer, rt subframes, settle and capture updates, lighting presets, materials, Isaac version, models_info, tag images) and the rgb files | L1-only folder resumed with L1+L2; a label-code change; rig change; settings change. Removing the lighting, code or settings comparison fails 2-3 tests each |
| 2 | `episodes.build` refuses a set from more than one generator or config sha256 (also design cases), before writing anything | `TestEpisodeSetIntegrity` |
| 3 | `n_episodes` counts `labels['lightings']` | L1-only set |
| 4 | `check_batch` skips and reports `.partial` / `.old*`, refuses a folder whose id is not its name | `TestCheckBatchFolders` |
| 5 | `contact_sheets` handles None fractions, obstructions and boxes (and `summary_md` a set without reference fractions) | synthetic folder |
| 6 | stale comment: the removed region is the visible-phase mask at the revisit pose | |
| 7 | label sanity guard in `labels.json`, the renderer output, `check_batch` (exit 1) and `set.json` | pure and render-flow tests |
| 8 | arrange CLI refuses a non-empty `--out` without `--overwrite`; with it only its own arrangement and index files are removed | `TestOutputFolderHygiene` |
| 9 | reversed slot ranges raise | `TestSlotRanges` |
| 10 | hash test regenerates under another reported Python version and cwd and checks that neither enters the hashed record | adding `platform.python_version()` or `Path.cwd()` to the provenance fails it (both mutations) |
| 11 | slot tests for all six classes with fake placements | M16 side flipped after 3 attempts, one shared reserve cursor, and a white box replacement served from the S8 reserve each fail it |
| 12 | README "Known gaps" and the manifest line corrected (verify reads the sim view labels and validates the manifest keys) | |
| 13 | `positions_far_{A,B15,B20}_draft.json` vendored in `tests/fixtures/positions/` (byte-identical; README there), arrange tests point to them | |
| 14 | region denominator test: reference and visible masks of different area (100 and 64 px), obstruction share 16/64 | the reviewer's mutation (`region_px` from the reference mask) and the obstruction-from-reference mutation both fail it |

`tests/test_sim_rig.py` (not mine) still reads the drafts from `data/raw` and skips when they are absent.

## 6. Not checked (shelf_v2 render check)

- No M16 board with a tall target was rendered (no valid placement was found under the rules).
- The feasibility scan and matrices were run with the shelf_v2 config before its text-only note edits (identical numbers; the
  config sha256 inside them differs from the final one); the scan has 60 placements per target and a grid, so absence is bounded
  by that sampling.
- Full-batch throughput with inference; the tabletop environment; evidence scores on these views.

---

# Follow-up: shelf_v3 (student decisions 1a, 2a, 3b; tag-free scenes)

Evidence is in `SIM/shelf_v3_geometry/` (geometry, dev 60, test list) and `SIM/shelf_dev_v3_check/` (render check). Code: branch
`cp2-fix-sim`; `sim/arrange.py` sha256 `1dc84d9a...`, config `shelf_v3` canonical-JSON sha256 `2f527422...`. `configs/sim/arrange_shelf_v1.json`
and `arrange_shelf_v2.json` are untouched (v2 is the record of `shelf_dev_v2_check`; both keep their tags). v3 is the CLI default.

## 7. What v3 changes against v2

1. **M16 is the physical board again** (`height_m` 0.12 = pick_M16+3 / pick_M16-5 in `configs/rig_geometry.json`), so it matches the
   physical 10-10 board. The v3-rule scan (`height_scan.out`) agrees with v2: tuna can feasible from 0.10 m, foam brick from 0.10 m (87 /
   161 combinations at 0.11 m, 204 / 250 at 0.12 m for L / R); the tomato soup can has none up to 0.25 m; the meat can has 1-2
   combinations on side L from 0.13 m and the mug 2 from 0.14 m (boards taller than the physical one), none on side R.
2. **No `same_far_max`.** `adjacent_min` 0.95, `opposite_near_max` 0.10 and at least 2 clear stops are kept; partial cover of the
   far same-side stop is accepted, so pHHCC and CCHHp boards are still one-sided. The design cases keep their templates.
3. **No corner tags or tag cards** (`scene.tags` false; `scene.tags_enabled` false in the scene spec; `isaac_batch` builds no tag quads;
   the render-settings hash records `tags_enabled`). Reason: no fiducials on a home robot's shelf. Evidence v0.3 aligns query and
   reference itself; frozen v0.2.1 falls back to no alignment without tags. No tag detection in 80 v3 renders (SD1000, DES1: 0 hits).
4. **Frame check without tags:** the #3 outline and the target's projected hull inside the frame, margin >= 0 px with k1, for the nominal
   and revisit poses (`frame_check`). The per-stop pitch values in `positions_shelf_A.json` are unchanged; the file itself is not edited,
   because every v1 / v2 arrangement records its sha256 and `episodes` checks it (the rationale is in the v3 config's `scene.frame_check`
   and the README). **Redraw rate** on the 60 dev slots + DES0-3 (320 stops, same draws, `redraw_rate.txt`): 89 rejected draws at 52
   stops under the tag rule (P1 55, P5 34); 0 under the tag-free rule. In the render check: 0 redraws at 60 stops.
5. **True alignment** (`sim/alignment.py`, `true_alignment.json`): format in the README. Unit-tested on synthetic cameras (plane points
   projected through both cameras map through H to 1e-6 px). Validated against the tag-based homography on the v2 check set (which has
   tags): front-plane H vs `capture/tags.py align` H, 25 views over the central 60 % of the frame, mean differences 0.08-0.21 px, largest
   0.58 px (`valid_align.txt`).

## 8. Targets under the v3 rules

Matrix (`matrix_24.json`, 24 probe seeds per cell; `matrix_100_tall.json`, 100 tries for the zero cells): identical to v2. Admitted by the
rule (>= 2 of 24 tries): S8 meat can, mug, foam brick, tuna can (tomato soup can: no valid placement found in the tested generator search,
0 of 100 tries); M16-L and M16-R foam brick, tuna can (tomato, meat can, mug: 0 of 100 tries each side); IND28 and white box all five
(IND28 tomato soup can 2 of 24); YCB box foam brick, tuna can (tomato, meat can, mug: 0 of 100). Removing `same_far_max` opened nothing.
Zero in N tries is not a proof (see item 1).

**Dev 60 (generation only, `dev60/`, `dev60_tables.txt`)**, 186 s on the Jetson:

| Class | tomato soup can | potted meat can | mug | foam brick | tuna can | Total |
|---|---|---|---|---|---|---|
| S8 | 0 | 3 | 3 | 3 | 3 | 12 |
| M16-L | 0 | 0 | 0 | 4 | 5 | 9 |
| M16-R | 0 | 0 | 0 | 5 | 4 | 9 |
| IND28 | 2 | 1 | 1 | 1 | 1 | 6 |
| white box | 3 | 2 | 3 | 2 | 2 | 12 |
| YCB box | 0 | 0 | 0 | 6 | 6 | 12 |
| **All** | 5 | 6 | 7 | 21 | 21 | 60 |

Retries and replacements (`dev60/index/generation_log.json`): 76 failed tries, 0 replacements, no reserve seed used. Failed tries / slots with
retries: S8 7 / 2, M16-L 10 / 5, M16-R 10 / 5, IND28 33 / 3, white box 8 / 5, YCB box 8 / 6.

**Analytic patterns of the 18 M16 boards** (`m16_patterns.txt`; H = coverage >= 0.95, C <= 0.10, p between): 15 pure (8 CHHCC, 7 CCHHC), 3 with a
partial far same-side stop: SD1000 pHHCC (P1 0.234), SD1033 CCHHp (P5 0.249) and SD1046 CCHHp (P5 0.14); SD1016 (0.081) and SD1017 (0.033)
stay C. Adjacent same-side stop >= 0.956 (L) / 0.997 (R); near opposite stop <= 0.074 (L) / 0.054 (R).
**Clear stops:** no arrangement (all 60 slots and DES0-3) other than IND28 has fewer than 2 stops with coverage <= 0.001; the minimum is 2 (SD1000).

**Test-100 slot list** (list only, `test_slot_list_targets.json`): S8 5 / 5 / 5 / 5 (meat can, mug, foam brick, tuna can); M16-L foam brick 8, tuna can 7;
M16-R foam brick 7, tuna can 8; IND28 2 of each of the five; white box 4 of each; YCB box foam brick 10, tuna can 10. Totals: tomato soup can 6,
meat can 11, mug 11, foam brick 36, tuna can 36.

## 9. Render check, tag-free (GB10)

12 arrangements, not the full 60: M16-L SD1000 and SD1004, M16-R SD1007 and SD1009, the first IND28 (SD1001), S8 (SD1002), YCB box (SD1005) and white
box (SD1010) slots, and DES0-DES3. No class's target set changed against v2, so no further slot was added. One Isaac process, exit code 0,
1706 s from start to exit, Isaac Sim 6.2.0.0, boot 9.37 s (`SIM/shelf_dev_v3_check/render_report_20261103T091505.json`). `sim.episodes --contact-sheets`
and `sim.check_batch` ran (`check/check.json`, `check_stdout.txt`).

- **No tags:** `check_batch` says "TAGS: SKIPPED: the scenes have no corner ArUco tags", and tag detection and registration are not computed;
  `render.json` settings record `tags_enabled` false and no tag images.
- **Label sanity:** no arrangement flagged (lowest ratio 1.0 in all 12).
- **Render seconds:** min 123.15, median 137.9, max 174.36, mean 140.8 (the first five took 123-129 s, as before; the later ones 132-174 s; the cause was not investigated, the GB10 showed no other Isaac process of mine).
- **Revisit:** 0 redraws at the 60 stops. Deltas within -9.0 to +8.7 mm (dx), -9.9 to +9.9 mm (dy), -10.0 to +9.4 mm (dz), -0.49 to +0.50 deg (yaw), -0.50 to +0.50 deg (pitch), -0.50 to +0.50 deg (roll).
- **Patterns and sidedness** (rendered occluded views):

| Arrangement | Class / side | Target | Visible fraction P1 / P2 / P3 / P4 / P5 | Pattern | Rendered side |
|---|---|---|---|---|---|
| SD1000 | M16-L | foam brick | 0.75 / 0.00 / 0.034 / 1.00 / 1.00 | pHHCC | L |
| SD1004 | M16-L | tuna can | 1.00 / 0.00 / 0.002 / 0.969 / 1.00 | CHHCC | L |
| SD1007 | M16-R | foam brick | 1.00 / 0.945 / 0.00 / 0.00 / 1.00 | CCHHC | R |
| SD1009 | M16-R | foam brick | 1.00 / 0.956 / 0.00 / 0.00 / 1.00 | CCHHC | R |
| DES1 / DES2 | M16 R / L | study mug | as in v1 / v2 | CCHHC / CHHCC | R / L |

  Other classes: IND28 HHHHH (SD1001, DES3), S8 CpppC (SD1002, meat can), YCB box CpHCC (SD1005), white box CpHpC (SD1010), DES0 CCHCC.
  SD1000 is the expected pHHCC (analytic P1 0.234, rendered 0.25 hidden) and still one-sided.
- **Coverage match** (1 - rendered visible fraction against the analytic coverage): at most 0.018 at any stop of the 8 random arrangements (SD1010 / SD1002 / SD1000 /
  SD1004 / SD1005 within 0.011-0.018), at most 0.003 for the design cases.
- **Visible-phase queries differ from their references:** none of 120 pairs identical, mean absolute difference 0.92 / 3.67 / 9.49 grey (min / median / max).
- **True alignment** (`check/check.json`, `true_alignment.json`; 60 views = 12 arrangements x 5 stops): the largest displacement of the reference box's corners
  between reference and query under the true homography (what no alignment leaves): front plane min 2.36, median 14.96, max 33.89 px (mean 16.5); target plane
  min 3.62, median 14.04, max 31.54 px (mean 15.3).

## 10. Tests

`python -m unittest discover -s tests -t .`: 588 run, OK, 10 skipped (573 before this follow-up). New: v3 config (diff against v2, physical 12 cm board, no far-stop limit,
wording of exclusions, balanced slot targets, tag-free scene and frame check, generated M16 boards one-sided with 2 clear stops), the plane homography on
synthetic cameras, `true_alignment.json` through `episodes`, and `check_batch` on a tag-free set.
