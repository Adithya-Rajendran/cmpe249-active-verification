# The remaining 49 tabletop development arrangements: fixed-threshold transfer diagnostic (2026-11-06)

**Simulation, development data.** This is the reporting that Amendment 2 of `notes/2026-11-04-registration-repair-plan.md`
(5b8a38b) requires for the 49. It keeps the first 11 arrangements and the 49 apart. The thresholds are the fixed shelf
v03_image candidate (sha256 4a06de0e...: tau_q 0.3, tau_occ 0.35, tau_match 0.65, k 2, lam 0), as they were for the
first 11. This is a transfer diagnostic, not the final calibration, which is in `notes/2026-11-06-pooled-calibration.md`.
Simulator truth is used for scoring only. After this diagnostic the 49 were used for calibration and training, so from
here on they are **development data, not an independent test set**.

## What ran

- **Order.** The 49 were rendered after Amendment 2. Until A's design was pinned (f4d4fe5) and implemented (5ff3b89,
  088e1dc), only operational outputs were read: completion, exit codes, counts, timings and alerts.
- **Render.** 53 / 53 arrangements (the 49 plus the design cases TDES0-TDES3) on the GB10 at commit 1e65727. Render
  time per arrangement: median 176.0 s, range 168.7-205.5 s. The watcher raised no alert, and episodes and the frame
  check exited 0.
  - Evidence: share `sim/tabletop_dev49/check/check.json`; driver status `runs/tabletop_dev49/status.json` (this
    worktree, untracked).
- **Evidence.** C0 unchanged (`python -m perception.evidence_v03 --align image`), from
  `configs/pipeline/tabletop_dev49_evidence.json` at ee2c4fa. Output: share
  `sim/tabletop_dev49/evidence/evidence_tabletop_dev49_evidence_ee2c4fa50b38_3e22530d10/`.
  - Evidence of record `v03_image_refL1.jsonl`: sha256 8b32ad4f... (random arrangements) and 803c8ac5... (design
    cases).
  - The harness C0 (`perception/registration_v032.py --candidate C0`) reproduces it exactly: largest difference 0,
    equal refusals.
  - The left/right descriptor (`perception/side_descriptor.py`) ran in the same job.
- **Views.** `tools/registration_compare.py`, C0 against C0, for refusals, per-view errors and plane geometry. Outputs:
  `runs/2026-11-05-tabletop-dev49/transfer/regcompare/` and `regcompare_design/` (this worktree, untracked).
- **Decisions.** `verify.run` (real-v5) with the shelf candidate, b = 0-4, every cap. Outputs:
  `runs/2026-11-05-tabletop-dev49/transfer/run49/`; the first 11 rerun the same way in `run11/`; a compact table in
  `summary/decisions_b3.txt`.

**About "PASS" in the comparison report.** `registration_compare` was built to compare a candidate with C0. Run as C0
against C0, it marks criteria 2-6 PASS because each compares C0 with itself. That label is not a registration
result.
- Criterion 1 (refusals < 5 %) depends on refusals alone. On the 49 they are 16 / 735 (2.2 %) in L1 and 15 / 735
  (2.0 %) in L2.
- Amendment 2 keeps C0's failure of that criterion on the first 11 (12 / 165 and 9 / 165). Nothing here makes C0 a
  passing registration.

## Views, by lighting

"Present gate pass" counts visible-phase views that pass the tau_q / tau_occ gate. Geometry is the disagreement with a
specified simulator plane on the views C0 accepts. On the tabletop that is the horizontal plane through the target's
centre, a proxy, with the table plane z = 0 beside it. A plane disagreement is not an error against a true whole-scene
alignment.

| C0 | First 11, L1 | First 11, L2 | 49, L1 | 49, L2 |
|---|---|---|---|---|
| refused | 12 / 165 | 9 / 165 | 16 / 735 | 15 / 735 |
| present-labelled clear misses | 0 / 110 | 0 / 110 | 0 / 490 | 0 / 490 |
| removed-labelled strong matches | 0 / 55 | 0 / 55 | 1 / 245 | 1 / 245 |
| diagnostic removal yield | 24 / 27 | 25 / 27 | 123 / 133 | 109 / 133 |
| present gate pass | 53 / 55 | 48 / 55 | 243 / 245 | 212 / 245 |
| target-centre plane, median / p90 / max px | 2.22 / 8.01 / 17.12 | 2.29 / 8.25 / 19.72 | 2.16 / 5.50 / 22.43 | 2.14 / 5.70 / 60.12 |
| table plane z = 0, median / p90 / max px | 1.66 / 8.78 / 18.39 | 1.65 / 8.78 / 20.99 | 2.00 / 5.88 / 23.30 | 2.04 / 5.52 / 60.51 |
| target silhouette IoU, median / p90 / max | 0.98 / 0.99 / 1.00 | 0.98 / 0.99 / 1.00 | 0.98 / 0.99 / 1.00 | 0.98 / 0.99 / 1.00 |

The first 11's column is the C0 arm of the repair experiment (`runs/regrepair_c2_tabletop/score/C2_refL1.md`).

- **Refusals on the 49** fall in 4 arrangements, all in occluded or removed phases. None is in a visible phase.
  - L1: TD5055 8, TD5022 4, TD5031 2, TD5033 2.
  - L2: TD5022 6, TD5055 5, TD5031 2, TD5033 2.
  - By family: L1 has ycb_box 10, IND28 4, white_box 2; L2 has ycb_box 7, IND28 6, white_box 2.
- **The removed-labelled strong match** is TD5017-removed, stop P1, in both lightings. TD5017 is M16, right side,
  target tomato soup can. Registration at that view is close to the plane (2.5 px, IoU 0.99), so the strong match does
  not come from a failed registration. Its cause was not examined further.
- **The largest plane disagreement** is TD5017-removed-L2, stop P2: 60.1 px, IoU 0.637. It is an accepted
  registration far from the plane. That view is a clear miss, and it is labelled removed.
- **Design cases TDES0-TDES3** are reported only, 60 views per lighting:
  - 0 refused; 0 / 40 present-labelled clear misses; 0 / 20 removed-labelled strong matches;
  - removal yield 10 / 10;
  - plane disagreement at most 3.47 px.
  - **In L2, 0 / 20 visible-phase views pass the gate, against 20 / 20 in L1.** At these thresholds no L2 design-case
    view could give a clear miss. This is a lighting effect on the gate and is recorded here as an observation.

## Decisions at b = 3, by lighting

The table gives correct removals among removed episodes: 49 per lighting for the 49, 11 for the first 11. R is the
expected count over its 20 seeds. FULL uses all five stops, uncapped.

| Travel | Policy | 49, L1 | 49, L2 | First 11, L1 | First 11, L2 |
|---|---|---|---|---|---|
| uncapped | F | 35 | 31 | 7 | 6 |
| uncapped | R | 38.95 | 34.30 | 7.40 | 7.90 |
| uncapped | **G** | **43** | **36** | **8** | **9** |
| uncapped | Dir | 43 | 36 | 8 | 8 |
| uncapped | FULL | 43 | 36 | 8 | 9 |
| 2 steps | G | 0 | 0 | 0 | 0 |
| 2 steps | G-lookahead | 22 | 24 | 5 | 5 |
| 2 steps | R | 10.25 | 11.70 | 2.70 | 2.85 |
| 2 steps | Dir | 33 | 32 | 8 | 7 |
| 3 steps | G (= F, = G-lookahead) | 22 | 24 | 5 | 5 |
| 3 steps | Dir | 33 | 32 | 8 | 7 |
| 4 steps | G (= F, = G-lookahead) | 22 | 24 | 5 | 5 |
| 4 steps | Dir | 33 | 32 | 8 | 7 |

- **False removals:** 0 in every row, for every policy, travel setting and lighting, on both subsets.
- **False presents:** on the 49, one per lighting for G, F, Dir, G-lookahead and FULL. It is TD5017-removed, from the
  strong match above. This holds uncapped and under every cap. R has 0.25-0.60 in expectation, depending on the
  travel setting. The first 11 have none.
- **Unknowns, G uncapped:**
  - the 49: removed episodes 5 (L1) and 12 (L2); occluded-present episodes 4 and 4, the 4 occluded episodes with no
    geometrically diagnostic stop;
  - the first 11: removed 3 and 2; occluded 2 and 2.
- **Effort, uncapped, on the 49:** G 1.11 / 1.16 additional views and 67 / 68 cm mean travel; Dir 1.16 / 1.16 views and
  37 / 37 cm.
- **G against the ceiling.** Uncapped, G equals FULL on the 49 in both lightings, as it did on the first 11. G's
  unknowns are episodes that no ordering of the five stops resolves at these thresholds.

Dir is the post-hoc development control defined in `notes/2026-11-05-side-descriptor.md`. These numbers come from
fixed shelf thresholds on tabletop evidence. They are not the study's thresholds and do not predict the final
comparison.
