# Test results: `simulation-freeze-1` (2026-11-06)

**SIMULATION.** The frozen study's test was run by the FREEZE.md §2 runbook. It started from the addendum commit 1ad530b, a child of the freeze commit 6d0ed94; `docs/FREEZE_ADDENDA.md` holds Addenda 1 and 2.
- Every number is a simulation result.
- Unless stated otherwise, numbers come from `verify.run` real-v7 on the test split: `runs/2026-11-06-test/<env>/results.json` and `table.md`, untracked, in the worktree that ran it.
- The compact extracts are in `runs/2026-11-06-test/summary/`.
  - `extract.py` and `extract2.py` copy values from results.json without recomputing them.
  - `agreement.py` is a post-hoc script outside the freeze record. It counts per-episode decision, path and effort agreement at b = 3 from logs.jsonl.
  - Its G vs G-lookahead counts match verify.run's own diagnostics exactly: decisions differ in 54 / 112 episodes at 2 steps and 8 / 7 at 6 steps, and paths in 52 / 58 at 6 steps.
- No development number appears below unless it is labelled as development. The first 11 tabletop arrangements, the remaining 49 and the test are kept apart.

**How to read the intervals** (FREEZE.md §5).
- Every interval is the 95 % percentile interval of an arrangement-level cluster bootstrap, B = 10 000, paired for contrasts.
  - The main intervals are stratified by quota slot class.
  - The family-level intervals are stratified within the family, by side for pooled M16, or unstratified for a single slot class. Each has its own derived seed.
- They are pointwise and unadjusted: each covers one metric of one policy or pair, in one environment, lighting and setting.
- No claim is made for "some cell" or "every cell". An interval that contains 0 is inconclusive, never equivalence.
- ZW marks a zero-width interval. It is not evidence of zero risk, of a perfect rate or of equivalence.
- Each environment × lighting has 100 arrangements and 300 episodes: 100 visible-present, 100 occluded-present and 100 removed.

## 1. The run and its validity

**Steps.**
- **Generation** (`sim.testsplit`, 10:36–10:43 PST). Both environments exited 0 with 100 test arrangements, each recording FREEZE.md sha256 bb7baa02… and the freeze id.
- **Render** (GB10 `run_test`):
  - shelf 10:45–14:19 PST, median 128.3 s, max 130.1 s;
  - tabletop 15:02–19:56 PST, median 176.0 s, max 181.6 s.
- **Episodes, the set check (with table clipping on the tabletop) and evidence:** every step rc 0. Shelf evidence finished at 15:02 and tabletop evidence at 21:14 PST.
- **Scoring** (`verify.run`, every declared flag) took 626 s and 619 s of wall time, as timed by the runbook shell (`runs/2026-11-06-test/verify_run_timing.txt`).

**Validity, checked before any outcome was read** (`tools/test_validity.py`; `runs/2026-11-06-test/<env>/validity*.json`).
- **Operational outputs: ALL PASS, 23 / 23 per environment.**
  - generation, 9 checks;
  - render, 5 checks: exactly the generated set, the frozen isaac_batch.py and isaac_rig.py, the development render settings, and the render log naming the frozen code and this FREEZE.md;
  - episodes, 2 checks;
  - evidence, 6 checks: the frozen uv.lock and packages on the GB10, the frozen evidence code at the shipped commit dccf8d2, the job on this set, the record counts, and the harness C0 reproducing the evidence of record (largest difference 0, equal refusals);
  - the GB10 checkpoint hashes.
- **Then with `--verify-run`: ALL PASS, 29 / 29 per environment.** The run used the frozen candidate, positions, models, thresholds and real-v7 analysis code, with every declared secondary run on. It ran on this set's evidence and descriptors.
- **Order.** Both environments passed before either environment's outcomes were opened.

**Counts that §11 requires** (per environment).

| | Shelf | Tabletop |
|---|---|---|
| Test arrangements | 100 (slots 0–99, primary seeds 100000–100099) | 100 (110000–110099) |
| Reserve replacements | 0 | 0 |
| Placement retries (failed tries), all within the maximum of 20 | 123: IND28 22, M16-L 29, M16-R 25, S8 16, white_box 16, ycb_box 15 | 69: IND28 24, M16-L 15, M16-R 24, S8 0, white_box 2, ycb_box 4 |
| Revisit (frame-check) redraws | 0 of 500 stops | 0 of 500 stops |
| Re-renders and exclusions | none | none |
| Label-sanity flags | none | none (table clipping passed) |

The full slot lists are in `index/slots.json` beside the arrangements, on the share at `sim/<env>_test_geometry/arr`.

**Disclosure.** Before step 4, one status printout of the shelf pipeline's check stage showed the last check lines for ST100096–ST100098: their geometric occlusion patterns and diagnostic-view counts. These are frozen-code labels, not evidence or decisions, and no frozen choice depended on them.

## 2. Primary: A − G at b = 3, uncapped

| | Removal recall | Coverage | False removal rate | Discordant removed episodes | Reading |
|---|---|---|---|---|---|
| Shelf L1 | +0.030 [0.000, 0.070] | +0.010 [0.000, 0.023] | 0.000 ZW | 3, in 3 arrangements | inconclusive |
| Shelf L2 | +0.040 [0.010, 0.080] | +0.013 [0.003, 0.027] | 0.000 ZW | 4, in 4 arrangements | the intervals exclude 0: A higher (pointwise) |
| Tabletop L1 | −0.010 [−0.040, 0.020] | −0.003 [−0.013, 0.007] | 0.000 ZW | 3, in 3 arrangements | inconclusive |
| Tabletop L2 | +0.040 [0.000, 0.090] | +0.013 [0.000, 0.030] | 0.000 ZW | 6, in 6 arrangements | inconclusive |

**In counts.** Correct removals out of 100 removed episodes, A vs G:

| | A | G | A − G | Discordant episodes favouring A / G |
|---|---|---|---|---|
| Shelf L1 | 85 | 82 | +3 | 3 / 0 |
| Shelf L2 | 85 | 81 | +4 | 4 / 0 |
| Tabletop L1 | 85 | 86 | −1 | 1 / 2 |
| Tabletop L2 | 81 | 77 | +4 | 5 / 1 |

There were no wrong decisions, so every discordant episode is "removed" by one policy and "unknown" by the other.

**Both error directions.**
- **False removals:** neither policy made any, out of 200 present episodes, in any cell. The false-removal contrast has no discordant episode.
- **False presents:** none, out of 100 removed episodes.

**Unknowns.**
- Each policy left 10 of 200 present episodes UNKNOWN in every cell.
- The geometric labels give 10 occluded-present episodes per cell with no diagnostic stop besides the start, all IND28.
- On removed episodes, a policy's unknowns are 100 minus its correct removals.

**Effort.** Mean views are equal: 2.14 / 2.14 / 2.14 / 2.17 for both policies. A travels more:

| | A | G |
|---|---|---|
| Shelf L1 | 154.9 cm | 144.6 cm |
| Shelf L2 | 155.9 cm | 144.9 cm |
| Tabletop L1 | 72.7 cm | 68.5 cm |
| Tabletop L2 | 74.1 cm | 69.1 cm |

**The preregistered expectation.** SIM_PREREGISTRATION §14 conditionally expected an inconclusive primary, with an SD of 0.012–0.017 for A − G removal recall.
- That estimate rested on 5 discordant development episodes.
- The test had more discordant episodes (3, 4, 3, 6), and its recall intervals are 0.06–0.09 wide.
- Three cells are inconclusive. In the fourth (shelf L2), the recall and coverage intervals exclude 0.
- This is one cell of four, read pointwise and without adjustment. It is not a claim that A wins somewhere or everywhere.

## 3. All policies at the primary setting

At b = 3, uncapped, no multi-view k = 2 policy, and not FULL, made a false removal (0 / 200) or a false present (0 / 100) in any cell.
- Per arrangement, that is a false-removal rate of 0 / 100 in every cell.
- Every multi-view k = 2 policy left 10 / 200 present episodes UNKNOWN. S, with one view, left 100 / 200.
- So the multi-view policies differ only in how many removed episodes they resolve.

Correct removals out of 100, with mean views and mean travel in cm:

| Policy | Shelf L1 | Shelf L2 | Tabletop L1 | Tabletop L2 |
|---|---|---|---|---|
| S | 0 (1.00, 0) | 0 | 0 | 0 |
| F | 49 (2.31, 111.9) | 50 (2.31, 112.5) | 82 (2.21, 48.4) | 76 (2.22, 48.7) |
| R (20 seeds, averaged) | 64.75 (2.32, 125.4) | 65.9 (2.33, 125.8) | 84.2 (2.23, 54.1) | 77.25 (2.24, 54.8) |
| G | 82 (2.14, 144.6) | 81 (2.14, 144.9) | 86 (2.14, 68.5) | 77 (2.17, 69.1) |
| G-lookahead | 82 (= G) | 81 (= G) | 86 (= G) | 77 (= G) |
| G-myopic | 83 (2.31, 125.5) | 83 (2.31, 126.5) | 86 (2.14, 68.5) | 77 (2.17, 69.1) |
| Dir | 85 (2.25, 94.1) | 85 (2.27, 96.9) | 85 (2.16, 36.4) | 81 (2.17, 38.0) |
| A | 85 (2.14, 154.9) | 85 (2.14, 155.9) | 85 (2.14, 72.7) | 81 (2.17, 74.1) |
| A-nocue | 82 (= G) | 81 (= G) | 86 (= G) | 77 (= G) |
| FULL (5 views, not budget-matched) | 86 (5.00, 298.8) | 86 | 89 (5.00, 141.4) | 82 |

S resolves only visible-present episodes, so its coverage is 0.333. The -fcm (k = 1) variants are in §4.7.

## 4. The declared secondary comparisons

### 4.1 A − G-lookahead across the travel caps (b = 3)

Removal recall, L1 / L2. One step is 49.8 cm on the shelf and 23.6 cm on the tabletop. False removals are 0 ZW at every cap.

| Cap | Shelf | Tabletop |
|---|---|---|
| uncapped | +0.030 [0.000, 0.070] / +0.040 [0.010, 0.080] | −0.010 [−0.040, 0.020] / +0.040 [0.000, 0.090] |
| 1 step | 0 ZW / 0 ZW | 0 ZW / 0 ZW |
| 2 steps | −0.280 [−0.340, −0.220] / −0.260 [−0.320, −0.210] | −0.570 [−0.640, −0.500] / −0.550 [−0.620, −0.480] |
| 3 steps | +0.180 [0.140, 0.220] / +0.160 [0.130, 0.190] | +0.160 [0.100, 0.220] / +0.140 [0.090, 0.190] |
| 4 steps | +0.180 [0.140, 0.220] / +0.160 [0.130, 0.190] | +0.160 [0.100, 0.220] / +0.140 [0.090, 0.190] |
| 6 steps | −0.030 [−0.070, 0.000] / −0.050 [−0.090, −0.010] | −0.030 [−0.070, 0.000] / −0.040 [−0.080, −0.010] |

At the 1-step cap, coverage differs although recall does not: shelf +0.057 [0.043, 0.070] / +0.053 [0.050, 0.060], tabletop +0.050 ZW / +0.047 [0.040, 0.050]. The tabletop ZW value has 15 discordant episodes in 15 arrangements.

At the 6-step cap, A's decisions equal G's in every episode in all four cells (verify.run `same_decisions`).

The levels behind these contrasts, correct removals out of 100, L1 | L2:

| Cap | Shelf G | Shelf G-lookahead | Shelf Dir | Shelf A | Tabletop G | Tabletop G-lookahead | Tabletop Dir | Tabletop A |
|---|---|---|---|---|---|---|---|---|
| 1 step | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 |
| 2 steps | 0 \| 0 | 28 \| 26 | 46 \| 42 | 0 \| 0 | 0 \| 0 | 57 \| 55 | 73 \| 69 | 0 \| 0 |
| 3 steps | 28 \| 26 | 28 \| 26 | 46 \| 42 | 46 \| 42 | 57 \| 55 | 57 \| 55 | 73 \| 69 | 73 \| 69 |
| 4 steps | 28 \| 26 | 28 \| 26 | 46 \| 42 | 46 \| 42 | 57 \| 55 | 57 \| 55 | 73 \| 69 | 73 \| 69 |
| 6 steps | 79 \| 77 | 82 \| 82 | 85 \| 85 | 79 \| 77 | 78 \| 70 | 81 \| 74 | 84 \| 81 | 78 \| 70 |

### 4.2 A − Dir (explanatory control, defined post hoc in development)

**What Dir is.** Dir has no fitted parameter. It goes to the side indicated by the start view's left/right descriptor, and falls back to G when the descriptor is missing or 0.

**Agreement with Dir** (`summary/agreement.json`, post hoc, from logs.jsonl; and verify.run's `same_decisions` flags). A's decisions equal Dir's in every episode at b = 3:
- uncapped, with every interval 0 ZW;
- under the 1-, 3- and 4-step caps.

**Paths and effort.** Uncapped and under the 3- and 4-step caps they differ in 400 of 600 shelf episodes and 383 of 600 tabletop episodes. Under the 1-step cap the two runs are identical, paths and travel included.

**This is specific to b = 3.**
- At b = 2, uncapped, A's removal counts equal G's, not Dir's: shelf 79 / 77 against Dir's 46 / 42, and tabletop 78 / 70 against 73 / 70.
- At b = 1 on the shelf, Dir resolves 189 / 188 present episodes against A's 190.

**Effort.**
- Uncapped: A travels 154.9 / 155.9 cm (shelf L1 / L2) against Dir's 94.1 / 96.9, and 72.7 / 74.1 cm (tabletop) against 36.4 / 38.0. Views are 2.14 vs 2.25 / 2.27 (shelf) and 2.14 / 2.17 vs 2.16 / 2.17 (tabletop).
- Under the 3- and 4-step caps: 84.7 vs 51.6 / 51.8 cm (shelf) and 40.1 vs 24.8 / 25.2 cm (tabletop), with 2.03 vs 2.04 views (shelf) and 2.03 vs 2.03 (tabletop).

**Where they differ** (recall, L1 / L2):

| Cap | Shelf | Tabletop | Decisions differ in |
|---|---|---|---|
| 2 steps | −0.460 [−0.530, −0.390] / −0.420 [−0.490, −0.360] | −0.730 [−0.800, −0.660] / −0.690 [−0.760, −0.620] | 88 / 142 episodes |
| 6 steps | −0.060 [−0.110, −0.020] / −0.080 [−0.140, −0.030] | −0.060 [−0.110, −0.020] / −0.110 [−0.170, −0.060] | 14 / 17 episodes |

### 4.3 A − A-nocue, and A-nocue's agreement with G

- **The contrast.** A − A-nocue equals A − G in every cell: +0.030 [0.000, 0.070] and +0.040 [0.010, 0.080] on the shelf, −0.010 [−0.040, 0.020] and +0.040 [0.000, 0.090] on the tabletop.
- **The agreement.** At b = 3, A-nocue's paths, decisions and effort equal G's in all 600 episodes per environment, uncapped and at every cap. This was observed with `agreement.py`, a post-hoc script.
- **What this is.** It is observed agreement at the stated settings. A-nocue drops four features together (sep_occluded, obstruction_start, side_toward, side_missing), so no effect is attributed to the side descriptor alone.

### 4.4 A − F, A − R, G − F and G − R (b = 3, uncapped)

Removal recall, L1 / L2. False removals are 0 ZW throughout.

| Contrast | Shelf | Tabletop |
|---|---|---|
| A − F | +0.360 [0.290, 0.430] / +0.350 [0.280, 0.420] | +0.030 [−0.010, 0.070] / +0.050 [0.010, 0.100] |
| A − R | +0.203 [0.159, 0.244] / +0.191 [0.147, 0.234] | +0.008 [−0.017, 0.034] / +0.038 [0.012, 0.068] |
| G − F | +0.330 [0.250, 0.400] / +0.310 [0.240, 0.380] | +0.040 [−0.010, 0.090] / +0.010 [−0.040, 0.060] |
| G − R | +0.173 [0.123, 0.219] / +0.151 [0.103, 0.196] | +0.018 [−0.010, 0.046] / −0.002 [−0.033, 0.029] |

On the tabletop, the naive orders already resolve most removed episodes (F 82 / 76, R 84.2 / 77.25), so G's and A's margins over them are small.

### 4.5 Each multi-view policy against S (b = 3, uncapped; S at its actual effort of 1 view)

Removal recall, L1 / L2. Coverage rises by 0.46–0.59 in every case, and false removals are 0 ZW.

| Policy | Shelf | Tabletop |
|---|---|---|
| A | +0.850 [0.810, 0.890] / +0.850 [0.810, 0.890] | +0.850 [0.810, 0.890] / +0.810 [0.760, 0.860] |
| Dir | the same as A | the same as A |
| G, G-lookahead and A-nocue | +0.820 [0.770, 0.870] / +0.810 [0.750, 0.860] | +0.860 [0.820, 0.890] / +0.770 [0.710, 0.830] |
| G-myopic | +0.830 [0.780, 0.870] / +0.830 [0.780, 0.870] | +0.860 / +0.770 (= G) |
| R | +0.647 [0.602, 0.694] / +0.659 [0.613, 0.705] | +0.842 [0.805, 0.873] / +0.772 [0.716, 0.825] |
| F | +0.490 [0.420, 0.560] / +0.500 [0.430, 0.570] | +0.820 [0.770, 0.870] / +0.760 [0.690, 0.820] |

### 4.6 G − G-lookahead and G − G-myopic

- **G − G-lookahead.**
  - Identical uncapped and at the 1-, 3- and 4-step caps, with every interval 0 ZW.
  - At 2 steps: −0.280 [−0.340, −0.220] / −0.260 [−0.320, −0.210] on the shelf and −0.570 / −0.550 on the tabletop. Decisions differ in 54 / 112 episodes.
  - At 6 steps: −0.030 [−0.070, 0.000] / −0.050 [−0.090, −0.010] on the shelf and −0.030 / −0.040 on the tabletop.
- **G − G-myopic,** uncapped:
  - shelf −0.010 [−0.060, 0.040] / −0.020 [−0.070, 0.030], inconclusive;
  - tabletop identical (0 ZW).

### 4.7 X vs X-fcm (H1: the k = 2 rule against first-clear-miss, k = 1), b = 3, uncapped

Removal recall for X − X-fcm, L1 / L2.

| X | Shelf | Tabletop |
|---|---|---|
| F | −0.380 [−0.450, −0.310] / −0.380 [−0.450, −0.310] | −0.080 [−0.130, −0.030] / −0.090 [−0.150, −0.040] |
| R | −0.246 [−0.288, −0.203] / −0.240 [−0.285, −0.196] | −0.055 [−0.088, −0.026] / −0.093 [−0.137, −0.052] |
| G, G-lookahead, A-nocue | −0.080 [−0.130, −0.030] / −0.100 [−0.160, −0.050] | −0.040 [−0.080, −0.010] / −0.100 [−0.160, −0.050] |
| G-myopic | −0.070 [−0.120, −0.030] / −0.080 [−0.130, −0.030] | −0.040 [−0.080, −0.010] / −0.100 [−0.160, −0.050] |
| A, Dir | −0.050 [−0.090, −0.010] / −0.060 [−0.110, −0.020] | −0.050 [−0.090, −0.010] / −0.070 [−0.120, −0.030] |

- **Recall and effort.** The k = 1 rule resolves more removed episodes with fewer views.
- **False removals.** In shelf L2 every -fcm variant made 1 false removal out of 200 present episodes (R-fcm: 0.85, averaged over seeds). The X − X-fcm false-removal contrast there is −0.005 [−0.015, 0.000] (R −0.004 [−0.013, 0.000]): inconclusive, because each interval contains 0. Every k = 2 policy had none. Elsewhere the false-removal contrasts are 0 ZW.

### 4.8 Gate-off ablation (F and R only)

X − X-gateoff, L1 / L2:

| Policy | Shelf recall | Shelf false removal rate | Tabletop recall | Tabletop false removal rate |
|---|---|---|---|---|
| F | −0.510 [−0.580, −0.440] / −0.500 | −0.050 ZW / −0.050 ZW | −0.130 [−0.190, −0.080] / −0.210 [−0.280, −0.140] | −0.030 [−0.045, −0.015] / −0.040 [−0.050, −0.025] |
| R | −0.353 / −0.341 | −0.094 [−0.102, −0.087] / −0.093 | −0.128 / −0.212 | −0.076 / −0.079 |

Turning the occlusion gate off raises recall and adds false removals. Out of 200 present episodes:

| | Shelf L1 | Shelf L2 | Tabletop L1 | Tabletop L2 |
|---|---|---|---|---|
| F-gateoff | 10 | 10 | 6 | 8 |
| R-gateoff (seed-averaged) | 18.9 | 18.65 | 15.1 | 15.8 |

The shelf F contrast's ZW value has 10 discordant episodes in 10 arrangements per lighting.

### 4.9 τ_match risk–coverage sweep (b = 3, uncapped; the grid 0.40–0.70 and the fixed extension 0.75–0.90, secondary)

- **A plateau around the frozen 0.70.** Over these ranges, recall, coverage and both error counts equal their values at 0.70:

  | Policy | Shelf | Tabletop |
  |---|---|---|
  | A | 0.65–0.90 (L1), 0.65–0.85 (L2) | 0.60–0.90 |
  | G | 0.65–0.90 (L1), 0.65–0.85 (L2) | 0.60–0.90 |
  | Dir | 0.70–0.85 | 0.60–0.90 |

- **Below the plateau, false presents appear.**
  - Shelf at 0.60: 1 false present out of 100 removed episodes for A and G. At 0.40: 52 for A and G in both lightings, and 53 / 52 for Dir.
  - Tabletop at 0.55: 2–4.
- **Above it, at 0.90, false removals appear on the shelf.** Out of 200 present episodes, L1 / L2:

  | Policy | False removals |
  |---|---|
  | A, G, G-lookahead, A-nocue | 0 / 2 |
  | Dir | 1 / 3 |
  | G-myopic | 1 / 3 |
  | F | 7 / 7 |
  | R | 2.70 / 3.10 |

  On the tabletop only R adds any: 0.35 / 200 in L1. Recall is unchanged for A, G and Dir.

Every policy's full curves are in `table.md`.

### 4.10 A − G on the common targets (foam_brick, tuna_fish_can; b = 3, uncapped)

Within each family, the common targets carry equal weight.

| | Arrangements | Removal recall | Coverage | Discordant episodes |
|---|---|---|---|---|
| Shelf L1 | 72 | +0.028 [0.000, 0.070] | +0.009 [0.000, 0.023] | 2 |
| Shelf L2 | 72 | +0.028 [0.000, 0.070] | +0.009 [0.000, 0.023] | 2 |
| Tabletop L1 | 40 | 0 ZW | 0 ZW | 0 |
| Tabletop L2 | 40 | +0.050 [0.000, 0.125] | +0.017 [0.000, 0.042] | 2 |

False removals are 0 ZW in every row. In tabletop L1, A and G decide alike on every episode, and their effort differs on 11 of 120 episodes (73.0 vs 68.7 cm).

### 4.11 Breakdowns by family (H3, descriptive; b = 3, uncapped)

A − G removal recall, L1 / L2. The occluder-family level, with M16's sides pooled, is secondary reporting. The slot-class rows and the side-stratified resampling are kept beside it.

| Family (n) | Shelf | Tabletop |
|---|---|---|
| IND28 (10) | 0 ZW (every policy 0 / 10) | 0 ZW (every policy 0 / 10) |
| M16 pooled (30) | +0.033 [0.000, 0.100] / +0.033 [0.000, 0.100] | 0 ZW / +0.033 [0.000, 0.100] |
| — M16-L (15) | 0 ZW (A and G 15 / 15) | 0 ZW |
| — M16-R (15) | +0.067 [0.000, 0.200] / +0.067 [0.000, 0.200] | 0 ZW / +0.067 [0.000, 0.200] |
| S8 (20) | +0.050 [0.000, 0.150] / +0.100 [0.000, 0.250] | 0 ZW (every multi-view policy but F and R 20 / 20) |
| white_box (20) | +0.050 [0.000, 0.150] / +0.050 [0.000, 0.150] | 0 ZW / +0.100 [0.000, 0.250] |
| ycb_box (20) | 0 ZW | −0.050 [−0.200, 0.100] / +0.050 [−0.100, 0.200] (3 discordant each) |

- **IND28.** No arrangement has a geometrically diagnostic non-start stop (0 / 10 in both environments). These episodes are unresolvable by design.
- **Same decisions, different effort.** Where the interval is ZW, A and G decide alike on every episode, and their effort can still differ:
  - IND28: on the shelf in 20 of 30 episodes (298.8 vs 232.4 cm); on the tabletop in 16 and 13 of 30 (135.1 and 130.4 vs 110.0 cm);
  - shelf ycb_box: 4 of 60 (142.7 vs 136.1 cm);
  - a few tabletop M16-L and white_box episodes.
- **Target composition** (arrangements per target).
  - Shelf:
    - M16: foam_brick 15, tuna_fish_can 15 (M16-L 8 / 7, M16-R 7 / 8);
    - ycb_box: foam_brick 10, tuna_fish_can 10;
    - S8: foam_brick, mug, potted_meat_can and tuna_fish_can, 5 each, with no tomato_soup_can;
    - white_box: all five targets, 4 each;
    - IND28: all five targets, 2 each.
  - Tabletop: all five targets in every family (M16 6 each, split 3 / 3 by side; S8, white_box and ycb_box 4 each; IND28 2 each).

### 4.12 Sensitivity runs (reported only, never selected; b = 3, uncapped)

**k = 3** (three clear misses for REMOVED). False removals and false presents stay at 0, and recall falls sharply. Removal recall, L1 / L2:

| Policy | Shelf | Tabletop |
|---|---|---|
| A, Dir | 0.450 / 0.410 | 0.710 / 0.640 |
| G, G-lookahead, A-nocue | 0.240 / 0.240 | 0.550 / 0.490 |
| G-myopic | 0.270 / 0.250 | 0.550 / 0.490 |
| R | 0.160 / 0.147 | 0.441 / 0.403 |
| F | 0.060 / 0.040 | 0.340 / 0.340 |

**λ = 0.7, 0.9 and 1.2** (G and G-lookahead). False removals are 0.

| λ | Shelf recall (L1 / L2) | Shelf travel | Tabletop recall | Tabletop travel |
|---|---|---|---|---|
| 0.7 and 0.9 | 0.830 / 0.830 | 125.5 / 126.5 cm, 2.31 views | 0.850 / 0.810 | 52.5 / 53.0 cm, 2.21 / 2.22 views |
| 1.2 | 0.830 / 0.830 | 108.9 / 110.2 cm, 2.37 views | 0.850 / 0.810 | 42.2 / 42.8 cm, 2.26 / 2.27 views |

At λ = 0, G gets 0.820 / 0.810 (shelf) and 0.860 / 0.770 (tabletop).

### 4.13 Diagnostics (§10)

- **Path and decision disagreement, A vs G** (b = 3, uncapped). Paths differ in 0.41 / 0.25 of episodes on the shelf (L1 / L2) and 0.39 / 0.27 on the tabletop. Decisions differ in 0.01 / 0.01 and 0.01 / 0.02.
- **Geometric diagnostic coverage.** In 90 / 100 arrangements per environment, occluded episodes have at least one diagnostic non-start stop and removed episodes at least two. The exceptions are the 10 IND28 arrangements.
- **Gate per-view rates at the frozen thresholds.** The views of one arrangement are correlated.

  | Rate | Shelf L1 | Shelf L2 | Tabletop L1 | Tabletop L2 |
  |---|---|---|---|---|
  | False-clear rate (non-diagnostic views of occluded-present episodes) | 35 / 291 = 0.120 | 32 / 291 = 0.110 | 16 / 211 = 0.076 | 9 / 211 = 0.043 |
  | False clear-miss rate (diagnostic present views) | 0 / 709 | 0 / 709 | 0 / 789 | 0 / 789 |
  | Clear-miss rate (diagnostic removed views) | 199 / 209 | 197 / 209 | 266 / 289 | 249 / 289 |
  | False strong-match rate (diagnostic removed views) | 0 / 209 | 0 / 209 | 0 / 289 | 0 / 289 |
  | Strong match on any removed view | 0 / 500 | 0 / 500 | 0 / 500 | 0 / 500 |

- **The rest.** Distinct view sets per budget and cap, budget curves, per-condition tables, cap curves and the pooled-lighting section are in `table.md`. The pooled section is secondary, with no bootstrap.

### 4.14 View-level reporting (Amendment 2; `tools/registration_compare.py --report-only`)

Source: `runs/2026-11-06-test/<env>/views/C0_refL1.md`. The harness C0 equals the evidence of record (largest difference 0, refusals equal).

| C0, L1 / L2 | Shelf | Tabletop |
|---|---|---|
| Registration refusals | 0 / 1500, 0 / 1500 | 48 / 1500, 50 / 1500 |
| Present-labelled clear misses | 0 / 1000, 1 / 1000 | 0 / 1000, 0 / 1000 |
| Removed-labelled strong matches | 0 / 500, 0 / 500 | 0 / 500, 0 / 500 |
| Diagnostic removal yield | 199 / 209, 197 / 209 | 266 / 289, 249 / 289 |
| Present gate pass | 500 / 500, 500 / 500 | 499 / 500, 471 / 500 |
| Target-plane error, common accepted views (median / p90 / max px) | 1.37 / 2.66 / 5.04; 1.44 / 2.80 / 5.49 | 2.14 / 4.87 / 47.59; 2.15 / 4.74 / 47.54 (proxy plane) |
| Front-plane error, common accepted (px) | 0.64 / 1.39 / 3.17; 0.65 / 1.40 / 3.38 | — |
| Table-plane error, common accepted (px) | — | 1.76 / 5.13 / 51.19; 1.78 / 5.10 / 51.16 |
| Silhouette IoU, common accepted (median / p90 / max) | 0.96 / 0.99 / 1.00; 0.96 / 0.99 / 0.99 | 0.98 / 0.99 / 1.00; 0.98 / 0.99 / 1.00 |

On the tabletop, the target plane is a proxy: the horizontal plane through the target's centre, with the table plane beside it (SIM_PREREGISTRATION §9). A plane disagreement is not an error against a true whole-scene alignment.

## 5. Limitations (reported with the results, not retuned)

- **The shelf L1 gate false-clear subgroup.** In development it was 21 / 176, above alpha_clear = 0.10, while the pooled constraint held.
  - On the test, both shelf lightings exceed alpha_clear: 35 / 291 = 0.120 (L1) and 32 / 291 = 0.110 (L2).
  - The tabletop stays below it: 16 / 211 = 0.076 and 9 / 211 = 0.043.
  - These are per-view rates, and the views of one arrangement are correlated.
  - At b = 3 no k = 2 policy made a false removal. The k = 1 variants made one, in shelf L2 (§4.7).
- **The tabletop L2 design cases** (TDES0–TDES3) failed the gate on their visible-phase views in development.
  - The test has no design case.
  - On the test, the tabletop L2 visible-phase present gate pass is 471 / 500, against 499 / 500 in L1. That is the same lighting direction. It is reported only.
- **C0's failed tabletop refusal criterion** (< 5 %) stands as recorded: 12 / 165 (L1) and 9 / 165 (L2) views refused on the first 11 tabletop development arrangements. The test's tabletop refusals, 48 / 1500 and 50 / 1500, are reported only, with no criterion applied.
- **Little room at the primary** (Addendum 2, S2; development data, conditional).
  - On the pooled development replays, an evidence-aware ordering oracle exceeds G by 6 of 240 removal episodes (196 vs 190), and G equals the oracle on present episodes.
  - The oracle reads future evidence. That count bounds those development episodes only.
- **The test-seed incident** is in `notes/2026-11-06-test-seed-incident.md`. The seed schedule was kept.
- **The timestamp** went to the researcher's own second mailbox, not the instructor (Addendum 1).
- **Post-hoc agreement counts.** The episode-level agreement counts (A vs Dir paths, and A-nocue = G) come from `agreement.py`, a post-hoc script outside the freeze record. Its overlapping counts match verify.run's diagnostics.
- **The methodology audit's limits** (Addendum 2; development data) bound what these results say about learning:
  - A's history-only features cannot reorder candidates;
  - its target is geometric visibility, not terminal verification value;
  - it has no remaining-budget input;
  - the scenes restrict occlusion complexity: one designated occluder, two clear stops except IND28, and a fixed start at P3.

  The test is consistent with the no-budget-input limit: under the 2-step cap A resolves no removed episode, as G does, while Dir and G-lookahead do.
- **Scope.** Simulation only, with two environments, two lightings, five stops, one start stop and the YCB-Video objects. The evidence is the frozen C0 pipeline.

## 6. What the results support

### Supported

Each claim is for its environment, lighting and metric.

- **The primary.** At b = 3 uncapped, A's removal recall exceeded G's in shelf L2: +0.040 [0.010, 0.080], or 85 vs 81 of 100 removed episodes.
  - This is one cell of four, read pointwise and unadjusted.
  - It rests on 4 discordant episodes in 4 arrangements, all favouring A. The interval's lower bound, 0.010, is one episode.
  - In shelf L1, tabletop L1 and tabletop L2 the difference is inconclusive (+0.030, −0.010 and +0.040); the intervals contain 0.
  - Neither policy made a false removal or a false present in any cell.
- **A and Dir, at b = 3.** A's decisions equal Dir's in every episode, uncapped and under the 1-, 3- and 4-step caps. Dir has no fitted parameter and uses the same start-view descriptor.
  - Dir travels 37–50 % less uncapped and under the 3- and 4-step caps (38–50 % uncapped). Under the 1-step cap the two runs are identical.
  - The image cost points the other way: uncapped, A takes slightly fewer views in three cells (2.14 vs 2.25 / 2.27 on the shelf, 2.14 vs 2.16 in tabletop L1) and the same in tabletop L2 (2.17). Both costs are reported.
  - So at the primary setting, the A − G differences above are equally differences between Dir and G.
  - This is observed agreement, not a causal attribution, and it is specific to b = 3: at b = 2, A's counts equal G's.
- **A-nocue and G.** At b = 3, A-nocue's paths, decisions and effort equal G's in every episode at every cap. This is observed; A-nocue drops four features together.

### Secondary and descriptive

These are not findings on a par with the primary, and none is promoted to it.

- **Travel caps.**
  - At 3 and 4 steps, A (= Dir) resolves more removed episodes than G-lookahead: +0.14 to +0.18, with each of the four cells' intervals excluding 0, read pointwise.
  - At 2 steps, A resolves no removed episode, as G does (both still resolve 190 of 200 present episodes per cell), while G-lookahead resolves 26–57 and Dir 42–73 removed episodes of 100.
  - At 6 steps, A decides exactly as G does.
    - It is below Dir in all four cells, and those intervals exclude 0.
    - It is below G-lookahead in shelf L2 (−0.050 [−0.090, −0.010]) and tabletop L2 (−0.040 [−0.080, −0.010]).
    - It is inconclusive against G-lookahead in shelf L1 and tabletop L1 (−0.030 [−0.070, 0.000]).
- **The stopping rule.**
  - k = 2 had 4–38 points lower removal recall than k = 1 in every cell, with the pointwise intervals excluding 0.
  - k = 2 made 0 false removals out of 200 present episodes per cell. k = 1 made 1 out of 200 in shelf L2, with R-fcm at 0.85 averaged over seeds, and 0 elsewhere.
  - The false-removal contrast is inconclusive (−0.005 [−0.015, 0.000]), so no false-removal advantage is shown.
- **τ_match.** The frozen 0.70 sits on a plateau. Lower values add false presents. At 0.90 several policies add shelf false removals (§4.9).

### Not supported

- Any general claim that learned view selection is better or worse than geometric selection. The supported A − G result above is narrower: one fitted ranker, one cell, read pointwise.
- Any decision-performance benefit of A beyond Dir. At the primary setting (b = 3, uncapped) the test demonstrates none: their decisions are identical, as they are under the 1-, 3- and 4-step caps, and Dir resolves more removed episodes under the 2- and 6-step caps. This agreement is observed; it does not establish what causes the A − G difference.
- Any claim about the side descriptor alone.
- Any equivalence claim.
- Any claim about learned active perception in general. Under Addendum 2, these results test one restricted, budget-blind A on scenes with limited occlusion complexity, not history-dependent or budget-aware learned selection.
