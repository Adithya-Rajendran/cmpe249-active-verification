# Post-hoc review of the frozen test (2026-11-12)

**SIMULATION. Post hoc and exploratory.** Every analysis below was chosen and run after all test outcomes of `simulation-freeze-1` had been read and reported (`notes/2026-11-06-test-results.md`). None is a frozen analysis. No frozen file changed, and no estimate of the frozen analysis changed. `docs/FREEZE_ADDENDA.md`, Addendum 3, records this note.

- **Who.** The researcher asked for an independent review of the completed study on 2026-11-12. An AI assistant (Claude) ran it. The researcher decides what enters the manuscript.
- **Evidence.** `runs/2026-11-12-independent-review/` in the main checkout (untracked): the scripts `rep.py`, `analysis.py`, `contrasts.py` and `process_checks.py`, their outputs `out_*.txt`, and `data/`, copies of the evidence, side-descriptor, label and manifest files of the test and development sets from the share.
- **Inputs.** The frozen per-view evidence and side descriptors of the test set, the frozen thresholds (τ_q 0.3, τ_occ 0.4, τ_match 0.7, k 2), the two frozen ranker models, and the development files that the frozen calibration names (`shelf_dev_v3`, `tabletop_dev60_merged`).
- **Counts** are removed episodes resolved of 100 per environment and lighting unless stated. Intervals are the study's interval: the 95 % percentile interval of the arrangement-level paired bootstrap, stratified by slot class, B = 10,000. They are pointwise and unadjusted.

## 1. Replay check

`rep.py` implements the stopping rule and the policies F, G, G-lookahead, Dir, A and A-nocue again. It imports nothing from the study's code.
- **Evidence.** The 6,000 evidence rows in the saved run logs equal the evidence files on the share.
- **Runs.** Its replays equal the saved logs in decision, path and travel in 172,800 of 172,800 runs: six policies, b = 1 to 4, six travel caps, 600 episodes, two environments (`out_replay_check.txt`).
- **Intervals.** Its bootstrap gives the four primary intervals of the results note: [0.000, 0.070], [0.010, 0.080], [−0.040, 0.020] and [0.000, 0.090].

## 2. The primary comparison: exact tests, and arrangements shared between lightings

`out_analysis.txt`, section A.

| Cell | A / G | Discordant arrangements for A / for G | Exact sign test, two-sided |
|---|---|---|---|
| Shelf L1 | 85 / 82 | 3 / 0 | p = 0.250 |
| Shelf L2 | 85 / 81 | 4 / 0 | p = 0.125 |
| Tabletop L1 | 85 / 86 | 1 / 2 | p = 1.000 |
| Tabletop L2 | 81 / 77 | 5 / 1 | p = 0.219 |

- **The shelf L2 interval.** A percentile bootstrap interval excludes 0 whenever fewer than 2.5 % of the resamples hold no discordant arrangement. With four one-sided discordant arrangements of 100, that share is about 0.017; with three it is about 0.048. So the interval of shelf L2 excludes 0 and that of shelf L1 does not, although they differ by one arrangement. The exact test of shelf L2 gives p = 0.125.
- **The lighting cells are not separate samples.** L1 and L2 use the same 100 arrangements. On the shelf, 3 of the 4 discordant arrangements of L2 are the 3 of L1. On the tabletop, 1 arrangement is discordant in both lightings.
- **One unit per arrangement** (the two lightings summed): shelf 4 for A and 0 for G (p = 0.125); tabletop 6 and 2 (p = 0.289); both environments, 200 arrangements, 10 and 2 (p = 0.039). This pooling was not declared before the test.

## 3. Fixed orders

A *fixed order* is a priority list of the four stops. At each step the policy takes the first affordable unvisited stop of the list. It reads no image to choose a stop; the shared stopping rule still reads the views it takes. F is the fixed order P1, P2, P4, P5.

### 3.1 G is a fixed order on the test set

With early stopping off, G's complete order is P1, P5, P4, P2 in 600 of 600 shelf test episodes and in 586 of 600 tabletop test episodes (the other 14: P1, P5, P2, P4). A-nocue's is P1, P5, P4, P2 in 600 of 600 episodes in both environments. Preregistration §14 records the same for G on the shelf development set.

### 3.2 All 24 fixed orders at the primary setting

Removed episodes resolved of 200 (both lightings), b = 3, no travel cap, by the set of the first three stops (`out_fixed_order_groups.txt`). The order within the set does not change the count.

| First three stops | Shelf | Tabletop |
|---|---|---|
| P1, P2, P5 | 166 | 166 |
| P1, P4, P5 (G's) | 163 | 163 |
| P1, P2, P4 (F's) | 99 | 158 |
| P2, P4, P5 | 98 | 161 |
| A, for comparison | 170 | 166 |

- F's first three stops leave out P5. On the shelf, a removed episode's clear misses are most often at the two end stops only (pattern X...X in §5), so an order without both end stops among its first three resolves about half as many.
- No fixed order made a false removal or a false present, and each resolved 190 of 200 present episodes per cell.

### 3.3 A fixed order chosen on the development data (Fix)

- **Rule.** For each environment, budget and travel cap, each of the 24 orders is replayed on that environment's 60 development arrangements with the frozen thresholds. The order with the most removed episodes resolved and no false removal is taken. Ties go to the shorter mean travel on removed episodes, then to the lower order in sequence.
- **The rule was written after the test.** It uses development data only, but it was chosen by a reviewer who knew the test outcomes. Section 3.2 shows how far the choice can matter at the primary setting: any order with both end stops among its first three gives 163 or 166 of 200.
- **Chosen routes** (the stops reached under the cap): no cap, shelf P2, P1, P5 and tabletop P4, P5, P1; 2 and 3 steps, P4, P5 in both; 4 steps, P2, P4, P5 in both; 6 steps, as with no cap.
- **Hindsight best.** For each episode, the best of the 24 orders at that budget and cap, found with all five views known. Every path a selector can take under the shared stopping rule is the path of one of the 24 orders, so this is a ceiling for any selector at that setting. It is not an attainable policy.

**Table.** Removed episodes resolved of 100, L1 / L2, at b = 3 (`out_contrasts.txt`). No entry includes a false removal or a false present.

| Setting | G | G-lookahead | Dir | A | Fix | Hindsight best |
|---|---|---|---|---|---|---|
| Shelf, 1 step | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Shelf, 2 steps | 0 / 0 | 28 / 26 | 46 / 42 | 0 / 0 | 24 / 25 | 46 / 47 |
| Shelf, 3 steps | 28 / 26 | 28 / 26 | 46 / 42 | 46 / 42 | 24 / 25 | 46 / 47 |
| Shelf, 4 steps | 28 / 26 | 28 / 26 | 46 / 42 | 46 / 42 | 48 / 50 | 52 / 54 |
| Shelf, 6 steps | 79 / 77 | 82 / 82 | 85 / 85 | 79 / 77 | 83 / 83 | 86 / 86 |
| Shelf, no cap | 82 / 81 | 82 / 81 | 85 / 85 | 85 / 85 | 83 / 83 | 86 / 86 |
| Tabletop, 1 step | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Tabletop, 2 steps | 0 / 0 | 57 / 55 | 73 / 69 | 0 / 0 | 61 / 52 | 86 / 77 |
| Tabletop, 3 steps | 57 / 55 | 57 / 55 | 73 / 69 | 73 / 69 | 61 / 52 | 86 / 77 |
| Tabletop, 4 steps | 57 / 55 | 57 / 55 | 73 / 69 | 73 / 69 | 85 / 76 | 87 / 79 |
| Tabletop, 6 steps | 78 / 70 | 81 / 74 | 84 / 81 | 78 / 70 | 86 / 77 | 89 / 82 |
| Tabletop, no cap | 86 / 77 | 86 / 77 | 85 / 81 | 85 / 81 | 86 / 77 | 89 / 82 |

With no cap the hindsight best equals FULL (all five views). At b = 2 with no cap, Fix, A and G give the same counts (79 / 77 on the shelf, 78 / 70 on the tabletop), Dir gives 46 / 42 and 73 / 70, and the hindsight best is again 86 / 86 and 89 / 82.

**Paired contrasts** (difference in removal recall [interval]; discordant arrangements one way / the other; exact sign test). Order of cells: shelf L1, shelf L2, tabletop L1, tabletop L2.

| Contrast | Cells |
|---|---|
| A − Fix, no cap | +0.02 [−0.020, 0.060], 3 / 1, p = 0.625 · +0.02 [−0.020, 0.060], 3 / 1, p = 0.625 · −0.01 [−0.040, 0.020], 1 / 2, p = 1.000 · +0.04 [0.000, 0.090], 5 / 1, p = 0.219 |
| G − Fix, no cap | −0.01 [−0.060, 0.040], 3 / 4, p = 1.000 · −0.02 [−0.070, 0.030], 3 / 5, p = 0.727 · 0, no discordant arrangement · 0, no discordant arrangement |
| Dir − Fix, 2 steps | +0.22 [0.180, 0.270], 22 / 0, p < 0.001 · +0.17 [0.110, 0.240], 22 / 5, p = 0.002 · +0.12 [0.050, 0.200], 21 / 9, p = 0.043 · +0.17 [0.100, 0.250], 24 / 7, p = 0.003 |
| G-lookahead − Fix, 2 steps | +0.04 [−0.030, 0.110], 22 / 18, p = 0.636 · +0.01 [−0.060, 0.080], 22 / 21, p = 1.000 · −0.04 [−0.140, 0.060], 25 / 29, p = 0.684 · +0.03 [−0.060, 0.120], 25 / 22, p = 0.771 |
| Fix − A, 4 steps | +0.02 [−0.020, 0.060], 3 / 1, p = 0.625 · +0.08 [0.030, 0.140], 9 / 1, p = 0.021 · +0.12 [0.060, 0.180], 13 / 1, p = 0.002 · +0.07 [0.010, 0.130], 10 / 3, p = 0.092 |
| Fix − G-lookahead, 4 steps | +0.20 [0.150, 0.260], 21 / 1 · +0.24 [0.180, 0.310], 25 / 1 · +0.28 [0.200, 0.360], 30 / 2 · +0.21 [0.140, 0.280], 24 / 3; each p < 0.001 |
| Fix − A, 6 steps | +0.04 [0.010, 0.080], 4 / 0, p = 0.125 · +0.06 [0.020, 0.110], 6 / 0, p = 0.031 · +0.08 [0.040, 0.130], 8 / 0, p = 0.008 · +0.07 [0.030, 0.120], 7 / 0, p = 0.016 |

- **At the primary setting** Fix resolves 329 of 400 removed episodes over the four cells, A and Dir 336, and G 326. On the tabletop Fix visits G's three stops and gives G's decisions.
- **Effort at the primary setting** (mean views; mean travel in cm, over all 300 episodes of a cell): Fix 2.37; 108.9, 2.37; 110.2, 2.25; 41.0 and 2.28; 43.8. A: 2.14; 154.9, 2.14; 155.9, 2.14; 72.7 and 2.17; 74.1. Dir: 2.25; 94.1, 2.27; 96.9, 2.16; 36.4 and 2.17; 38.0. Fix takes more views than A and travels 29–44 % less.
- **Under the 2- and 3-step caps** Dir resolves 12 to 22 more removed episodes per 100 than Fix. By occluder family at 2 steps, both lightings together: on the shelf M16 gives Dir 58 and Fix 28 of 60, S8 17 and 8 of 40, and the two box families are equal (7 and 7, 6 and 6); on the tabletop M16 gives 56 and 27 of 60, S8 35 and 30, the white box 24 and 31, and the YCB box 27 and 25. So M16, the family with a designed open side, carries 30 of Dir's 39 extra shelf removals and 29 of its net 29 on the tabletop.
- **Under the 4-step cap** the fixed order P2, P4 and then an end stop reaches three stops within four steps. Dir and A go to the near and the far stop of one side and can reach no third stop. G-lookahead checks only that REMOVED stays reachable; it does not look for the route with the most stops.

## 4. A and Dir visit the same stops

At b = 3 with no cap, A and Dir visit the same set of three further stops in 600 of 600 shelf episodes and in 598 of 600 tabletop episodes (`out_analysis.txt`, section C).
- A's first two stops are the two end stops in all 1,200 episodes. Its third stop is the inner stop on the side the descriptor names.
- Dir takes the inner and then the end stop on that side, and then the other end stop.
- No removed view is a strong match, and no present episode has two clear misses (§6). The decision therefore depends only on which stops are visited before the budget ends. The equal decisions of A and Dir at this setting follow from the equal sets.

## 5. Where the clear misses are in removed episodes

`out_analysis.txt`, section D. Pattern over P1 to P5, X = clear miss, both lightings.
- **The start view is never a clear miss** (0 of 400 removed episodes). With k = 2, one further view cannot give REMOVED, and neither can a 1-step cap, which reaches one further stop.
- **Shelf.** S8, the white box and the YCB box most often leave clear misses at the two end stops only (X...X: 13 of 40, 27 of 40 and 26 of 40). M16 leaves the end stop of one side and both stops of the other (28 of 30 per side).
- **Tabletop.** S8 and the white box most often leave all four further stops clear (XX.XX: 28 of 40 and 21 of 40).
- **IND28** leaves none, by design (39 of 40; one episode has one).

## 6. Safety margin and separability

`out_analysis.txt`, section E.
- **Present episodes.** Among all five views, 1 of 800 present episodes has one clear miss (a shelf IND28 episode, at P1), and none has two. So no order of stops could have given a false removal at k = 2 on this test set, and at most one episode could have given one at k = 1. The test cannot compare selectors on false removals. It cannot compare k = 1 with k = 2 on them either, beyond that one episode.
- **Hidden views.** A view of a fully hidden present target (visible fraction ≤ 0.05) passes the obstruction test in 1 of 640 cases.
- **Match scores.** For a present target at a geometrically diagnostic view the minimum is 0.843 (shelf) and 0.866 (tabletop). For a removed target the maximum is 0.660 and 0.597. The threshold is 0.70. The development sets separate in the same way (minimum 0.858 and 0.817; maximum 0.651 and 0.676).
- **What differs between a reference and a revisit** in these scenes: the occluder, the target, the lighting and a small camera offset. The distractors do not move.

## 7. Calibration recount and alignment refusals

`out_process_checks.txt`.
- **Recount.** At the frozen thresholds the pooled development counts are 521 / 568 correct clear misses, 0 / 1,768 false clear misses and 50 / 632 false clears, as in `frozen/frozen_candidate.json`.
- **Which threshold the data set.** τ_occ: 0.40 gives a false-clear rate of 0.079, and 0.45 gives 0.127, above the limit of 0.10. τ_q: the false-clear count stays 50 / 632 from 0 to 0.40. τ_match: the correct clear-miss count is 521 from 0.70 to 0.85, and the tie-break takes the lowest value.
- **False presents** have no limit in the calibration. The highest match score of a removed target in the development data is 0.676, on the tabletop.
- **Refusals.** The alignment refuses 48 and 50 of 1,500 tabletop test views (L1, L2) and 28 and 24 of 900 tabletop development views; none on the shelf. The development criterion of 2026-11-04 was fewer than 5 %. The first 11 tabletop arrangements had given 12 and 9 of 165.

## 8. What this adds to the reading of the results note (§6)

- **The primary.** "A's removal recall exceeded G's in shelf L2" holds for the declared interval. The exact test does not support a difference in any single cell. The two shelf cells rest on nearly the same arrangements.
- **A and Dir.** Their equal decisions at b = 3 with no cap are a consequence of equal visited sets, not an independent agreement in each of 1,200 episodes.
- **Naive order.** F's deficit on the shelf comes from the stops its order reaches within the budget. A fixed order chosen on development data is within 0.02 to 0.04 of A in three cells and 0.01 above it in one, and no cell's exact test separates them.
- **Travel caps.** The start-view descriptor helps under the 2- and 3-step caps, mostly through M16. Under the 4- and 6-step caps a fixed order resolves more than A. G-lookahead is not a strong planning baseline.
- **Safety.** The absence of false removals reflects the evidence step on these scenes. It says nothing about the selectors.

## Limits

- Everything here is post hoc. Which analyses to run, the rule that chooses Fix, and which contrasts to report were all decided with the test outcomes known.
- Fix is chosen with the frozen thresholds on all 60 development arrangements of an environment, not with fold-wise thresholds. With the same settings my replay of the development set gives G 192, A 193 and Dir 193 of 240 at b = 3 with no cap. Addendum 2 quotes 190 for G from the methodology audit; I did not trace the difference.
- The literature was not re-checked for this note.
