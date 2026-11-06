# Final pooled calibration with A, A-nocue and Dir: development results (2026-11-06)

**Simulation, development data.** This is the planned final shared calibration (SIM_PREREGISTRATION §7; Amendment 2
of `notes/2026-11-04-registration-repair-plan.md`). One global threshold set is calibrated on both environments'
development arrangements, with complete grouped cross-validation. The held-out policy check includes A, A-nocue and
the direction-rule control Dir. Every number is a development estimate. The 49 tabletop arrangements in it are
development data (`notes/2026-11-06-tabletop-dev49-transfer.md`), not an independent test. Nothing here is a test
result.

## What ran

- **Code.** Commit 78263bc (`verify/select.py` select-v4 revision 2, `verify/ranker.py`, `verify/policies.py`,
  `verify/side.py`).
- **Command.** `python -m verify.select` with two dev sets, `--ranker A --ranker A-nocue --model-id v03_image --tag
  pooled-dev120`, on the Jetson: 4 min 18 s, exit 0. Output: `runs/2026-11-06-pooled-dev120/select/` (this worktree,
  untracked). The command line is in `progress/LOG.md`.
- **Inputs** (share `sim/`; every sha256 is in the report's header):
  - shelf: `shelf_dev_v3`, evidence of record `v03_image_refL1.jsonl` (860b9da4...), descriptors
    `side/evidence_shelf_dev_v3_side_8291f9a1394d_e6fdc8a63f/side_refL1.jsonl` (cdd080f2...);
  - tabletop: `tabletop_dev60_merged`, the first 11 and the 49 merged by `tools/merge_dev_sets.py` (ace89e2;
    evidence 79a8eff0..., descriptors 8d36e497..., `merge_provenance.json`).
- **Scale.** 120 arrangements, 720 episodes and 3600 views. The held-out check made 234 720 replays.

## The candidate

`runs/2026-11-06-pooled-dev120/select/frozen_candidate.json`, sha256
3a45b23cbe82856660a9c1e34de3074238b8d8204512663d972a215939e9c26d. **eligible_for_freeze: true**: calibration succeeded
on all development data and in all 120 leave-one-arrangement-out folds.

| tau_q | tau_occ | tau_match | k | lam |
|---|---|---|---|---|
| 0.3 | 0.4 | 0.7 | 2 (fixed) | 0 (fixed) |

The shelf-only candidate had tau_occ 0.35 and tau_match 0.65.

- **Joint search.**
  - 539 of 693 grid settings meet both alpha_clear constraints.
  - The chosen setting is the unique maximum of correct clear misses: 521 / 568 (0.917) diagnostic removed views.
  - False clear-miss rate: 0 / 1768. False-clear rate: 50 / 632 (0.079, alpha_clear 0.10). Strong matches on removed
    views: 0 / 568.
- **Subgroups** (reported, not adjusted; the constraint is defined on the pooled views):
  - the gate's false-clear rate on shelf L1 views is 21 / 176 (0.119);
  - held out, shelf L1 + L2 is 37 / 352 (0.105).
- **Gate pass at the chosen setting, all dev views:**

  | Lighting | Removed diagnostic | Present diagnostic |
  |---|---|---|
  | L1 | 266 / 284 | 866 / 884 |
  | L2 | 255 / 284 | 796 / 884 |

  Three folds hold out an arrangement whose removed diagnostic views all fail the gate in a lighting: TD5042 L2
  (0 / 3), TD5043 L2 (0 / 2) and TD5055 L1 and L2 (0 / 4 each).
- **Folds.**
  - 118 folds choose the same thresholds.
  - The fold without SD1005 chooses tau_q 0.65.
  - The fold without TD5017 chooses tau_match 0.65. Without TD5017's removed views, 0.65 and 0.7 tie on correct clear
    misses, and the declared tie-break takes the lower tau_match. Held out, that fold then calls TD5017-removed
    PRESENT in both lightings. This is the only false present in the held-out check.
- **Held-out per-view rates** (L1 + L2): false-clear 50 / 632; false clear-miss 0 / 1768; clear miss on removed
  518 / 568; strong match on removed 2 / 568 (TD5017).
- **Dev check (Step 4).** No policy is above FRR 0.1 at any setting, pooled or per environment.

## A and A-nocue

- **Fold-trained** (120 folds):
  - A: C = 10 in 103 folds and C = 1 in 17.
  - A-nocue: C = 1 in all 120.
  - 9142-9262 training rows per fold; label rate 0.629-0.637.
- **Frozen**, trained on all 120 arrangements with the candidate's thresholds (`python -m verify.ranker`; output
  `runs/2026-11-06-pooled-dev120/models/`):
  - A: C 10, 9302 rows; sha256 297e11af452d169525dd60c974291e5969e7c124194a809398f7519663381a19;
  - A-nocue: C 1, 9302 rows; sha256 eab90ded6364d689d59104070b52fd47d851818ca9aebd1d718ee3c8c93370c8.

Standardised coefficients:

| Feature | A, folds: mean [min, max] | A, frozen | A-nocue, folds: mean | A-nocue, frozen |
|---|---|---|---|---|
| az_from_start | +0.938 [+0.864, +0.994] | +0.948 | +0.994 | +0.995 |
| sep_visited | -0.540 [-0.689, +0.097] | -0.644 | +0.439 | +0.439 |
| sep_occluded | +1.283 [+0.637, +1.423] | +1.388 | - | - |
| obstruction_start | +0.055 [+0.005, +0.121] | +0.056 | - | - |
| side_toward | +0.879 [+0.791, +0.924] | +0.889 | - | - |
| side_missing | -0.291 [-0.326, -0.179] | -0.306 | - | - |
| match_max | -0.032 [-0.103, +0.002] | -0.034 | -0.020 | -0.020 |
| quality_min | +0.566 [+0.409, +0.642] | +0.577 | +0.470 | +0.470 |

The frozen models pass verify.run's checks against the candidate. On each environment's development data,
`verify.run` (real-v5) with `--candidate`, `--side` and both `--model` files exits 0. It records the candidate, model
and descriptor sha256 (`runs/2026-11-06-pooled-dev120/devrun_shelf/`, `devrun_tabletop/`). Those runs are a pipeline
check: the frozen models and thresholds were fitted on the same arrangements.

## Held-out policy check, b = 3

Each fold's thresholds, and each fold's own A and A-nocue, run on the held-out arrangement. The table gives correct
removals among 60 removed episodes per environment and lighting. R is the expectation over 20 seeds.

| Travel | Policy | Shelf L1 | Shelf L2 | Tabletop L1 | Tabletop L2 |
|---|---|---|---|---|---|
| uncapped | F | 28 | 27 | 43 | 40 |
| uncapped | R | 37.65 | 35.55 | 47.25 | 44.35 |
| uncapped | **G** | **47** | **45** | **51** | **47** |
| uncapped | **A** | **48** | **47** | **51** | **45** |
| uncapped | A-nocue | 47 | 45 | 51 | 47 |
| uncapped | Dir | 48 | 47 | 51 | 45 |
| uncapped | FULL | 49 | 47 | 51 | 47 |
| 1 step | every policy | 0 | 0 | 0 | 0 |
| 2 steps | G (= F, = A, = A-nocue) | 0 | 0 | 0 | 0 |
| 2 steps | G-lookahead | 15 | 15 | 28 | 29 |
| 2 steps | Dir | 28 | 26 | 44 | 41 |
| 3 and 4 steps | G (= F, = G-lookahead, = A-nocue) | 15 | 15 | 28 | 29 |
| 3 and 4 steps | A (= Dir) | 28 | 26 | 44 | 41 |
| 6 steps | G (= A, = A-nocue) | 46 | 44 | 48 | 40 |
| 6 steps | G-lookahead | 47 | 45 | 48 | 40 |
| 6 steps | Dir | 48 | 47 | 51 | 45 |

Outputs: `runs/2026-11-06-pooled-dev120/summary/heldout_b3.txt` and `heldout_b3_caps1_6.txt` (from `heldout_table.py`
there); disagreements from `discord.py`.

- **Errors.** No false removal in any row (0 of 120 present-truth episodes per cell). One false present per tabletop
  lighting for every policy except R (TD5017, above).
- **Unknown present-truth episodes.** 6 / 120 per cell at every travel setting except the 1-step cap, where F, R, G,
  G-lookahead and A-nocue leave 15 and A and Dir leave 6.
- **Effort, uncapped, mean travel per episode** (shelf L1 / L2, tabletop L1 / L2): G 145 / 146 / 68 / 69 cm; A
  156 / 158 / 71 / 74 cm; Dir 94 / 96 / 36 / 37 cm. Mean views are 2.12-2.26 for each of them.
- **A against G at the primary setting.** The two disagree on 5 episodes in 4 arrangements, out of 240 removed
  episodes:
  - A removes and G stays unknown: SD1024-removed-L2 (white_box) and SD1047-removed-L1 and -L2 (M16-R);
  - G removes and A stays unknown: TD5006-removed-L2 and TD5053-removed-L2 (both ycb_box).
- **What A does.** At the primary setting, A's held-out decision equals Dir's in every episode, and A-nocue's equals
  G's in every episode. Under the 3- and 4-step caps, A again equals Dir in every episode. Under the 2-step cap A
  fails like G, because it goes to a far stop first. Under the 6-step cap A equals G. Dir is never below A in removals
  at any travel setting.

## Resolution of the primary contrast at the test size

`runs/2026-11-05-precision/precision.py`, B 4000, seed 0 (output
`runs/2026-11-06-pooled-dev120/summary/precision_A_G.txt`). It resamples the held-out development arrangements to the
test quota of each slot class (100 per environment). This is a design property, not a prediction, and it assumes test
arrangements disagree the way development ones do.

| A - G, removal recall | Dev estimate | SD at 100 arrangements | 2.8 SD (about 80 % power, two-sided 5 %) |
|---|---|---|---|
| shelf L1 | +0.017 | 0.012 | 0.034 |
| shelf L2 | +0.033 | 0.017 | 0.049 |
| tabletop L1 | 0.000 | 0.000 (no disagreement on dev) | - |
| tabletop L2 | -0.033 | 0.017 | 0.047 |

FRR differences are 0 everywhere on dev. Coverage differences are a third of the recall differences, because coverage
counts all three conditions. A difference of the size seen on dev is about the size the test resolves. **The likely
test outcome for A - G at the primary setting is inconclusive in at least some cells.** Inconclusive is not
equivalence (SIM_PREREGISTRATION §1).

## Reading

- These are development results under the declared design. They do not change it: the endpoint, k, lam, the caps and
  A's design stay as pinned.
- Two conditional development diagnostics. A's measured advantage on development data comes from the side
  descriptor, and the deterministic rule Dir, built on the same descriptor, does at least as well at every travel
  setting. They are recorded here before any test arrangement exists. The test may differ.
