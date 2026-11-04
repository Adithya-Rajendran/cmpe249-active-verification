# Evidence variants on the full shelf dev set (2026-11-04)

**Simulation only, development data.** 60 random dev arrangements of shelf_dev_v3 (tag-free) plus DES0–DES3, which are reported separately and never calibrated on. Nothing here is a test result.

## What ran

- **Driver:** `tools/pipeline.py configs/pipeline/shelf_dev_v3_evidence.json --from-stage evidence`, with zero alerts from the evidence stage.
- **Evidence:** on the GB10, from `git archive` of 694f46c. Provenance is in `runs/shelf_dev_v3_evidence/evidence_provenance.json`.
  - Every output has one record per job: 1800 random-arrangement views and 120 DES views.
  - The outputs are in `/mnt/bench-data/datasets/cmpe249-active-verification/sim/shelf_dev_v3/evidence/`.
  - Step times on the random-arrangement views: frozen v0.2.1 225 s, v0.3 2013 s (about 1.1 s per view, CPU-bound ECC).
- **select-v4:** the same α = 0.10, grid and 60 leave-one-arrangement-out folds for every variant. The local runs are in `runs/shelf_dev_v3_evidence/select/<variant>/`.
- **Summary:** `runs/shelf_dev_v3_evidence/summary.md` and `summary.json`, written by `tools/evidence_summary.py`.

| Variant | Evidence | select-v4 |
|---|---|---|
| `v021_frozen` | `perception/evidence.py` unchanged (tag registration) | eligible, τ_q 0.8 / τ_occ 0.1 / τ_match 0.4, but the gate rejects every view |
| `v021_image` | v0.2.1 match, quality and obstruction on v0.3's image registration (ablation, same pass as v0.3) | **infeasible**: best false-clear rate 0.341 (120/352) at τ_q 0.8, τ_occ 0.1 |
| `v03_image` | evidence v0.3 (background cue + image registration) | eligible, τ_q 0.3 / τ_occ 0.35 / τ_match 0.65; dev false-clear rate 0.057 (20/352), false clear-miss rate 0 (0/848) |

## Per lighting

**Registration refusals:**
- frozen v0.2.1: 900/900 in L1 and 900/900 in L2 (no tags);
- both image-aligned variants: 0/900 in each lighting.

**Gate pass at the illustrative setting** (τ_q 0.6, τ_occ 0.3, the same for both variants), occluded views by class:

| Class | v0.2.1 cues, image-aligned (L1 / L2) | v0.3 (L1 / L2) |
|---|---|---|
| Fully hidden | 53/97 / 50/97 | 0/97 / 0/97 |
| Partial | 46/79 / 44/79 | 8/79 / 7/79 |
| Removed, diagnostic | 124/124 / 122/124 | 113/124 / 107/124 |
| Present (visible condition) | 300/300 / 300/300 | 300/300 / 297/300 |

With identical alignment, the background cue alone is what removes the fully hidden false clears. It costs some diagnostic removal views and some clear present views in L2.

**v0.3 held out** (thresholds from the leave-one-arrangement-out folds; uncapped; b = 3; primary k = 2):

| | L1 | L2 |
|---|---|---|
| Per-view false-clear rate | 12/176 (0.068, upper 95 % 0.108) | 9/176 (0.051, upper 0.088) |
| Correct clear misses on removed diagnostic views | 115/124 | 112/124 |

| Policy | L1 FRR | L1 removal recall | L1 coverage | L2 FRR | L2 removal recall | L2 coverage |
|---|---|---|---|---|---|---|
| S | 0/120 | 0/60 | 0.333 | 0/120 | 0/60 | 0.333 |
| F | 0/120 | 24/60 | 0.767 | 0/120 | 24/60 | 0.772 |
| R | 0/120 | 34.15/60 | 0.823 | 0/120 | 33/60 | 0.821 |
| G | 0/120 | 45/60 | 0.883 | 0/120 | 43/60 | 0.878 |
| FULL | 0/120 | 46/60 | 0.889 | 0/120 | 44/60 | 0.883 |

- The `-fcm` rows in the summary use the first-clear-miss rule (k = 1). They are a reported sensitivity, not the primary rule.
- The **frozen v0.2.1** candidate is eligible only because nothing passes the gate. Its held-out removal recall is 0/60 in both lightings, and its coverage (about 0.79) comes from present decisions only. That eligibility is degenerate, not useful performance.

**DES cases** (v0.3 thresholds):
- no hidden stop and no DES3 stop passes the gate;
- every clear stop passes;
- removed stops pass where the occluder does not cover the region.
The full table is in `summary.md`.

## Open for checkpoint 3

- The v0.2.1 quality (sharpness) term is sensitive to resampling: up to 2 px of registration difference moved it by more than 0.1 on 16 of 360 views of the check set. This has not been measured on the full dev set, which needs the true-alignment diagnostic run (evaluation only).
- The v0.3 false-clear rate is 0.057 on the dev data, with a held-out per-view upper bound of 0.108 in L1. That bound sits just above α in L1, so dev data gives little margin. These are descriptive dev numbers, not test claims.
