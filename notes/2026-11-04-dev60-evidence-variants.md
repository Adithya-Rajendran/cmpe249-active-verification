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

**Reading:**
- Image alignment alone was not enough: the v0.2.1 cues on aligned images are infeasible.
- Under identical alignment, the background cue improves the result. It removes the fully hidden false clears, at the cost of some diagnostic removal views and some clear present views in L2.
- This isolates the cue's added benefit. It does not show that alignment is unimportant: without registration, frozen v0.2.1 refuses every view.

**v0.3 held out** (thresholds from the leave-one-arrangement-out folds; uncapped; b = 3; primary k = 2):

| | L1 | L2 |
|---|---|---|
| Per-view false-clear rate | 12/176 (0.068, upper 95 % 0.108) | 9/176 (0.051, upper 0.088) |
| Correct clear misses on removed diagnostic views | 115/124 | 112/124 |

"False present" means a removed target answered "present". That is a wrong answer, but FRR, recall and coverage do not show it.

| Policy | L1 FRR | L1 recall | L1 false present | L1 coverage | L2 FRR | L2 recall | L2 false present | L2 coverage |
|---|---|---|---|---|---|---|---|---|
| S | 0/120 | 0/60 | 0/60 | 0.333 | 0/120 | 0/60 | 0/60 | 0.333 |
| F | 0/120 | 24/60 | 0/60 | 0.767 | 0/120 | 24/60 | 1/60 | 0.772 |
| R | 0/120 | 34.15/60 | 0/60 | 0.823 | 0/120 | 33/60 | 0.75/60 | 0.821 |
| G | 0/120 | 45/60 | 0/60 | 0.883 | 0/120 | 43/60 | 1/60 | 0.878 |
| FULL | 0/120 | 46/60 | 0/60 | 0.889 | 0/120 | 44/60 | 1/60 | 0.883 |

**v0.3 confusion counts for G** (truth → decision; present episodes 120 per lighting, removed 60):

| | L1 | L2 |
|---|---|---|
| present → present / removed / unknown | 114 / 0 / 6 | 114 / 0 / 6 |
| removed → removed / present / unknown | 45 / 0 / 15 | 43 / 1 / 16 |
| accuracy among resolved episodes | 159/159 | 157/158 |

`summary.md` has these counts for every policy and variant. The summary now always reports false present and the confusion counts alongside coverage.

**The one wrong "present" (SD1024-removed-L2) is a look-alike behind the target, not a threshold quirk.** Every policy except S answers "present" there; R does so in 0.75 of its seeds.
- P1 is a diagnostic view (the occluder covers 0 of the region). Removing the tomato-soup can reveals a SPAM can that stood directly behind it and fills the box.
- DINOv2 match against the reference soup can is 0.651 in L2 and 0.649 in L1; that fold's τ_match is 0.65.
- In L2 the strong-match rule answers "present", and that rule does not consult the occlusion gate. In L1 the episode stays unknown.
- **This is a newly recorded limit**, now added to `perception/README.md` ("Known limits"). It is related to the recorded "Loose boxes keep removed targets present", because both come from comparing the whole box. The mechanism is different: here a similar-looking object behind the target fills the box once the target is gone.
- One example does not justify moving τ_match. It stays a documented failure mode for the error analysis.

- The `-fcm` rows in the summary use the first-clear-miss rule (k = 1). They are a reported sensitivity, not the primary rule.
- The **frozen v0.2.1** candidate is eligible only because nothing passes the gate, and it is worse than useless:
  - held-out removal recall is 0/60 in both lightings;
  - its coverage of about 0.79 includes **wrong "present" answers on removed targets**: F 29/60 (L1) and 28/60 (L2), G 27/60 in each lighting, FULL 29/60 and 28/60. They come from strong matches at τ_match 0.4.
  - Its select-v4 eligibility is degenerate. Report it only as a record of the tag dependence.

**DES cases** (v0.3 thresholds):
- no hidden stop and no DES3 stop passes the gate;
- every clear stop passes;
- removed stops pass where the occluder does not cover the region.
The full table is in `summary.md`.

## Open for checkpoint 3

- The v0.2.1 quality (sharpness) term is sensitive to resampling: up to 2 px of registration difference moved it by more than 0.1 on 16 of 360 views of the check set. This has not been measured on the full dev set, which needs the true-alignment diagnostic run (evaluation only).
- The v0.3 held-out per-view false-clear rate is 12/176 in L1, with a binomial upper bound of 0.108.
  - **This is not a freeze failure.** The freeze rule (docs/SIM_PREREGISTRATION.md, §7) requires the empirical training constraints to hold and calibration to succeed in every fold. An independent review audit (`runs/2026-11-03-full-dev-review/audit.json`, main checkout) confirmed that all 60 v0.3 training folds meet both constraints. Held-out bounds are reported checks.
  - The views share arrangements, so the binomial bound is descriptive. Arrangement-level uncertainty stays the primary analysis.
- **Little removal-recall headroom for A on this dev set** (from the same review audit). An ordering that knew every view's scores in advance reaches at most 46/60 (L1) and 44/60 (L2), against G's 45/60 and 43/60. SD1047 is the only removed episode in each lighting that is achievable but missed by G.
  - That is conditional on these scores, not a prediction of test performance.
  - A may still save views or travel. Keep the declared primary comparison and report the planned secondary budget curves. An A–G tie is a valid result.
