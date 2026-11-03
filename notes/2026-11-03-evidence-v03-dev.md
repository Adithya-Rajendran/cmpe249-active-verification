# Evidence v0.3 on dev data: an occlusion cue for plain occluders (2026-11-03)

**Simulation only, development data, nothing frozen.** Everything here comes from the Isaac Sim check batch (6 random arrangements and 4 design cases, 300 query views). Every choice below was made while looking at these views. They are the data the cue was built on, so they are not a test of it. The 60 dev arrangements are the real development set.

**Inputs.**
- The check batch: `/mnt/bench-data/datasets/cmpe249-active-verification/sim/shelf_dev_v1_check/` (called `ROOT` below). The primary jobs are `jobs_refL1.json` (random) and `jobs_design_refL1.json` (design), and the frozen v0.2.1 outputs are `evidence_refL1{,.diag}.jsonl` and `evidence_design_refL1{,.diag}.jsonl`.
- Evidence paths are written as `RUN/...`, where `RUN` is `runs/2026-11-03-evidence-v03/` in the `cp3-evidence` worktree on the Jetson. It is gitignored and local only.
- The analysis scripts are in `RUN/scripts/`. Labels are joined only for scoring, in `RUN/scripts/load.py`.

**Code.**
- `perception/evidence_v03.py` (sha256 `a47d05f4…`): v0.2.1 plus one cue. It imports the frozen `perception/evidence.py` (sha256 `d0e0927f…`) read-only and never edits it.
- `tests/test_evidence_v03.py`: 17 synthetic tests. The full suite passes: 550 tests, 10 skipped (`RUN/test_suite.txt`).

## 1. Diagnosis: what v0.2.1 misses, and why

These numbers come from the frozen diag files plus a CPU recomputation of the gain and change parts (`RUN/scripts/features.py` → `RUN/diag/features.jsonl`). My recomputed box change share matches v0.2.1's `changed` exactly: max |diff| = 0 (`RUN/diag/features_by_class.txt`).

**Which component fails on the hidden views.** There are 42 hidden non-diagnostic occluded views (visible fraction ≤ 0.05; 22 random, 20 design). In the table, "caught" means the cue is > 0.10, the lowest τ_occ on the grid (`RUN/v03/diag_hidden_by_family.txt`).

| family | L | n | detector cue | DINOv2 novelty | v0.2.1 obstruction | v0.3 background cue |
|---|---|---|---|---|---|---|
| IND28 (SD1001, DES3) | L1 | 10 | 0 | 1 | 1 | 10 |
| IND28 | L2 | 10 | 0 | 0 | 0 | 10 |
| M16-L (SD1000, DES2) | L1 | 3 | 0 | 3 | 3 | 3 |
| M16-L | L2 | 3 | 1 | 1 | 2 | 3 |
| M16-R (SD1007, DES1) | L1 | 4 | 0 | 1 | 1 | 4 |
| M16-R | L2 | 4 | 0 | 1 | 1 | 4 |
| S8 (SD1002, DES0) | L1 | 2 | 0 | 2 | 2 | 2 |
| S8 | L2 | 2 | 0 | 2 | 2 | 2 |
| white box (SD1010) | L1 | 1 | 0 | 1 | 1 | 1 |
| white box | L2 | 1 | 0 | 0 | 0 | 1 |
| YCB box (SD1005) | L1 | 1 | 1 | 1 | 1 | 1 |
| YCB box | L2 | 1 | 0 | 1 | 1 | 1 |

- **The change test works. The two cues that turn change into obstruction do not.** On the hidden views, 0.80–1.00 of the box changed against the reference. But:
  - the detector fires on 2 of 42 views (a plain board is no catalogue object);
  - novelty is ≤ 0.10 on 19 of 20 IND28 views and 6 of 8 M16-R views, because the board's patches are explained by the plain shelf surfaces in the reference window.

  This confirms the suspected cause; before this, it was only inferred.
- **Novelty is still blind when the explaining set is background only.** I tried leaving the target's own patches out of the explaining set (`RUN/scripts/dino_local.py`, run on the GB10). Hidden-view novelty stays at median 0.06 (L1) and 0.00 (L2).
  - A same-location DINOv2 patch change does fire on hidden views (median 0.97–0.98).
  - It fires just as much on removed diagnostic views (0.40–0.94), so it cannot tell a removal from an occlusion.
  - Evidence: `RUN/diag/c3_dino_local.txt`. A recomputed whole-window novelty matches v0.2.1 exactly, max |diff| 0.
  - No DINOv2 variant was taken further.
- **The gain correction does not absorb the box change, but the occluder does pull it.**
  - v0.2.1's gain is the ring-mean ratio. On the hidden views it is 13–15 % off the visible-condition gain at the same stop and lighting (median), and up to 28 % off.
  - On those views the box change share is still 0.80–1.00, because a lit board differs from a dark target by far more than 25 grey levels.
  - A per-channel median over the box grown by 2.0 (search box excluded) is off by at most 0.8 % (L1) and 6.0 % (L2) on the same views, and at most 1.1 % and 6.2 % on any view class (`RUN/scripts/gains.py` → `RUN/diag/gain_estimators.txt`).
  - With the new cue, the ring-mean gain would have given almost the same numbers on this batch: hidden views ≥ 0.80 either way (`RUN/diag/c2_gain_absorption.txt`). v0.3 uses the median because it is not pulled, not because it changed a result here.
- **The S8 and YCB-box hidden views.**
  - They were caught by v0.2.1, almost always through novelty: S8 0.20–0.39, YCB box 0.27.
  - The detector caught only SD1005-L1 (0.93), via a "bottle" box covering the target.
  - Their margins over τ_occ are thin: SD1002-L2 reads 0.20, so it passes the gate at τ_occ ≥ 0.20.
  - Missed by v0.2.1 at τ_occ 0.10: 19 of 20 IND28 hidden views, 6 of 8 M16-R, the white box in L2 (0.06) and one M16-L view (DES2-L2 P3).
- **Why "the box changed" cannot be the cue on its own.** A removed target changes the box too: v0.2.1's own change share is 0.41–0.98 on the removed diagnostic views (`RUN/diag/features_by_class.txt`).
  - The question is what the box shows now.
  - After a removal it shows the background that was behind the target, which continues the background around it in the reference.
  - Behind an occluder it shows a surface in front. Here that is a board lit from the front, in front of the shaded shelf interior.
  - The montages are in `RUN/diag/m1.jpg` and `RUN/diag/m3_rowinterp.jpg`.

## 2. Candidates

Each candidate is combined with v0.2.1 by `max`. Match and quality are not touched.

- **C1, ring change (tried first, as the brief asked).** The share of the context ring outside the search box that changed under the robust gain.
  - **Rejected.** In this design the occluder is also present in the removed condition, often beside the empty spot. The removed views that can give a clear miss are exactly the ones where it stands beside the target, and it changes the ring there.
  - Removed diagnostic views: 0.05–0.37. Hidden views: 0.04–0.57 (`RUN/v03/cue_distributions.txt`).
- **C2 = v0.3, background consistency (adopted for development).** The share of the box whose pixels satisfy both of these:
  - (a) they changed against the reference: v0.2.1's test (blur σ 2, |Δ| > 25 on any channel), with the robust gain;
  - (b) they differ by > 25 from the background predicted behind the target. The prediction is made from the reference alone, by row-wise linear interpolation across the search box between the mean of the 4 pixels on each side.

  Inputs: the reference image and box, the query, and the tag registration. No mask, label, geometry or phase is used.

Every parameter, and why:

| parameter | value | why |
|---|---|---|
| change_delta, change_sigma, search_pad | 25, 2.0, 0.25 | v0.2.1's own values, reused and not re-tuned |
| gain window | box grown by 2.0, search box excluded, per-channel median of ref/query | not pulled by a board several times the target's size (numbers above) |
| background model | row-wise linear interpolation | the shelf's structure (floor / back-wall junction, lips) runs across the target horizontally |
| edge_px | 4 | averages render noise at the interpolation ends; not tuned |
| invalid query pixels | never obstruction | v0.2.1 quality already counts them (`valid`) |

Alternatives tried on these views and dropped (removed diagnostic views, n = 44; `RUN/diag/proto.jsonl`, `RUN/scripts/proto.py`):
- **cv2.inpaint (Telea).** It leaves an X-shaped artefact over the 180 × 165 px hole. Removed views read median 0.25 / 0.20 (L1 / L2), max 0.70.
- **Column-wise interpolation.** It crosses the floor / wall junction: median 0.38 / 0.37, max 0.85.
- **Row AND column (obstruction only if both say "not background").** Removed max drops only to 0.73, and two hidden views fall to 0.13 and 0.00.
- **Row-wise (kept).** Median 0.01 / 0.04. 40 of 44 are ≤ 0.18; 4 read 0.72–0.76.

The prototype and the module agree within 0.0034 (an interpolation-centre fix). The GB10 run reproduces the Jetson cue within 0.0034 / 0.0011 (`RUN/v03/consistency.txt`).

**The v0.3 run.**
- It ran on the GB10 from `~/cmpe249-sim/evidence_v03/` against the stale repo copy. Its `perception/evidence.py`, `perception/common.py`, `capture/tags.py` and `verify/schema.py` hash the same as this worktree's.
- 180 views in 38.6 s and 120 views in 26.7 s.
- Outputs: `RUN/v03/evidence_v03_refL1.jsonl` (sha256 `c8fcaa43…`) and `RUN/v03/evidence_v03_design_refL1.jsonl` (`375fd627…`), with `.diag.jsonl` files.
- A second run gave byte-identical files.
- **Match and quality are byte-identical to the frozen v0.2.1 outputs on all 300 views, and v0.2.1's obstruction is reproduced exactly** (`RUN/v03/consistency.txt`).
- v0.3 changes the obstruction of 133 of 180 random and 87 of 120 design views.

## 3. Per-view results, per lighting, with counts

The table is at the gate select-v4 chose for v0.3 (τ_q 0.8, τ_occ 0.2, τ_match 0.55), with all three cues at the same gate (`RUN/v03/eval_per_view_tq08_tm055.txt`). The same counts at τ_q 0.6 and τ_occ 0.10, 0.15, 0.20, 0.30 and 0.50 are in `RUN/v03/eval_per_view.txt`.

Missed occlusions are views that pass the gate. False obstruction is obstruction > τ_occ. Removed clear misses are diagnostic removed views that are clear misses at τ_match 0.55.

| random arrangements (select-v4's views) | L | v0.2.1 | C1 ring | v0.3 background |
|---|---|---|---|---|
| missed: hidden (vf ≤ 0.05) | L1 | 7/11 | 0/11 | 0/11 |
| | L2 | 8/11 | 0/11 | 0/11 |
| missed: partial 0.05–0.5 | L1 | 2/3 | 1/3 | 0/3 |
| | L2 | 2/3 | 1/3 | 0/3 |
| missed: partial 0.5–0.9 | L1 | 3/4 | 3/4 | 0/4 |
| | L2 | 3/4 | 3/4 | 0/4 |
| false obstruction: present diagnostic | L1 | 0/42 | 5/42 | 2/42 |
| | L2 | 0/42 | 6/42 | 4/42 |
| false obstruction: removed diagnostic | L1 | 0/12 | 5/12 | 2/12 |
| | L2 | 0/12 | 6/12 | 2/12 |
| removed clear misses | L1 | 12/12 | 7/12 | 10/12 |
| | L2 | 12/12 | 6/12 | 10/12 |

| design cases (not part of select-v4) | L | v0.2.1 | C1 ring | v0.3 background |
|---|---|---|---|---|
| missed: hidden | L1 | 7/10 | 0/10 | 0/10 |
| | L2 | 7/10 | 0/10 | 0/10 |
| false obstruction: present diagnostic | L1 | 0/30 | 6/30 | 0/30 |
| | L2 | 0/30 | 6/30 | 0/30 |
| false obstruction: removed diagnostic | L1 | 0/10 | 6/10 | 0/10 |
| | L2 | 0/10 | 6/10 | 0/10 |
| removed clear misses | L1 | 10/10 | 4/10 | 10/10 |
| | L2 | 10/10 | 4/10 | 10/10 |

**What the v0.3 false obstructions are.**
- The 4 removed diagnostic views, at 0.72–0.76, are SD1002-P1 and SD1005-P5 in both lightings (`RUN/diag/m3_rowinterp.jpg`). They are the background model's known limit:
  - in SD1002-P1, a white side panel edge is right behind and beside the target;
  - in SD1005-P5, the blue can beside the target in the reference is dragged into the prediction, and the parked YCB box now casts a shadow over the empty spot.
- The present diagnostic views flagged by v0.3 (6 of 84) are occluded-condition views with visible fraction ≥ 0.9, where the occluder covers a corner of the rectangular box (max 0.35). They are strong matches anyway, so no decision changes.

**The margin.**
- Removed diagnostic views other than those four read at most 0.18. The chosen τ_occ is 0.20.
- Partial occlusions read from 0.34 (vf 0.81) to 0.83 (vf 0.14). The false-clear constraint tolerates 3 of 36, so it holds up to τ_occ 0.45 (0.35: 1/36, 0.40 and 0.45: 3/36, 0.50: 6/36).
- The fragile end is the low one. Below 0.20, correct clear misses drop: 16/24 at 0.15 and 14/24 at 0.10 (`RUN/select/v03bg/selection_report.md`, τ_occ curve). See §5.

## 4. select-v4 (cp2 code in this worktree, f5593b7)

The command is `python -m verify.select --manifest ROOT/manifest.json --evidence <jsonl> --view-labels ROOT/view_labels.json --positions ROOT/positions_shelf_A.json --out RUN/select/<name>`. It ran on the Jetson CPU env; stdout is in each folder. Summary: `RUN/select_summary.txt`.

| | v0.2.1 (`RUN/select/v021`) | C1 ring (`RUN/select/C1ring`) | v0.3 background (`RUN/select/v03bg`) |
|---|---|---|---|
| exit | 3, infeasible (reproduces the smoke note) | 0, eligible for freeze | 0, eligible for freeze |
| grid settings meeting both constraints | 0 / 693 (best false-clear 20/36 = 0.556) | 77 / 693 | 616 / 693 |
| chosen τ_q, τ_occ, τ_match | none | 0.8, 0.1, 0.45 | 0.8, 0.2, 0.55 |
| false-clear (resubstitution) | — | 1/36 | 0/36 |
| false clear-miss (resubstitution) | — | 0/84 | 0/84 |
| correct clear misses on removed diagnostic | — | 3/24 | 20/24 |
| gate pass, removed diagnostic L1 / L2 | — | 2/12 / 1/12 | 10/12 / 10/12 |
| gate pass, present diagnostic L1 / L2 | — | 32/42 / 28/42 | 40/42 / 37/42 |
| held-out per-view clear misses on removed, L1 / L2 | — | 2/12 / 0/12 | 8/12 / 8/12 |
| held-out per-view false-clear, L1 / L2 | — | 1/18 / 0/18 | 0/18 / 0/18 |
| CV folds feasible | — | 6/6 | 6/6 (SD1007's fold picks τ_match 0.45) |

**C1 is feasible only because the gate rejects almost every view.** It has 3 of 24 correct clear misses, and its held-out recall is 1/6 (L1) and 0/6 (L2) for every policy. That is not a fix.

Held-out policy check for v0.3, at budget 3, uncapped. FRR is 0/12 for every policy in both lightings (`RUN/select/v03bg/selection_report.md`, Step 3b).

| policy | L1 recall | L1 coverage | L2 recall | L2 coverage |
|---|---|---|---|---|
| S | 0/6 | 8/18 | 0/6 | 8/18 |
| F | 1/6 | 13/18 | 1/6 | 14/18 |
| R | 1.65/6 | 13.65/18 | 1.30/6 | 14.05/18 |
| G | 2/6 | 14/18 | 2/6 | 15/18 |
| G-lookahead | 2/6 | 14/18 | 2/6 | 15/18 |
| FULL (all 5 views) | 2/6 | 14/18 | 2/6 | 15/18 |

**Why recall stops at 2/6, even for FULL** (`RUN/v03/removed_episodes_at_candidate.txt`, at the candidate thresholds on all dev views):
- SD1001 (IND28) has no diagnostic stop by design, so it reads UNKNOWN.
- SD1002 and SD1005 each lose one of their two diagnostic stops to the background-model limit above, so they reach only 1 clear-miss stop, below k = 2.
- In the held-out check, SD1007's fold picks τ_match 0.45, and SD1007's removed matches are 0.41–0.52. So only SD1000 and SD1010 reach REMOVED.
- **v0.3's own false obstruction costs 2 of 6 removed episodes per lighting on this batch.**

## 5. Recommendation and what to check on the 60

**Recommendation.**
- Carry v0.3 (v0.2.1 plus the background-consistency cue) into the 60 dev arrangements as the development candidate. Do not freeze it.
- Keep v0.2.1 as the baseline on the same images.
- Drop the ring-change cue and the DINOv2 variants.
- On this batch, v0.3 turns select-v4 from infeasible (false-clear ≥ 20/36) into feasible:
  - 0/36 false-clear;
  - 20/24 correct clear misses;
  - eligible in all six folds;
  - held-out FRR 0/12 per lighting.

**What I expect to hold on the 60.**
- The plain-board catch. On every hidden view the cue reads ≥ 0.80 in both lightings, with a large contrast margin: a front-lit board against the shaded shelf interior.
- Match and quality are unchanged by construction.
- Feasibility of the false-clear constraint is the likely outcome, because the hidden views sit far above any τ_occ on the grid.

**What could break.**
- **The removed side of τ_occ.** 40 of 44 removed diagnostic views read ≤ 0.18 against a chosen τ_occ of 0.20.
  - The values from 0.11 to 0.18 come from a vanished contact shadow and from occluder edges next to the box.
  - On 60 arrangements some will cross 0.2. τ_occ 0.15 already drops correct clear misses from 20/24 to 16/24 (`RUN/select/v03bg/selection_report.md`, τ_occ curve).
- **Backgrounds that are not continuous along rows** (a side-panel edge, another object, an occluder's shadow next to the target).
  - 2 of 6 random arrangements had one such diagnostic removed view.
  - They read 0.72–0.76 and cost those episodes' removal outright.
  - Count them on the 60. This is the cue's main failure mode, and it is in the safe direction (UNKNOWN, never a false REMOVED).
- **Boards that are not brighter than the background behind the target.** A board placed deep in the compartment, in shade, would give lower contrast. That should be measured, not assumed. Look at the hidden views' cue minimum per family and placement depth.
- **Partial occlusions.** These read 0.34–0.83 here. A board that covers a small part of the target (vf just under 0.9) will read near τ_occ.

**What to verify on the 60.** Per lighting:
- the minimum cue on hidden views;
- the distribution of removed diagnostic cue values, especially the count in 0.15–0.35;
- the count of row-discontinuity false obstructions;
- the feasible τ_occ window;
- held-out removal recall against FULL.

Also run the secondary `jobs_same.json` pairing, which was not run here.

## Caveats

- Six random arrangements and four design cases, a single render each. Views from one arrangement are correlated, and every count here is resubstitution on the data the cue was designed on, apart from the CV numbers, which hold out an arrangement but not the design choices.
- The design cases are not part of select-v4. Their per-view counts are reported separately.
- The occluders are rendered in the shelf's own colour (`configs/sim/arrange_shelf_v1.json`, `materials.white_box`, rgba 0.93/0.93/0.90). The cue relies on lighting contrast, not on colour.
- Registration is a front-plane homography. Parallax of deeper structure is absorbed by the blur, which was not tested separately.
