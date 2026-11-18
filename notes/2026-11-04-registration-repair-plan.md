# Registration repair: bounded development experiment (pinned 2026-11-04)

SIMULATION, development data only. Pinned after the researcher's decisions of 2026-11-04 (1a, 2a, 3a, with corrections);
the commit that adds this file is the pin, and no candidate is scored on development data before it.

Why: the 11-arrangement tabletop transfer diagnostic missed the pre-set go/no-go "ECC refusals < 5 %" (L1 12/165, L2
9/165; `runs/2026-11-04-tabletop-transfer11/regdiag/regdiag.md`), and the researcher's review found a direction bug in
the 2026-11-03 block-rejection prototype (`notes/2026-11-03-evidence-v03-align.md`, correction). Unchanged v0.3
(`perception/evidence_v03.py`, 0.3.1-dev, `image_align`) is the baseline C0. The criterion, the shelf v03_image
thresholds (tau_q 0.3, tau_occ 0.35, tau_match 0.65; frozen_candidate.json sha256 4a06de0e...) and the frozen v0.2.1
file (sha256 d0e0927f...) do not change. Nothing is recalibrated. All criteria below are DEVELOPMENT criteria and
practical tolerances, not statistical confidence statements and not validated physical accuracy limits.

## What the diagnosis says a candidate must do

- 10 of 21 refusals: the occluder's edges make even the TRUE warp score cc < 0.5 (TD5006). Rejecting those pixels from
  the fit and from the cc check is needed; a better start is not (two are already within 0.8 px of the truth).
- 11 of 21: the truth scores >= 0.5 but ECC converged 120-440 px away (TD5000, TD5004). A rejection pass that starts
  from that warp may not recover it, so starting points are a separate factor (M). The groups overlap: 8 of the 10
  low-cc views are also badly fitted.
- Repetitive grain: no evidence. Parallax: not the refusal cause, but a single homography is pulled up to ~20 px off
  the target-centre plane on 45 accepted tabletop views (silhouette IoU median 93.8 % vs 97.6 % for the true plane,
  researcher's audit `runs/2026-11-04-registration-deep-research/local_audit.json`).

## Common definitions

- Images, pyramid, mask: v0.3's (`_edges` with sigma 1.5; fit mask = frame minus a 24 px border minus grow(box, 0.5);
  levels 4 .. 1, warps scaled by `_scaled`; `_pyr_down_mask`). "Level 1" is the finest fitted level (x1/2).
- Convention: W maps QUERY pixels to REFERENCE pixels. The query in reference coordinates is
  `cv2.warpPerspective(Q, W, size)` (no WARP_INVERSE_MAP). Pinned by
  `runs/2026-11-04-block-rejection-repair/test_direction.py`; the candidate module carries the same test.
- Pass 1: v0.3's coarse-to-fine ECC (homography, levels 4 .. 1, 200 iterations, eps 1e-6) from the given start (C1's
  is the prototype's `ecc_global` pass 1, numerically the same procedure; a test checks the two give the same warp on
  synthetic pairs), with NO refusal check applied to it: its warp is kept for the repair whatever its cc, shift or scale. A cv2.error in pass
  1 refuses the view (there is no warp to repair).
- Blocks: 16 x 16 px at level 1 (= 32 px at full resolution), on the grid anchored at (0, 0); a block is usable if at
  least 25 % of its pixels are in the level-1 fit mask and valid under the warp being scored.
- Final checks, applied ONCE, to the final (post-repair) warp only: (i) v0.3's checks unchanged: cc >= 0.5, box
  centre shift <= 100 px, local scale at the box within 1 +- 0.1 (cc as defined per candidate below); (ii) support
  guard: the support kept by the repair is >= 50 % of the pass-1 level-1 fit-mask pixels, and >= 6 of the 9 cells of
  a 3 x 3 grid over the level-1 frame keep >= 25 % of their pass-1 fit-mask pixels (C2: weighted, see C2). A view
  failing any check is refused (identity, align_ok False, quality 0, as v0.3). A cv2.error in a repair pass refuses
  the view; there is no fallback to the pass-1 warp.
- Only the warp leaves the aligner (as v0.3: `evidence_for` receives H). Pixels left out of, or down-weighted in, the
  fit still enter v0.2.1's quality and match and v0.3's change and obstruction computations unchanged. A test feeds the
  same H to the evidence step with and without a fit exclusion and requires identical records.
- Image inputs only at run time: the reference, its box and the query. No mask, label, pose or truth.

## Candidates (fixed; no parameter is swept)

**C0**: v0.3 `image_align`, unchanged.

**C1**: exactly the direction-corrected prototype: `runs/2026-11-04-block-rejection-repair/align_lib.py`
`ecc_global(ref, qry, box, "homography", "edges", robust=True)` (levels 4, stop_level 1, excl_f 0.5, sigma 1.5,
border 24): pass 1 from identity; then `block_inliers` with block 16 at level 1, residual = |z(reference edges) - z(query
edges warped by W)| with z-scores over the valid fit mask, a block drops if its median residual > median + 4 x max(MAD,
1e-3) (MAD = 1.4826 x median absolute deviation over usable blocks); then one ECC refit at level 1 from the pass-1 warp on
the retained mask. cc for the final check = that refit's cc (on the retained mask). Its relative threshold drops
textured blocks of correct fits (test_direction.py, known limit); that is part of C1 and is not changed after results.
The C1 code moves into the candidate module byte-for-byte in its numerics (a test compares it with the folder copy).

**C1+M**: C1 run from each of two starts (the only change to C1's code: `ecc_global` takes the start warp, scaled to
level 4, instead of identity), one winner per view:
- S0 identity;
- S1 a translation from phase correlation of the level-4 edge images, each multiplied by its level-4 fit mask and by
  a Hann window (`cv2.createHanningWindow`), `cv2.phaseCorrelate`; the shift t is taken so that query(x) ~ ref(x + t)
  (sign fixed by a unit test with a known shift), and W0 = [[1, 0, 16 t_x], [0, 1, 16 t_y], [0, 0, 1]] at full scale.
  If |16 t| > 100 px, S1 is not run (S0 only).
Ranking: only starts whose final warp passes every final check are eligible; none eligible refuses the view (reasons
from S0 recorded). Each eligible warp gets the VALIDATION SCORE on COMMON support in reference coordinates:
- common blocks = the level-1 blocks of the pass-1 S0 fit mask (reference coordinates) that are usable under EVERY
  eligible warp and whose reference edge block has std > 1e-6;
- block score = zero-mean normalised cross-correlation (ZNCC) of the reference edge block and the query edge block
  warped into reference coordinates by that warp (bilinear);
- validation score = the MEDIAN block score over the common blocks (robust to a minority of occluder blocks).
Validity: if the common blocks are fewer than 50 % of the S0 pass-1 usable blocks, or cover fewer than 6 of the 9 cells
(a cell counts with >= 25 % of its S0 usable blocks), the scores are not comparable and S0 wins if eligible, else S1.
Tie-break: S1 wins only if its score exceeds S0's by more than 0.01; otherwise S0. Each view records both scores, the
common-block count and the winner.

**C2** (scored only if C1 and C1+M both fail): R-ECC-style block reweighting. Following Evangelidis, Diego & Horaud
(2013), sect. 3.2, pairwise alignment only, in our own pinned form (not claimed to reproduce their exact estimator):
- pass 1 as above (unweighted, cv2), from identity;
- then K = 3 reweighting rounds, each: (a) per usable block b (blocks as defined, warp = the current one), residual
  r_b = 1 - ZNCC_b of the reference edge block and the warped query edge block; (b) robust location m = median(r_b) and
  scale s = max(1.4826 x median(|r_b - m|), 0.02); (c) weight w_b = 1 / (1 + (max(0, r_b - m) / (2.385 s))^2) (Cauchy,
  only residuals worse than the median are down-weighted, so w_b <= 1 and the median block has weight 1; no further
  normalisation); unusable blocks w_b = 0; each pixel takes its block's weight, times the level-1 fit mask; (d) a
  weighted ECC refit at level 1 from the current warp: the ECC objective of Evangelidis & Psarakis (2008) with every
  inner product, mean and norm taken with the pixel weights, homography, Gauss-Newton updates as in the paper, at most
  200 iterations, stop when the update's max abs parameter change < 1e-6; implemented in numpy and validated before
  any scoring: with all weights 1 it must match cv2.findTransformECC on synthetic warped pairs to <= 0.05 px at the
  box corners;
- cc for the final check = the weighted ZNCC at the final warp;
- weighted support guard (replaces the binary one for C2): effective support N_eff = (sum w)^2 / sum w^2 over level-1
  pixels >= 50 % of the pass-1 level-1 fit-mask pixel count, and >= 6 of the 9 cells with weight mass sum w >= 25 % of
  their pass-1 fit-mask pixel count;
- C2+M: C2 from S0 and S1 with the C1+M ranking rule unchanged.

## Evaluation (both environments, existing thresholds)

Sets (both already-examined development data): shelf_dev_v3 random arrangements (1800 views, 900 per lighting; C0
refused 0) and tabletop_transfer11 (330 views, 165 per lighting). The shelf design cases (120 views) are run and
reported separately and do not gate. Fixed, complete denominators: every job of the set's jobs_refL1.json is a view;
a refused view, an ECC error or a support-guard refusal counts as refused and is scored by the evidence step as v0.3
scores a refusal (identity, quality 0). Per environment and lighting, against C0: refusals; views newly refused and
recovered; box-corner error vs the true planes (shelf: target, front; tabletop: target-centre, table), grid error over
the fit region, target-alone silhouette IoU (`target_alone_P*` in the reference and `target_alone_revisit_P*` in the
query, query mask warped into reference coordinates by the warp, nearest neighbour, IoU); present-labelled clear misses,
removed-labelled strong matches, diagnostic removal yield (clear misses on removed diagnostic views), gate pass on
present views; run time per view. Quantiles: numpy default (linear) on the stated view set.

## Pass criteria (each per environment AND per lighting; all must hold)

1. Refusals < 5 %: tabletop <= 8 / 165, shelf <= 44 / 900.
2. Present-labelled clear misses: 0 (C0: 0).
3. Removed-labelled strong matches: no more than C0.
4. Diagnostic removal yield: no lower than C0.
5. Geometry, common-accepted views (accepted by BOTH C0 and the candidate; both summaries computed on exactly that view
   set), box-corner error vs the true target plane (tabletop: the target-centre plane, an approximation used only for
   this non-regression comparison): candidate median <= C0 median + 0.5 px, p90 <= C0 p90 + 1 px, max <= C0 max + 2 px.
6. Geometry, newly accepted views (refused by C0, accepted by the candidate): each view's box-corner error vs the true
   target plane <= the maximum over C0-accepted views (same environment and lighting) + 2 px.
7. ~~Run time <= 3 x C0 per view (median).~~ Replaced by mandatory run-time reporting: see Amendment 1.
Silhouette IoU, the table / front plane errors and the grid error are reported alongside 5 and 6 and do not gate.

## Order, stopping and lock

C1, then C1+M (a separate comparison: does the start help, at the same pass criteria). If neither passes: C2, then
C2+M. If none passes, stop and bring back an explicit limitation or design decision; the search is not widened. The
first candidate in this order that passes is LOCKED (module committed as 0.3.2-dev with its sha256 recorded, and the
evidence specs pointing at it) before the remaining 49 tabletop development arrangements are rendered and scored, which
are then a subsequent development check of the locked candidate. The test split stays untouched until calibration
and FREEZE.md.

## Step 1 (before any candidate): the historical comparison rerun

`runs/2026-11-04-block-rejection-repair/run.sh`: shelf_dev_v1_check, scored against the tag homography exactly as on
2026-11-03, three arms on one code root and environment: the preserved prototype unchanged (control), the same with
only the direction corrected, plain global ECC. All outputs kept. If the control does not reproduce 22 / 150 (L1) and
33 / 150 (L2) views over 10 px, that is investigated before any difference is attributed to the fix. This rerun
characterises C1; it does not change C1's definition.

## Pinned files (untracked run folder; sha256 at the pin)

- `runs/2026-11-04-block-rejection-repair/align_lib.py` (C1 source): 689ad6d1971561beee8fd89b85a061d35076e3bef652918df98183aad194ebf6
- `runs/2026-11-04-block-rejection-repair/test_direction.py`: 4da62e87cc297e55e82169c03ae72cf395d4e447b03f03d3ad102d31c3ad6ad3
- `runs/2026-11-04-block-rejection-repair/run.sh` (step 1): c932e40af92386bd73391cddcd5dae684afcb1ee59adeff3f67d9b88040e48db
- preserved prototype `runs/2026-11-03-evidence-v03-align/scripts/align_lib.py` (main checkout): a8e021f0800eeb1c6b2f52ed1f6adfe0055d74b0cb19f339219a777e5e21263e

## Amendment 1 (2026-11-04, researcher's choice (c)): criterion 7 replaced by mandatory run-time reporting

POST-HOC development amendment, made AFTER inspecting C1's historical registration accuracy and timings (step 1,
`runs/2026-11-04-block-rejection-repair/RESULT.md`: on shelf_dev_v1_check, Jetson, 6 worker processes, cv2 one thread
per process, the block-rejection arm took a median 23.96 s per view against 2.32 s for plain global ECC, about 10x),
and BEFORE the planned full-dev candidate EVIDENCE evaluation. It does not precede all candidate scoring: C1's
historical fitting performance had already been examined. The original criterion 7 (run time <= 3 x C0, median)
stays above, struck through, and in the history of this file (pin commit b144e04).

Why: as drafted, criterion 7 would have failed C1 and C1+M on timing alone, and probably C2; the 2026-11-03 summary
already showed 24 s against 2.6 s per view when the 3x budget was proposed (a drafting error, not a result-driven
change of the accuracy or evidence criteria).

Replacement (reported, never gating): per environment and lighting, for C0 and each scored candidate, the median, p90
and maximum of (i) registration latency (the aligner call alone) and (ii) complete evidence latency (one view's whole
evidence step, registration included), with the candidate / C0 ratio of each; and (iii) parallel batch throughput
(views per hour of wall clock for the whole job). C0 and the candidate are measured on the GB10 as comparable jobs:
the same module and harness (`perception/registration_v032.py`, C0 = v0.3 `image_align` unchanged), the same jobs, the
same number of worker processes, and the recorded thread settings (cv2, torch, OMP). The Jetson step-1 timings stay
reported with their actual conditions. Slow execution remains a disclosed limitation: removing the gate does not
establish practical online speed.

Unchanged: criteria 1-6 (refusals per environment and lighting; present-labelled clear misses and removed-labelled
strong matches; diagnostic removal yield; common-accepted and newly accepted geometry), the candidate definitions and
order, the evidence thresholds and every other setting. The remaining 49 tabletop arrangements still wait until a
candidate passes the retained criteria and is locked.

## Amendment 2 (2026-11-05, researcher's choice (a)): the registration search is closed; development continues with unchanged C0

POST-HOC development amendment, made AFTER the bounded candidate experiment above had failed (no candidate passes the
retained criteria in both environments; `notes/2026-11-05-registration-repair-results.md`, commit 61cb0e0) and AFTER the
saved-evidence headroom analyses (main checkout `runs/2026-11-05-research-direction/` and
`runs/2026-11-05-publication-plan/`; this worktree `runs/2026-11-05-headroom-check/`), and BEFORE any of the remaining
49 tabletop development arrangements (dev60 slots 11-59: TD5011-TD5050, TD6200, TD5052-TD5059) is rendered or scored.

Decision. The registration search is closed: no further candidate is defined, run or scored. Development continues
with C0, unchanged, in both environments. This explicitly permits development to proceed DESPITE the unmet go/no-go
requirement: "Order, stopping and lock" and the last sentence of Amendment 1 held the 49 until a candidate passed and
was locked; that hold is released for C0 only. It is the researcher's decision, recorded before the 49 exist as images.

What stands. Criterion 1 (refusals < 5 %) is not changed or relaxed. C0 FAILS it on the tabletop: 12 / 165 views refused
in L1 and 9 / 165 in L2 on the first 11 arrangements (`runs/2026-11-04-tabletop-transfer11/regdiag/regdiag.md`); on the
shelf it refused 0 / 900 in each lighting. Neither C0 nor the repair experiment passed, and nothing in this file may be
cited as a passing registration.

C0, pinned for both environments:
- `perception/evidence_v03.py` (EVIDENCE_VERSION 0.3.1-dev), sha256
  eeae5273db57647735dc8cc35f25f7dcc2c6a740b69c7d4c0a1ea4025901fd97, run as `python -m perception.evidence_v03 --align
  image`: `image_align` with the default AlignParams (sigma 1.5, levels 4, stop_level 1, excl_f 0.5, border 24,
  iters 200, eps 1e-6, min_cc 0.5, max_shift_px 100, max_scale_dev 0.10). The same bytes are at the shelf evidence
  commit 694f46c, the tabletop transfer commit e0dc134 and this commit.
- Its imports: frozen `perception/evidence.py` d0e0927ff9d36ee2a8027cc8050f1801e372faea1e12cd98c513345d258d46cf,
  `verify/schema.py` fc34832788f1e1b947f32c663b789b20e5449b9095cf2065f241ac2cbb739b8f, `verify/stopping.py`
  35575caea0c62a56dee08881d9b3add72c2182733f9a6e50487b344ca91a4765. Environment: `uv.lock`
  6f6743a67be3f88867000cda6cdbc351463f6ecbb6e4e5a599ef6b7e5e14d053, `pyproject.toml`
  245a585041f832a3c34f335b0bf9cd5427568c4950497daf29b6529bc6c7ee5d.
- Evidence of record so far, under `/mnt/bench-data/datasets/cmpe249-active-verification/sim/`:
  - shelf: `shelf_dev_v3/evidence/v03_image_refL1.jsonl` 860b9da4568d4ef7b226a39b5bab4a6de70e7bbb9fec4f23745e184d45a008f0
    (diag 376b15443c2f4833367053bd26a308099a3aaca01749fdac88f4bcceb0ae6fb1; `provenance_694f46ca67aa.json`
    5e3c6d0df687e9e224eb33da46106e4500dcc2815a7796b2ee52f907ee2df568);
  - tabletop: `tabletop_transfer11/evidence/evidence_tabletop_transfer11_evidence_e0dc13499670_790e41d78b/v03_image_refL1.jsonl`
    ca386f0489c752b1acdbe00ca3957f6eb8a2b7c39a7d13959cb77fb02d602bd3 (diag
    2117539d3a4ac6287617a4235f0db75ddae1e4d73ec51fbb899c088d023e4188; `provenance.json`
    2b45266e756161e2892b0142ae250618ff56ded59a9b6f5354b88078ffe36838).
- The harness `perception/registration_v032.py --candidate C0` (sha256
  8a1e224509211e65e38d5af13775796d4c8b007ea1666024f788906a6bb44371) reproduced both sets exactly (largest difference 0,
  equal refusals) and also records the warp H. It may run beside the evidence of record for evaluation-only geometry;
  the evidence of record stays the `perception.evidence_v03` output.

What a refused registration does (this corrects the earlier, overstated wording). A refused view gets the identity
warp, align_ok false and quality 0, so under the current gate (quality >= tau_q = 0.3) it can never contribute a clear
miss. Its match score and obstruction are still computed on the unregistered query, and the strong-match rule has no
quality gate, so a refused view CAN still return PRESENT. It also uses one view of the budget, and its obstruction value
can change G's occlusion references and so the later choice of views. Refusal is not abstention and does not protect
against every error.

Thresholds. The 49 are first scored as a TRANSFER DIAGNOSTIC with the fixed shelf v03_image thresholds
(frozen_candidate.json sha256 4a06de0eebe407aad27b58fdfce3751460b5357924aba93a6f6830095b93b983: tau_q 0.3, tau_occ 0.35,
tau_match 0.65, k 2, lam 0), as the first 11 were. That diagnostic stays separate from the planned final shared
calibration, which is unchanged: one global threshold set (SIM_PREREGISTRATION §7) calibrated on both environments'
development data (`verify/select.py`, pooled sets), with the existing joint grid, both alpha constraints and complete
grouped cross-validation (a candidate is eligible only if calibration succeeds on all development data and in every
planned fold). This amendment locks registration; it does not freeze thresholds.

The 49 and the four tabletop design cases. They are rendered after this commit and run through C0 unchanged (the
evidence of record, plus the harness C0 for geometry). TDES0-TDES3, the predeclared tabletop design cases outside the
60, are rendered in the same batch and reported separately. Until A's design is pinned (a commit recording A's
features, training label, model family, regularisation grid and selection rule, and the direction-rule control), only
operational integrity outputs are opened: completion and exit codes, file and row counts, render timings and retries,
the frame-check and schema results, and hashes. Scene content (images, masks, per-arrangement content) and performance
outputs (evidence scores, refusals, registration geometry, episode decisions) stay closed. Once opened for calibration
or training, the 49 are development data, not an independent test set.

Reporting. By environment and lighting, keeping the first 11 arrangements, the remaining 49 and the final test apart:
refusals; both error directions (per view: present-labelled clear misses and removed-labelled strong matches; per
episode: false removals and false presents); unknowns; and the geometry of accepted views, reported as disagreement
with the specified simulator plane (shelf: the target plane, the front plane beside it; tabletop: the horizontal plane
through the target's centre, a proxy, the table plane z = 0 beside it), with target-alone silhouette IoU where those
renders exist. A plane disagreement is not an error against a true whole-scene alignment.

Headroom in the preregistration package. The package includes the 2026-11-05 headroom findings, labelled CONDITIONAL
DEVELOPMENT DIAGNOSTICS: they hold for the current development evidence and thresholds, and the any-order ceilings use
views no policy may see. On shelf dev at b = 3 uncapped, G removes 45 / 60 (L1) and 43 / 60 (L2) against any-order
ceilings of 46 and 44, and present episodes are at their ceiling; on the first 11 tabletop arrangements G is at its
ceiling; G's choices equal the fixed route P3, P1, P5, P4, P2 on all 360 shelf episodes; and under the two-step travel
cap the any-order ceiling (25 and 24) exceeds the best one-side route (15 and 13), with 18 of those ceiling removals in
the M16 arrangements, each resolvable only on the side away from its designed board offset
(`runs/2026-11-05-headroom-check/side_audit.txt`). None of this predicts the final comparison: A may underperform G,
and the planned calibration or new data may change the ceilings.

Uncertainty stays arrangement-based: the paired, family-stratified cluster bootstrap of SIM_PREREGISTRATION §8.
Binomial bounds on episode counts are descriptive only; the independent-binomial "about 3 %" figure discussed on
2026-11-05 is not a risk bound of this study.

Unchanged: k = 2, lam = 0, A - G at b = 3 with uncapped travel as the primary comparison, the declared secondary and
sensitivity analyses, and the freeze before any test scene (no seed >= 100000 is generated, rendered or scored before
FREEZE.md). No further registration candidates, no new environment and no endpoint change.
