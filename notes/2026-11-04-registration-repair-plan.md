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
7. Run time <= 3 x C0 per view (median).
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
