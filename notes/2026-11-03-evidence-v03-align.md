# Evidence v0.3: registration without tags (2026-11-03)

**Simulation only, development data, nothing frozen.** On 2026-11-03 the student removed the corner ArUco tags from every simulated scene, dev and test, because a home shelf has no fiducials. Evidence v0.3 must therefore align the query onto the reference from the images alone.

**Baseline.** Frozen v0.2.1 stays the baseline. With no tags its `tag_align` returns None, so it falls back to identity: no registration, and `align_ok` False.

**Paths.**
- `ALIGN` is `/home/jetson/Projects/SJSU/Fall26/CMPE249/cmpe249-active-verification/runs/2026-11-03-evidence-v03-align/`, in the main copy, so it survives removal of the worktree.
- Scripts are in `ALIGN/scripts/`, outputs and summaries in `ALIGN/out/`.
- Evaluation scripts ran on the GB10's CPU, from `~/cmpe249-sim/evidence_v03/align`, against a `git archive` copy of this branch in `~/cmpe249-sim/evidence_v03/cp3`. The first local-window sweeps ran on the Jetson.

## Method

`perception/evidence_v03.py::image_align(ref, query, box)` is the default of `evidence_for`, `run_jobs` and `--align image`. Its inputs are the reference image, the expected box and the query. It reads no mask, label, pose or tag.

**The fit.**
- ECC (cv2.findTransformECC) with a homography model.
- It runs on the gradient magnitude of the blurred grey images (σ 1.5, the "edges" mode of `capture/ecc.py`).
- It uses the whole frame minus a 24 px border and minus the expected box grown by 0.5.
- It runs coarse to fine from pyramid level 4 (1/16) down to level 1 (1/2), and the warp is scaled back to full resolution.
- What it locks onto is the shelf's own strong edges against the backdrop, mostly the front plane, which is what the tags gave.

**Refusal.** The fit is refused when any of these holds:
- ECC fails or there are too few usable pixels;
- the correlation is below 0.5;
- the box centre moves more than 100 px;
- the local scale at the box is off 1 by more than 10 %.

A refused fit returns None. `evidence_for` then uses identity, sets `align_ok` False and quality 0, which is v0.2.1's rule for refused tags. The reason is written to `diag["registration"]`.

**Determinism.** OpenCV is set to one thread inside `image_align`, and the previous count is restored afterwards. Without that, the warp bytes on one view (DES2-removed-L1 P1) depended on the thread count. With it, they are identical across runs and across the caller's thread setting (`ALIGN/out/determinism_check.json`, `tests/test_evidence_v03.py::TestImageAlign`).

**Parameters.**

| parameter | value | why |
|---|---|---|
| model | homography | the revisit is a 6-DoF camera move; affine leaves up to 9 px on synthetic perspective (below) |
| image | gradient magnitude, σ 1.5 | the L2 lighting change; capture/ecc.py's measured reasons for "edges" |
| region | whole frame minus border 24 and grow(box, 0.5) | local windows fail on plain scenes (below); the box holds the change we are testing for |
| pyramid | level 4 → level 1 | the revisit moves the image ≤ 40 px; full resolution costs time and changed nothing measurable here |
| iters, eps | 200, 1e-6 | capture/ecc.py defaults |
| min_cc, max_shift_px, max_scale_dev | 0.5, 100, 0.10 | gross-failure guards, set from the revisit ranges (±1 cm, ±0.5°), not tuned |

No value was swept. The alternatives below differ in structure, not in a knob.

## (a) Synthetic warps of real renders

`ALIGN/scripts/synth_align.py`. Each case warps a nominal reference render by a known homography G: translation ±30 px, rotation ±0.6°, scale ±2 %, perspective ±2e-5/px, fixed seeds. The truth is therefore exact for every pixel, with no parallax.

The four cases are:
- **L1:** the L1 reference, warped;
- **L2:** the L2 reference, warped;
- **L2+board:** L2 plus a plain board pasted over the box, shifted half a box sideways;
- **L2+gone:** L2 with the target inpainted.

The error is the maximum over the four box corners, in query px. The true box shift has median 23.3 px and max 39.8 px.

The tag regions are excluded from the fit for every method. Results for `v03_image_align`, which is the module itself (`ALIGN/out/synthmod_v{1,2}check_summary.txt`):

| set | case | n | median | p90 | max | > 2 px |
|---|---|---|---|---|---|---|
| v1 check | L1 | 50 | 0.02 | 0.03 | 0.04 | 0 |
| | L2 | 50 | 0.10 | 0.19 | 0.39 | 0 |
| | L2+board | 50 | 0.15 | 0.34 | 0.67 | 0 |
| | L2+gone | 50 | 0.10 | 0.19 | 0.39 | 0 |
| v2 check | L1 | 60 | 0.02 | 0.03 | 0.20 | 0 |
| | L2 | 60 | 0.10 | 0.20 | 0.29 | 0 |
| | L2+board | 60 | 0.15 | 0.33 | 1.94 | 0 |
| | L2+gone | 60 | 0.10 | 0.20 | 0.29 | 0 |

For comparison on the same cases (`ALIGN/out/synth_v{1,2}check_summary.txt`):
- the global affine variant has max 9.0–9.2 px;
- SIFT + RANSAC homography has L2 p90 75–144 px and max 320–2459 px;
- local translation ECC has p90 8–24 px;
- identity has median 24 px.

## (b) The tagged check batches, tags excluded from both images

`ALIGN/scripts/eval_align.py`. Each tag quad is grown ×2.2 about its centre, which covers the white card. That region is taken from the reference and from the query, and the union is left out of the fit.

The scores are against the tag homography. **That reference is a front-plane homography, so it is itself off at the target's depth.** The true target-depth error waits for (c).

There is also a functional score: the changed share of the box and the ring on visible-condition views, where nothing moved but the camera and the lighting. It uses v0.3's change test with the robust gain.

Box-corner error against the tag homography, in px, `v03_image_align` (`ALIGN/out/module_v{1,2}check_summary.txt`):

| set | L | n | refused | median | p90 | max |
|---|---|---|---|---|---|---|
| v1 check | L1 | 150 | 0 | 0.40 | 1.11 | 1.93 |
| | L2 | 150 | 0 | 0.49 | 1.22 | 1.95 |
| v2 check | L1 | 180 | 0 | 0.51 | 1.34 | 2.65 |
| | L2 | 180 | 0 | 0.57 | 1.41 | 3.77 |

For reference, identity is median 14.0–14.8 px and max 33.6 px (`ALIGN/out/align_v{1,2}check*_summary.txt`).

Changed share on visible-condition views (box max / ring p90 / ring max; box > 0.1 = number of views whose box share exceeds 0.1):

| set | L | n | identity | tag | v0.3 image |
|---|---|---|---|---|---|
| v1 | L1 | 50 | box max 0.738, box > 0.1 in 44 | 0.058 / 0.003 / 0.014 | 0.059 / 0.001 / 0.007 |
| v1 | L2 | 50 | box max 0.687, box > 0.1 in 46 | 0.145 / 0.062 / 0.197 | 0.154 / 0.066 / 0.202 |
| v2 | L1 | 60 | box max 0.752, box > 0.1 in 51 | 0.061 / 0.000 / 0.016 | 0.076 / 0.000 / 0.014 |
| v2 | L2 | 60 | box max 0.758, box > 0.1 in 54 | 0.122 / 0.061 / 0.147 | 0.153 / 0.049 / 0.163 |

Fit diagnostics over all 660 tagged-set views (`ALIGN/out/module_v{1,2}check.jsonl`):
- no fit was refused;
- ECC correlation is at least 0.760 (v1) and 0.823 (v2);
- the box shift is at most 33.7 px;
- the local scale is 0.988–1.010;
- a view takes 1.6 s (v1) and 1.3 s (v2) on average, on one GB10 core.

**Without registration, the v0.3 cue is unusable.** With identity, the box already reads 0.31–0.34 changed (median) on visible views where nothing changed.

### Alternatives tried and dropped

All are on the v1 check, against the tag homography, with tags excluded (`ALIGN/out/align_v1check_summary.txt`, `align_v1check_global_summary.txt`). Counts are views over 10 px, out of 150 per lighting, L1 / L2.

**Local window** (box grown by 3, the box grown by 0.5 left out), ECC on edges:

| model | views > 10 px | max |
|---|---|---|
| translation | 33 / 29 | 406 px |
| translation without the robust pass | 30 / 23 | — |
| Euclidean | 31 / 33 | — |
| homography without the robust pass | 50 / 51 | — |

Most failures are on the design cases: a plain shelf around the mug leaves too little texture in a local window.

**SIFT + RANSAC homography over the whole frame.**
- L1 is median 1.98 px, with 10 views over 10 px.
- L2 is median 4.30 px, p90 252 px and max 11845 px, with 51 views over 10 px. It fails under the lighting change.
- Local SIFT was dropped after a two-view spot check whose output was not saved. No number is quoted for it.

**SIFT then local translation ECC:** 25 / 40 views over 10 px.

**Global ECC on edges:**

| variant | median | max | views > 10 px |
|---|---|---|---|
| homography (adopted) | 0.40 / 0.49 | 1.93 / 1.95 | 0 / 0 |
| affine | 0.84 / 0.80 | 2.02 / 1.98 | 0 / 0 |
| homography + robust block-rejection pass at half resolution | 1.74 / 2.95 | — | 22 / 33 |

The robust pass dropped good blocks.

**Refusal is a gross-failure guard only.** On six deliberately wrong pairs, a reference from one stop with a query from another (`ALIGN/out/wrong_stop_pairs.json`), four were refused (cc < 0.5 or scale off by more than 10 %). Two were accepted:
- SD1005 P2 against P3: cc 0.81, box shift 97 px;
- DES0 P2 against P3: cc 0.93, shift 62 px.

A homography between two stops is a real front-plane mapping, so ECC can lock onto it. Large-baseline parallax at the target's depth is then not detected. In the study, a reference is only ever paired with its own stop's revisit.

## (c) and the cue evaluation on the tag-free set: pending

These are waiting for `SIM/shelf_dev_v3_check/` and its ground-truth query-to-reference homographies, for the front plane and for the plane at the target's depth. At 09:12 the set was still rendering, and it had no jobs or truth files yet.

## Caveats

- (b) scores against a front-plane homography; the error at the target's depth is unknown until (c). The functional score on visible views is the closest proxy here, and it matches the tag registration.
- (a) has no parallax at all, so it measures the optimiser and its robustness to lighting, an occluder and a removal, not the single-plane model.
- The tag-region exclusion in (b) is a union mask over both images, applied in reference coordinates. Tag pixels that move into the fit region by more than the ×2.2 growth are possible but unlikely: the shift is ≤ 34 px and the card is ~75 px.
