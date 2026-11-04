# Quality under image registration vs the true alignment (2026-11-04)

**Simulation, development data, evaluation only.** These are the 1800 views of the 60 random shelf_dev_v3 arrangements. The simulator's true alignment is a diagnostic here: it is never an evidence, calibration or policy input.

## What ran

- **Driver:** `tools/pipeline.py configs/pipeline/shelf_dev_v3_truealign.json --from-stage evidence`. It finished with no alerts and wrote `runs/shelf_dev_v3_truealign/`.
- **Evidence:** `sim/true_align_evidence.py` (commit f62bc00) runs evidence v0.3 unchanged, but on the true homography of the target plane and of the shelf's front plane. Each plane took 266–267 s on the GB10.
- **Comparison:** `tools/quality_resampling.py` compares each view with the deployable v0.3 run on ECC image registration. Its output is `runs/shelf_dev_v3_truealign/quality_resampling.md` and `.json`.

## Results

| Quality moves by more than | vs true target plane (L1 / L2) | vs true front plane (L1 / L2) |
|---|---|---|
| 0.05 | 82 / 142 | 29 / 54 |
| 0.1 | 54 / 85 | 11 / 29 |
| 0.3 | 16 / 24 | 3 / 4 |

- **The moving part is sharpness, every time:** exposure and valid never moved on any of these views.
- **The fragility is in the measure, not the registration.** Between the two *true* warps, quality differs by more than 0.1 on 149 of 1800 views, and by more than 0.3 on 37. Image registration is closer to the front plane (40 views over 0.1) than the two true warps are to each other. ECC fits mostly the front plane.
- **Cause:** sharpness is a Laplacian-variance ratio on a nearest-neighbour resampled ring, and a pixel or two of warp changes it.

**Decisions at the v0.3 select-v4 thresholds** (τ_q 0.3, τ_occ 0.35, τ_match 0.65):
- 8 of 1800 views decide differently against the true target plane, and 7 against the true front plane.
- Only one of those flips comes from quality: SD1019-removed-L2 P5. Its image-registration sharpness is 0.29, just under τ_q 0.3; on either true warp it is 1.0. That is a diagnostic removed view the gate then refuses, the safe direction.
- The others are obstruction (±0.02 around 0.35) or match (around 0.65) values sitting at their thresholds. One of them is SD1024-removed-L1 P1, the look-alike case in `notes/2026-11-04-dev60-evidence-variants.md`, whose match goes from 0.649 to 0.666.
- At the stricter illustrative setting (τ_q 0.6), more views flip: 9 and 15 gate flips against the target plane, in L1 and L2.

**Quality barely binds on this data:**
- τ_q is the grid minimum, 0.3, in 59 of the 60 v0.3 folds.
- Along the τ_q curve the dev false-clear rate stays 20/352 from 0.3 up to at least 0.5, and correct clear misses drop only from 227 to 226.
- Renders have no motion blur or defocus, so on sim the gate's work is done by obstruction.

## Recommendation

- **Keep quality as it is:** no new evidence version and no re-calibration. On this data its resampling fragility changes one gate decision in 1800, in the safe direction.
- **Disclose it as a limit:**
  - the v0.2.1 sharpness term is not resampling-invariant;
  - it would matter at a higher τ_q, or on real captures where blur and exposure make quality bind.
- **If quality is ever changed** (for example, sharpness on the unwarped query ring), version it as a new evidence revision. Keep a controlled cue comparison under identical alignment, and repeat calibration and the held-out checks.
