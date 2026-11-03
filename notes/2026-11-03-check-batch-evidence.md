# Check batch through the frozen evidence step and select-v4 (2026-11-03)

**Simulation only.** Everything here was rendered in Isaac Sim. This is a **pipeline smoke test on 6 random arrangements**; it is not a study result and selects nothing.

**Input.** The checkpoint-2 check batch: `/mnt/bench-data/datasets/cmpe249-active-verification/sim/shelf_dev_v1_check/`.
- It holds one arrangement per quota slot class (SD1000 M16-L, SD1001 IND28, SD1002 S8, SD1005 ycb_box, SD1007 M16-R, SD1010 white_box) plus DES0–DES3.
- Rendering and its checks are in [the batch note](2026-11-03-dev-batch-check.md).

## What ran

- **Evidence v0.2.1** (frozen; `perception/evidence.py` sha256 `d0e0927f…`).
  - Ran on the GB10 from its repo copy. `perception/evidence.py`, `perception/common.py`, `capture/tags.py`, `capture/ecc.py` and `verify/schema.py` there are byte-identical to main. Only `verify/stopping.py` differs, and evidence uses it only for the diagnostic flags, not for the records.
  - `jobs_refL1.json`: 180 views in 34.5 s. `jobs_design_refL1.json`: 120 views in 22.4 s.
  - Outputs: `evidence_refL1.jsonl` (sha256 `a2a75da1…`) and `evidence_design_refL1.jsonl`, next to the jobs.
- **select-v4, joint search** (branch `cp2`), on the 6 random arrangements with `--manifest/--evidence/--view-labels/--positions`.
  - Exit 3, calibration infeasible. Record: `runs/2026-11-03-cp2-smoke/select/calibration_infeasible.json`.
- **verify.run** on the same files.
  - Exit 0 with the placeholder thresholds, labelled SIMULATION. Output: `runs/2026-11-03-cp2-smoke/run/`.
  - It shows that the sim manifest, evidence and view-labels path works end to end. Its numbers use placeholder thresholds and are not quoted.

## The finding: the occlusion cue misses plain occluders

Calibration fails on the **false-clear limit**: non-diagnostic views of occluded-present episodes that pass the τ_q / τ_occ gate.
- The best rate on the whole grid is **20 / 36 = 0.556**, at τ_q 0.6 and τ_occ 0.1, against α_clear = 0.10.
- The false clear-miss limit alone is easy to meet: 0 / 84 is achievable.

Which views pass the gate, from `runs/2026-11-03-cp2-smoke/gate_breakdown.txt` (script `gate_breakdown.py` next to it):

| Non-diagnostic occluded view | Pass the gate | Flagged |
|---|---|---|
| Hidden (visible fraction ≤ 0.05) | 13 | 9 |
| Partial, 0.05–0.5 | 3 | 3 |
| Partial, 0.5–0.9 | 4 | 4 |

- **These are real occlusions, not a labelling artefact.** 13 of the 22 fully hidden views pass with obstruction cues of 0.00–0.10.
- **By family** (hidden views, pass / flagged):
  - IND28 8 / 2;
  - M16 4 / 2: all 4 of SD1007 (M16-R) pass, and both of SD1000's are flagged;
  - white box 1 / 1;
  - S8 0 / 2;
  - YCB box 0 / 2.
- **Match on those hidden views** ranges from 0.33 to 0.57. Most would be clear misses, and two clear misses make a false REMOVED.
- **Likely cause.** This is an inference; no cue ablation has been run.
  - Evidence v0.2.1's obstruction cue is max(detector cue, DINOv2 novelty against the whole reference window).
  - A plain white board in front of a white shelf is neither a detected object nor novel against the white walls in the window.
  - This extends the same-footprint limit already recorded in `perception/README.md` ("Known limits found after the freeze") to large plain boards.

## What this means for the procedure

- select-v4 behaved as pre-registered (§7): it failed loudly, with no relaxation and no fallback.
- §7 says any change (grid, α, or evidence v0.3) is a dated addendum made on dev before the freeze.
- Six arrangements cannot fix a rate. But 0.556 is far above 0.10, so the full dev set is unlikely to be feasible with v0.2.1 as it stands.
- The direction (an evidence v0.3 cue, occluder appearance, or both) is the student's decision.
