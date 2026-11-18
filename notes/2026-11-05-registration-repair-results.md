# Bounded registration repair: results (2026-11-05)

**Simulation, development data.** This is the outcome of the pinned plan `notes/2026-11-04-registration-repair-plan.md` (b144e04, Amendment 1 at 0af6fb4). The evidence thresholds are the fixed shelf v03_image ones (sha256 4a06de0e...). Simulator truth is used for scoring only. Fitting masks stay separate from evidence masks.

**Outcome: no candidate passes the retained criteria in both environments, so nothing is locked.** Under the plan, the remaining 49 tabletop dev arrangements are not rendered or scored. How to proceed is the researcher's decision.

## What ran

- **Code:** `perception/registration_v032.py` (candidates C0, C1, C1M, C2, C2M) and `tools/registration_compare.py`, at commit 127462b (C1 at 1e34e22).
- **Driver:** `tools/pipeline.py configs/pipeline/regrepair_<cand>_<env>.json --from-stage evidence`, on the GB10 (gx10-8a10: 20 CPUs, NVIDIA GB10). Each run used 10 workers, with torch/OMP and cv2 at 1 thread each. Every C0 arm ran in the same job as its candidate.
- **Sets and jobs:**
  - Tabletop: `tabletop_transfer11`, 165 views per lighting.
  - Shelf: `shelf_dev_v3`, the random arrangements (900 views per lighting, gating) and the design cases (60 per lighting, reported only).
- **Control:** C0 through the harness reproduces the original v0.3 evidence on every set. The maximum difference in match, quality and obstruction is 0, and the refusals are equal.
- **Score reports:** `runs/regrepair_<cand>_<env>/score/*.md` and `.json`. These are untracked, and the evidence diags are on the share.

## Criteria by environment and lighting (random arrangements; C0 → candidate)

| Candidate | Tabletop L1 | Tabletop L2 | Shelf L1 | Shelf L2 |
|---|---|---|---|---|
| C1 | **1 fails**: refused 12 → 10 / 165 | **1 fails**: 9 → 9 | **3, 4, 5 fail**: refused 0 → 2 / 900 | **3, 5 fail**: refused 0 → 2 |
| C1M | **1 fails**: 12 → 10; identical to C1 | **1 fails**: 9 → 9 | **3, 4, 5 fail**: refused 0 → 1 | **3, 5 fail**: refused 0 → 2 |
| C2 | **6 fails**: refused 12 → 7; 5 newly accepted, one 72.46 px off | **1 fails**: 9 → 9 | pass: refused 0 → 0 | pass: refused 0 → 0 |
| C2M | **6 fails**: as C2 | **1 fails**: 9 → 9 | pass | pass |

Criteria:
1. Fewer than 5 % refusals: tabletop ≤ 8 / 165, shelf ≤ 44 / 900.
2. No present-labelled clear miss.
3. Removed-labelled strong matches no more than C0.
4. Diagnostic removal yield at least C0.
5. Geometry budget on the views both versions accept.
6. Newly accepted views within the C0-accepted maximum + 2 px.

Details behind the failures:
- **Shelf C1 and C1M, criterion 3:** removed-labelled strong matches go from 0 to 1 in L1 and from 1 to 2 in L2.
- **Shelf C1 and C1M, criterion 4:** in L1, removal yield goes from 115 to 114 out of 124.
- **Shelf C1 and C1M, criterion 5:** the maximum common-view error is 32.6 / 69.4 px for C1 (L1 / L2) and 103.7 / 69.4 px for C1M.
- **Tabletop C2 and C2M, criterion 6:** the bound is 17.12 + 2 px. TD5000-occluded-L1 P3 is accepted at 72.46 px, with silhouette IoU 0.535.

On the shelf, C2 and C2M keep C0's usefulness:
- present-labelled clear misses stay at 0 / 600;
- removed-labelled strong matches stay at 0 (L1) and 1 (L2) out of 300;
- removal yield goes from 115 to 115 (L1) and from 112 to 113 (L2) out of 124.

The shelf design cases (60 per lighting) are reported only. Criterion 5 fails there in L2 for C1 and C1M. C2 and C2M meet every criterion on them.

## Runtime

These figures are reported under Amendment 1 and do not gate. They are the median registration time against C0's, by lighting.

| Candidate | Tabletop L1 / L2 | Shelf L1 / L2 |
|---|---|---|
| C1 | 2.15 / 2.03× | 20.3 / 16.7× |
| C1M | 5.02 / 4.32× | 41.7 / 33.0× |
| C2 | 126 / 136× | 175 / 148× |
| C2M | 288 / 227× | 349 / 283× |

Shelf random batches, 1800 views each:

| Candidate | Wall time | Throughput | C0 throughput |
|---|---|---|---|
| C2 | 15,051 s | 430.5 views/hour | 41,094 views/hour |
| C2M | 30,639 s | 211.5 views/hour | 40,699 views/hour |

Median complete-evidence time per view is 66–68 s for C2 and about 134 s for C2M, against C0's 0.51–0.60 s.

The complete-evidence medians, p90s, maxima and the other batches are in the score reports. Wall clocks include model loading. Slow execution is a disclosed limitation.

## Mechanism

These are diagnostic findings from the tabletop development set.

- **All the remaining refusals are small shifts.** The true shift of every view still refused is under 15 px, so identity is a good start. Both the identity start and the phase-correlation start converge to the same occluder-driven warp. That is why the multistart variants differ from their single-start versions on at most one view.
- **C1's block test barely engages.** Its relative outlier test drops only a few blocks when the whole fit is contaminated.
- **C2 can accept a wrong fit.** Its robust down-weighting lets a wrong warp pass every final check: the correlation is high on the reduced support. This is the failure the plan was guarding against.
