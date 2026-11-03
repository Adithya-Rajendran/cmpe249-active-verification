# Amendment: the course study runs in simulation (2026-11-02)

**Status:**
- **Decided** by the student on 2026-11-02.
- **The instructor agrees.** He suggested simulation and told the student that the project may change from the proposed version as needed.
- **The parameters marked [open] are still open.** They must be fixed and recorded here before any test scene is rendered.

This amends [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) and [`PROPOSAL.md`](PROPOSAL.md). Both planned a physical study and allowed simulation only as a separately reported extension. The proposal said a synthetic replica "cannot replace independent physical evidence". That remains true: the course results will be **simulation results** and will be reported as such. A real-robot study with the PIs is planned after the course, and the capture tooling (`capture/`) is kept for it.

## Why

Physical capture was the bottleneck.
- In pilot 2, setup took 34 minutes and 13 photos took 6, at a single stop ([notes](../notes/2026-10-20-pilot2.md)).
- The plan needs about 35 photos per arrangement (5 stops × reference + 3 conditions × 2 lightings) for 30 arrangements. With a hand-moved stand, that is over a thousand placements.

The simulated rig ([notes](../notes/2026-11-02-simulation.md), [`sim/`](../sim/README.md)) renders all five stops in about 4 s with exact labels. It reproduces the 2026-10-10 camera and occluder design in two independent renderers.

## What stays as planned

- **Task and decisions:**
  - the task, and the outputs present / removed / unknown, where unknown counts against coverage;
  - five stops, and a policy sees only the views it requests;
  - policies S, F, R, G, A sharing one stopping rule. k is now **fixed at 2** (the plan's "at least two distinct positions"), with k = 3 as a sensitivity run; it is no longer selected on development data (2026-11-02).
- **Measurement:**
  - the metrics (false-removal rate, removal recall, coverage, resolved accuracy) and the budgets and travel caps as curves;
  - splits by arrangement, paired arrangement-level bootstrap, and the number of arrangements reported with every result.
- **Protocol:**
  - matched occluder layouts for occluded and removed episodes;
  - a predeclared indistinguishable subset;
  - the evidence thresholds calibrated per view, and ranker A trained, on development arrangements only, then **frozen before any test arrangement is rendered**. The travel penalty is **fixed at λ = 0** a priori, with the other λ classes as sensitivity runs; it is no longer selected on outcomes (select-v4, 2026-11-02).
- **Perception:** the same front end (YOLO-World + DINOv2 evidence). Evidence v0.3, if built, is developed on simulated dev arrangements and public data.

## What changes

| | Physical plan | Simulation study |
|---|---|---|
| Simulator | n/a | Isaac Sim 6.2 on the GB10 is the single simulator. The MuJoCo backend (`sim/mujoco_rig.py`) is kept only as an independent geometry cross-check, run on request |
| Images | Pi Camera Module 3 photos | Isaac Sim 6.2 RTX renders of the measured shelf rig with the same camera model (2304×1296, f 1731.26 px) |
| Truth and visibility labels | planned conditions + annotation | exact, from the scene and the simulator's instance masks |
| Arrangements | hand-built | generated from a seed by a fixed generator (objects, poses, occluders, conditions) |
| Lighting | lamps L1 / L2 | simulated presets L1 / L2 |
| Environments | one shelf (or one tabletop) | the measured shelf rig replica first, then a tabletop (decided 2026-11-02); more scene types optional |
| Stops | taped floor marks | camera poses from a positions file, one per environment |
| Travel | path length between stops | the same |

## Integrity rules specific to simulation

1. **Seeds.** Dev and test seeds come from disjoint ranges that are fixed here before rendering starts. No test seed is rendered, scored or viewed before the freeze ([`FREEZE.md`, to be written]).
2. **Freeze the generator too.** The generator's code, object set, parameter ranges, renderer settings and the Isaac Sim version are frozen, with a recorded hash, before test rendering. Changing any of them after a test render means new test seeds.
3. **No tuning on test.** If dev and test look different after the freeze, that is a finding, not a reason to tune.
4. **Reporting.**
   - Every result is labelled as simulation.
   - Pilots 1 and 2 are the only real photos. They serve as a sanity check of score ranges (sim vs real), reported separately and never pooled with the simulation results.
5. **Licences.** Object models keep their licences (YCB-Video / YCB: see the dataset README on the share; Google Scanned Objects: CC BY 4.0). Renders used in a paper will carry attribution.

## Decided by the student (2026-11-02)

- **Environments.** Start with the measured shelf rig, where the simulation is cross-checked against the 2026-10-10 design. Then add a tabletop with cameras around it.
- **Cameras.**
  - The arc of stops is kept: the far-spot option A, with each stop's pitch set so all four tags and #3 stay in frame. At option A's P1 that pitch is 7.35–8.10°, not the drafted 6.83°.
  - Additional cameras may be added, e.g. a denser arc or a second height, since in simulation they cost nothing.
  - The final stop set per environment, and the budgets and travel caps over it, are fixed before any test render.

- **Reference pairing.** The primary evaluation compares every query, at L1 and at L2, against the **L1 reference** at the same stop: the robot's memory is recorded once, and the lighting may differ at revisit. The other pairing (each query against the reference at its own lighting) removes only the mismatch in lighting between reference and query; lighting can still affect detection and image quality. It is not a relabelling of the primary run. The reference enters the evidence scores, so it can change the policies' decisions and the views they request. If it is used at all, it is a separate secondary run, reported separately.
- **Reporting by lighting.** Every result is reported **separately for L1 and L2 queries**: metrics, budget curves, per-condition tables, and the paired arrangement-level bootstrap within each lighting. With L1 references, L1 means "same lighting as the reference" and L2 "lighting changed since the reference". A number pooled over lighting may appear only as a labelled secondary line, never as a headline.
- **Policy G** (approved 2026-11-02, implementing the 2026-10-12 decision). Primary G is the normalised geometric selector with the reachability lookahead off. The lookahead is reported as the ablation "G-lookahead" on the same scorer, so the contrast isolates lookahead. The legacy G-myopic stays a separate ablation.

## Design choices confirmed by the student (2026-11-02 / 11-03)

The analysis rules are in [`SIM_PREREGISTRATION.md`](SIM_PREREGISTRATION.md). It gets a final review after the development loop (including A) and is frozen before any test scene is generated.

- **Counts:** 60 dev + 100 test random arrangements per environment (shelf and tabletop), with DES0–DES3 as separate dev checks. This is a practical allocation, not a power guarantee. Dev throughput is measured end to end before the counts are frozen.
- **Occluder quotas:** S8 20 %, M16 30 % (left/right balanced), IND28 10 %, white box 20 %, YCB box 20 %. They are fixed as slot lists before rendering; retries and replacements stay within the family.
- **α_clear = 0.10:** an empirical per-view calibration constraint, with no guarantee that any policy's FRR is ≤ 10 %. If no setting meets it, calibration fails loudly and nothing is relaxed.
- **Primary operating point:** uncapped, b = 3 (view-budgeted verification), with travel reported alongside. The finite caps and b = 1 / 2 are secondary and are not promoted on the basis of dev results.
- **No third environment.** Claims concern new arrangements in the shelf and tabletop environments.
- **Labels:** mask-derived labels are *geometric* (visibility), not recognisability, and set no theoretical ceiling on coverage.
- **Uncertainty:** the paired arrangement-level cluster bootstrap, stratified by occluder family, is primary.
- **Occlusion-gate-off ablation:** F and R only.
- **The 2026-10-10 design cases** (option-A picks, DES0–DES3) are kept as separate dev checks.
