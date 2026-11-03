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
  - policies S, F, R, G, A sharing one stopping rule, with k (at least 2 clear-miss stops) selected on development data.
- **Measurement:**
  - the metrics (false-removal rate, removal recall, coverage, resolved accuracy) and the budgets and travel caps as curves;
  - splits by arrangement, paired arrangement-level bootstrap, and the number of arrangements reported with every result.
- **Protocol:**
  - matched occluder layouts for occluded and removed episodes;
  - a predeclared indistinguishable subset;
  - thresholds, the travel penalty and ranker A selected on development arrangements only, then **frozen before any test arrangement is rendered**.
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

## Proposed defaults [open: confirm before any test render]

- **Held-out environment.** Whether to also test on an environment type not used in development (e.g. develop on the shelf and tabletop, test on a third) [open].
- **Arrangements.** 30 (10 dev / 20 test) as planned, × 3 conditions × 2 lightings = 180 episodes. Simulation makes more cheap, so a larger dev pool is possible, but the test arrangement count stays fixed in advance.
- **Objects.**
  - YCB-Video models first (the mug and bleach bottle already render with textures), then Google Scanned Objects.
  - Target and occluders are placed by the generator.
  - The design's occluder sizes (S8 / M16 / IND28) plus the pilot-2 failure case: a flat white box over the target's footprint.
- **Episodes from the 2026-10-10 design.** The four option-A occluder picks (CCHCC, CCHHC, CHHCC, HHHHH) are kept as a fixed, predeclared subset alongside the random arrangements.
