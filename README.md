# Evidence-Aware Verification for Robot Scene Memory

CMPE 249, San José State University, Fall 2026. Adithya Rajendran. Instructor: Kaikai Liu.

A robot saw an object on an earlier visit. When it looks again, should it keep that memory, record a removal, or wait for a clearer view?

## When a hidden box looks gone

A yellow outline marks a gelatin box on a simulated MuJoCo shelf. A panel hides the box in one revisit and covers its empty spot in the other.

![MuJoCo shelf PAM01 at stop P5 with a moved camera. The marked reference precedes the pixel-identical “object hidden” and “object removed” revisits.](docs/figures/mujoco-hidden-removed.png)

Each verifier received the same reference image and target box with one revisit at a time. The full revisits match pixel for pixel. Both passed the geometry check. Qwen3.5-9B, a vision-language model, answered `REMOVED` twice. Homography and dense geometry each answered `UNKNOWN` twice. Qwen's answer fit the empty spot but would erase a valid memory in the hidden case.

### Simulation results

Three methods with fixed settings scored 1,280 queries on 16 shelf and tabletop layouts shared by Isaac Sim and MuJoCo. At each camera and blocker position, the simulators kept or removed a gelatin box or a large marker. Each verifier could answer `PRESENT`, `REMOVED` or `UNKNOWN`.

Homography aligned the images with one transformation. Dense geometry mapped individual points. Both used the same decision rule. When geometry could map enough of the remembered location, Qwen inspected the images and local crops.

For clear removals, the camera shifted 6 cm and turned 3° with the blocker out of the way. False removals count all `REMOVED` answers when the target stayed.

| Simulation outcome | Homography | Dense geometry | Qwen3.5-9B with geometry check |
| --- | ---: | ---: | ---: |
| Clear removals detected in Isaac Sim | 67/80 | 76/80 | 80/80 |
| Clear removals detected in MuJoCo | 63/80 | 78/80 | 79/80 |
| False removals in Isaac Sim | 0/320 | 0/320 | 24/320 |
| False removals in MuJoCo | 1/320 | 2/320 | 27/320 |

Dense geometry avoided homography's MuJoCo error but introduced two others. The audit required zero new errors to call dense geometry an improvement. Qwen found 159 of 160 clear removals and falsely reported removal on 51 of 640 revisits where the target stayed.

### Why match the pairs?

Both simulators fully hid the target at the same 88 positions. MuJoCo produced identical images, target boxes and instructions across the two states at all 88. The same definite answer must be wrong in one scene. Isaac Sim produced zero pixel-identical pairs. Whether its pixel differences reveal removal remains open.

Homography abstained on all 88 identical pairs. Dense geometry called both states `REMOVED` on one pair. The geometry check returned `UNKNOWN` on 84 pairs before Qwen ran. Qwen answered `REMOVED` on all eight calls from the four remaining pairs. Across both simulators, geometry stopped 446 queries. Qwen itself chose `UNKNOWN` on two of its 834 calls.

We also revisited the shelf with the camera leveled. The panel still hid the box, but the geometry check now stopped both queries before Qwen saw them. From straight ahead, the system also reported a fully visible marker removed once in each simulator. [Shelf-angle experiment with pictures](notes/2026-11-18-level-shelf.md)

For the full study, read the [paper](paper/Rajendran-counterfactual-scene-memory.pdf). To check the example yourself, open the [saved image hashes and answers](paper/evidence/fresh-illustration.json).

*The figure uses YCB object models by Berk Calli and colleagues from the BOP collection of YCB-Video models. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*

The [evidence guide](paper/README.md) explains how to reproduce the results. The [prospective protocol](development/paper_audit/PROTOCOL.md) records the comparison rules.

## Current study

A missing detection does not establish removal. The target can remain behind an occluder.
The verifier returns PRESENT, REMOVED or UNKNOWN from a reference image and one revisit image.
REMOVED means absent from the remembered region under the stated task assumptions. It does not mean absent from the whole scene.

The image verifier checks separate evidence for presence and removal.
An agreement veto can reject its errors. A conditional cascade can add answers where it abstains.
The evaluation measures unsupported answers and useful correct answers together.

The paper compares two image-only geometry methods with a vision-language model that can answer UNKNOWN.
The earlier veto and cascade proposals still need broader visual support labels.

| Read | Purpose |
| --- | --- |
| [Standalone introduction](CASE_STUDY.md) | Illustrated case study for a first reading |
| [Paper](paper/Rajendran-counterfactual-scene-memory.pdf) | Three-verifier comparison on fresh counterfactual pairs, with independent checks |
| [Document guide](docs/README.md) | Current research documents and historical records |
| [Proposal](docs/PROPOSAL.md) | Research question, contribution and closest work |
| [Method](docs/METHOD.md) | Inputs, labels, decision rule and controls |
| [Evaluation plan](docs/EVALUATION_PLAN.md) | Data, comparisons, metrics and readiness checks |
| [Stress experiment](development/stress/PROTOCOL.md) | Declared camera levels, labels, controls and execution checks |
| [Research update](notes/2026-11-17-research-and-stress-plan.md) | Closest work and corrections to the earlier interpretations |
| [Stress execution](notes/2026-11-17-stress-execution.md) | Completed smoke, AI label audit and full rendering status |
| [Scoring and label review](notes/2026-11-17-scoring-and-label-review.md) | Completed controls, independent recount and label qualification |
| [Paired simulator results](notes/2026-11-17-cross-sim-pilot.md) | Fixed controls, one certified ambiguous error, and the recurring tabletop failure |
| [RGB geometry results](notes/2026-11-18-rgb-geometry-comparison.md) | Tabletop gains, two shelf errors and the decision to stop this version |
| [Visual support qualification](notes/2026-11-18-support-label-qualification.md) | Blinded Terra judgments, independent evidence review and all 32 image examples |
| [Locating the object first](notes/2026-11-18-label-localization.md) | Matched instructions, both simulator outcomes and the visual evidence for all 30 cases |
| [Simulator evaluation plan](development/cross_sim/PLAN.md) | Adapter qualification and decision rules for the paired simulator studies |
| [VA-TAMP review](notes/2026-11-17-va-tamp-and-simulation-scope.md) | Prior-work overlap, candidate contribution and simulation scope |
| [Freeze policy](docs/FREEZE_POLICY.md) | Development, release conditions and historical records |
| [Label audit](notes/2026-11-16-development-support-audit.md) | Support labels, ambiguous fragments and review limits |
| [Development pilot](notes/2026-11-16-development-pilot.md) | Completed runs, failed baseline qualification and next decisions |

Publication depends on evidence beyond the strongest controls and a defensible contribution.
Neither a new title nor a freeze establishes novelty.

## Earlier development

The first study tested camera view selection. That study is complete, with important limits in its design.
The current study tests whether an image supports a decision about a remembered object.
The first development pilot is complete. Its model verifier failed an obvious retained-target control.

The camera-displacement run is complete. Guard controls recover many abstentions without added physical errors.
AI support judgments remain provisional after qualification and a repeated audit.

The paired pilot tested 1,200 queries in Isaac Sim and MuJoCo. The wider alignment guard gave 803 correct answers and one false removal.
RGB geometry resolved all 25 clear tabletop removals at the largest displacement in each simulator.
It made two new false removals in one shelf setup. The declared rule ends this version.

In a separate label pilot, Terra identified all four partly visible retained objects per simulator.
It abstained on all four partly occluded removals in each simulator and failed qualification.

On four fresh layouts, asking Terra to locate the object first recovered two partial removals in MuJoCo and zero in Isaac.
Only MuJoCo passed qualification.
The fixed-camera calibration detected every parked-blocker change and missed its required partial yield in both engines.

The researcher selected simulation-only evaluation. Physical-camera transfer remains future work.
A comparison on general visual support still requires validated support labels.

## Completed study

The [completed study](https://github.com/Adithya-Rajendran/cmpe249-active-verification/tree/b81ded23d6710bf792e96ac53b5104fce6b177a1) ended at `b81ded2`.
The historical tag `simulation-freeze-1` identifies the [original freeze commit](https://github.com/Adithya-Rajendran/cmpe249-active-verification/tree/6d0ed94a73a20ff41549a2383b07f0358bf69ad9).
This checkout starts from the completed study. The new documents do not replace its results.

| Record | Purpose |
| --- | --- |
| [Frozen results](notes/2026-11-06-test-results.md) | Original declared comparisons |
| [Post-hoc review](notes/2026-11-12-posthoc-review.md) | Later analyses and interpretation limits |
| [Historical freeze](docs/FREEZE.md) | Original manifest and runbook |
| [Freeze addenda](docs/FREEZE_ADDENDA.md) | Exceptions, later analyses and closure |
| [Historical experiment plan](docs/EXPERIMENT_PLAN.md) | Original study only |
| [Historical reproduction guide](docs/REPRODUCE.md) | Original software and data pipeline |
| [Document snapshots](docs/archive/2026-11-16-reset/ARCHIVE.md) | Previous README and proposal |

The old test data are development data for the new study.
The old freeze does not authorize a new test or establish the validity of the new method.

## Existing software

| Folder | Existing function |
| --- | --- |
| [capture](capture/README.md) | Camera capture and acquisition checks |
| [perception](perception/README.md) | Image evidence and reference matching |
| [verify](verify/README.md) | Replay, old selectors and the shared stopping rule |
| [sim](sim/README.md) | Scene generation, rendering and geometric labels |
| [development](development/README.md) | Current pilot plans, completed studies and development tools |
| [public_data](public_data/README.md) | Development data adapters |

The historical code implements the completed study. Its pinned files and results remain unchanged.
The [development runner](development/pilot/README.md) implements the new controls and arrangement-level analysis.
The pilot uses existing simulator images. It does not establish an implemented robot-memory system or a method advantage.

## Course work

Read the [two-page progress feedback PDF](docs/feedback/2026-11-17/Rajendran-CMPE249-progress-feedback-2026-11-17.pdf) for the results and questions for Professor Liu.
Professor Liu's optional Canvas check-in invites feedback on project progress and technical questions.
The discussion focuses on support labels, the strongest comparison and transfer between simulators.

## AI assistance

AI tools assisted with literature review, software, analysis and document preparation.
The researcher must verify the sources, experimental records and final claims.
The existing submission files remain historical records of the original proposal.
