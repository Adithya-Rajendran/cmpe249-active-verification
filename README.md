# Evidence-Aware Verification for Robot Scene Memory

CMPE 249, San José State University, Fall 2026. Adithya Rajendran. Instructor: Kaikai Liu.

**Status: development, 2026-11-17 Pacific. No new study is frozen.**

The first study tested camera view selection. That study is complete, with important limits in its design.
The next study tests whether an image supports a decision about a remembered object.
The first development pilot is complete. Its model verifier failed an obvious retained-target control.

The camera-displacement run is complete. Guard controls recover many abstentions without added physical errors.
AI support judgments remain provisional after qualification and a repeated audit.

The next step qualifies MuJoCo scene support, then compares fixed image-only controls across Isaac Sim and MuJoCo.
The researcher selected simulation-only evaluation. Physical-camera transfer remains future work.
A further model comparison requires validated support labels, useful opportunity and successful verifier qualification.

## Current study

A missing detection does not establish removal. The target can remain behind an occluder.
The verifier returns PRESENT, REMOVED or UNKNOWN from a reference image and one revisit image.
REMOVED means absent from the remembered region under the stated task assumptions. It does not mean absent from the whole scene.

The image verifier checks separate evidence for presence and removal.
An agreement veto can reject its errors. A conditional cascade can add answers where it abstains.
The evaluation measures unsupported answers and useful correct answers together.

The main reference is image-only verification, including guard controls.
A later model comparison includes explicit UNKNOWN and calibrated confidence rejection.

| Read | Purpose |
| --- | --- |
| [Document guide](docs/README.md) | Current research documents and historical records |
| [Proposal](docs/PROPOSAL.md) | Research question, contribution and closest work |
| [Method](docs/METHOD.md) | Inputs, labels, decision rule and controls |
| [Evaluation plan](docs/EVALUATION_PLAN.md) | Data, comparisons, metrics and readiness checks |
| [Stress experiment](development/stress/PROTOCOL.md) | Declared camera levels, labels, controls and execution checks |
| [Research update](notes/2026-11-17-research-and-stress-plan.md) | Closest work and corrections to the earlier interpretations |
| [Stress execution](notes/2026-11-17-stress-execution.md) | Completed smoke, AI label audit and full rendering status |
| [Scoring and label review](notes/2026-11-17-scoring-and-label-review.md) | Completed controls, independent recount and label qualification |
| [Next simulator comparison](development/cross_sim/PLAN.md) | Backend qualification, paired scenes, physical-state scores and stopping rule |
| [VA-TAMP review](notes/2026-11-17-va-tamp-and-simulation-scope.md) | Prior-work overlap, candidate contribution and simulation scope |
| [Freeze policy](docs/FREEZE_POLICY.md) | Development, release conditions and historical records |
| [Label audit](notes/2026-11-16-development-support-audit.md) | Support labels, ambiguous fragments and review limits |
| [Development pilot](notes/2026-11-16-development-pilot.md) | Completed runs, failed baseline qualification and next decisions |

Publication depends on evidence beyond the strongest controls and a defensible contribution.
Neither a new title nor a freeze establishes novelty.

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

Professor Liu's optional Canvas check-in invites feedback on project progress and technical questions.
The discussion focuses on support labels, the strongest comparison and transfer between simulators.

## AI assistance

AI tools assisted with literature review, software, analysis and document preparation.
The researcher must verify the sources, experimental records and final claims.
The existing submission files remain historical records of the original proposal.
