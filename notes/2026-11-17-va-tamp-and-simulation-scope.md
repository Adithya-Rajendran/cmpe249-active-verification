# VA-TAMP review and simulation-only scope

2026-11-17. Research assessment and design decision. No new experiment ran.

## Position

Continue the narrow empirical question. Do not claim a new principle of evidence gating or active perception.
VA-TAMP does not establish that this project's exact controlled comparison is already solved.
The distinction is a research question, not a demonstrated contribution or assurance of publication.
Confidence is high about conceptual overlap and moderate about the remaining empirical opportunity.

The researcher explicitly selected Isaac Sim and MuJoCo, with no physical-camera study.
The paper will describe simulator results. Physical deployment remains future work.
The earlier real-camera plan is [archived](../docs/archive/2026-11-17-simulation-only/REAL_CAMERA_PLAN.md).

## Primary-source reading

[VA-TAMP](https://arxiv.org/html/2611.16213v1), sections III–VII, separates hypothesized meshes from observed occupancy and verifies action requirements through planned observations.
Table I reports 9/10, 10/10 and 10/10 successes across three manipulation tasks.
These are the authors' results, not ours.
Its initial view must contain goal-referenced objects. It does not report our matched RGB removal-support experiment.
The [arXiv record](https://arxiv.org/abs/2611.16213) dates v1 to 12 November 2026. Venue acceptance was not verified.

[Evidence-Gated TAMP, v2](https://arxiv.org/html/2608.20084v2), sections III–VI, is an additional close overlap.
EAFG uses observations to choose planning, further inspection or halting, including absent-object tasks.
Its simulation assumes error-free detection of visible objects. Appropriate halting requires inspection of specified source regions during scoring.
Thus, absence-aware gating is also established prior work. Our distinction cannot be the addition of an UNKNOWN output alone.

Two existing close sources remain relevant.
[Selective Commitment](https://arxiv.org/html/2609.23131v1), section III-D, defines defer as abstention from grasping, without certifying absence.
[Absence Detection from Robot Memory](https://arxiv.org/html/2611.07493v1), sections III–IV, verifies retrieved candidates and evaluates absent queries.
Neither comparison justifies a broad novelty claim for memory, verification or abstention.
This targeted reading does not establish an exhaustive literature gap.

## Candidate question

When a remembered target disappears from view, which decisions remain justified by the supplied RGB evidence?
Test presence, removal and unresolved observations with matched target interventions.
Measure correct useful decisions and unsupported commitments together, under viewpoint changes and simulator transfer.

The single-pair input boundary is a controlled test condition. It is not an invention by itself.
Using two simulators is a generalization check. It is not an independent novelty claim.
A publishable result must identify a repeatable failure or improvement beyond credible image-only and change-detection controls.
The [completed results](2026-11-17-scoring-and-label-review.md) currently demonstrate guard sensitivity and label uncertainty, without a model advantage.

## Changes to implement

1. Make the evidence for each decision inspectable. Record localization validity, quality, target match, obstruction and the failed requirement.
2. Keep previous target presence separate from evidence at the revisit. A blocked view cannot renew a previous definite state.
3. Use missing decision evidence as the target of any later view-selection experiment.

The first change is a logging and method-interface requirement. It introduces no new threshold in the completed runs.
The second is already consistent with the current single-pair method.
The third is conditional future work. A useful selector must improve terminal decisions under equal view and movement budgets.
Predicted visibility must use permitted observations and known candidate poses. It must not inspect an unrequested query image.

Do not copy a full manipulation stack into this study merely to resemble prior work.
An RGB-D occupancy comparator would have a different sensor boundary and needs a separately named comparison.
Perfect simulator occupancy can serve as an evaluator diagnostic, never as an ordinary RGB method's evidence.

## Simulation design and label consequences

The next work is [backend qualification and a paired simulator pilot](../development/cross_sim/PLAN.md).
The current MuJoCo backend implements the legacy shelf rig, not the current textured arrangement cohort.
Sources: [MuJoCo backend](../sim/mujoco_rig.py), [Isaac backend](../sim/isaac_rig.py) and [MuJoCo tests](../tests/test_sim_mujoco.py).
Terra independently inspected these sources in a read-only review. No runtime qualification was performed in this session.

Use common geometry and camera definitions across engines. Record renderer-specific materials and lights explicitly.
Keep every derivative of an arrangement in one split, including both engines.
Freeze the method before evaluating transfer. Report renderer differences separately from changes to object geometry.

Simulator state supplies exact physical labels under the defined intervention.
Masks describe visibility and obstruction. They do not prove that a target fragment is recognizable.
An exact identical-input pair with opposite physical states can establish unresolved evidence for that pair.
Check all supplied pixels and metadata. A zero target mask or small pixel difference cannot substitute for that check.
The completed stress study found no exact identical query pairs, so its UNKNOWN labels do not gain automatic certification.

AI support judgments remain explicit, provisional and separate from physical truth.
The qualification hold stays active for support-based claims. Human annotation is not a mandatory dependency of the new simulation scope.
Any alternative validation must establish the particular support endpoint it is used to score.

## Decision and stop conditions

The simulation-only direction replaces the real-camera next step. It does not reclassify completed stress results.
Keep the declared model-comparison hold and the requirement for strong geometric controls.
If credible controls already solve the intended decision problem, stop the proposed added-method claim.
If the closest literature already evaluates the same claimed comparison, select a different contribution.
Do not expand the task solely to manufacture a favorable error pattern.

No simulator run, model run, dependency installation, physical capture, commit or publication occurred during this review.
