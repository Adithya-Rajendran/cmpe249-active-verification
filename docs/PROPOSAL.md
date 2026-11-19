# Proposal: evidence-aware verification of a remembered object

**Development design, revised 2026-11-18 Pacific.**

## Question and scope

When camera pose changes, can a verifier retain supported decisions about a remembered object without committing on unanswerable views?

The input is a reference image, its target box and one revisit image.
The outputs are PRESENT, REMOVED and UNKNOWN.
The task concerns the remembered region. The primary scenes exclude relocation and substitution of an indistinguishable instance.
The method cannot infer absence from the entire scene from an empty remembered region.

The [method](METHOD.md) checks separate support for presence and removal.
The [evaluation](EVALUATION_PLAN.md) measures both factual correctness and whether the supplied view supports an answer.
The completed simulation measures image-only performance across camera displacements.
The [paired development pilot](../notes/2026-11-17-cross-sim-pilot.md) compared fixed controls across Isaac Sim and MuJoCo with recorded interventions.
It identified a repeatable alignment refusal within one MuJoCo target family and an exact-input ambiguity on which two controls committed.

The [RGB geometry comparison](../notes/2026-11-18-rgb-geometry-comparison.md) resolved the recurring removals and abstained on all 40 certified ambiguous pairs.
Two new false removals in one shelf arrangement ended this revision under its declared rule.

The researcher selected a simulation-only paper scope. Physical-camera transfer remains future work.
A later cascade must recover useful answers beyond credible image-only controls without excessive unsupported commitments.
Agreement veto is a separate error-reduction hypothesis. Neither mechanism is claimed as new.

## Candidate contribution

The proposed contribution is a controlled evaluation of evidence for memory updates.
Matched scenes separate physical state from observable evidence. A blocked target and a removed target can both require UNKNOWN.
Clear controls measure the cost of abstention. A declared camera displacement tests dependence on reference alignment.

This is a candidate empirical contribution. The gate itself is not claimed as a new principle or a calibrated safety guarantee.
A paper needs a reproducible finding that survives strong controls, fresh arrangements and an explicit generalization test.
The planned claims concern rendered observations in the declared simulators, not deployed robot-memory performance.
Simulator transfer alone is not novelty. A useful finding must survive strong controls and fresh arrangements.

## Why the design changed

The first study had little room for a better view selector at its primary setting.
Its frozen evidence could not produce the costly error under any view order in the completed test set.
The learned ranker also used a visibility target instead of terminal decision value.
The [post-hoc record](../notes/2026-11-12-posthoc-review.md) qualifies the completed results.
Those findings do not establish that active perception or learning is ineffective.

The new study changes the question and the decision unit.
It does not revise the old results or reuse the old test as independent evidence.

## Closest work and remaining uncertainty

| Source | Established overlap | Distinction to test |
| --- | --- | --- |
| [VA-TAMP, 2026 preprint](https://arxiv.org/html/2611.16213v1) | Observation-justified manipulation | Matched RGB verification of removal from a remembered region |
| [Evidence-Gated TAMP, 2026 v2](https://arxiv.org/html/2608.20084v2) | Evidence acquisition, feasibility gating and absent-object halting | Perception errors and supported removal decisions under a controlled RGB boundary |
| [CYWS-3D, 2023](https://arxiv.org/html/2308.10417v2) | Two-image change detection across viewpoints, with estimated-depth feature registration | Supported answers about a supplied region, rather than change boxes alone |
| [SceneDiff, 2026 revision](https://arxiv.org/html/2512.16908v2) | Verifiable changes across viewpoints and occlusion, including two-image evaluation | Explicit support-scored UNKNOWN and useful target decisions, subject to direct comparison |
| [GeoSCD, 2025](https://arxiv.org/html/2509.11292v1) | Two-image change masks with estimated geometric correspondences and explicit occlusion filtering | Selective target decisions need a declared adapter and direct comparison |
| [FOCUS, 2026](https://arxiv.org/html/2610.05639v1) | Uncertain object-state updates from aligned RGB-D revisits and a reference map | A single RGB pair without replacement or temporal fusion |
| [Jonathan, Paxton and Hager, 2017](https://arxiv.org/abs/1710.03948) | Robot scene reasoning retains objects through occlusion and checks evidence before removal. | Controlled paired evaluation of memory verification with explicit abstention |
| [SpatialUncertain, v2](https://arxiv.org/html/2605.30557v2) | Controlled occlusion, perspective ambiguity and abstention. Its perspective experiments include a reference view. | Matched hidden and removed states for a previously identified object |
| [VAD-R, 2026](https://arxiv.org/html/2609.32653v1) | Answerability, shortcut filtering, occlusion and absent information | A task-specific reference and revisit comparison, with useful memory decisions measured separately |
| [Selective Commitment, 2026](https://arxiv.org/html/2609.23131v1) | Robot memory, hidden and absent hypotheses, VLM observations and abstention | Verification of removal, rather than a decision to grasp or defer |
| [Absence Detection from Robot Memory, 2026](https://arxiv.org/html/2611.07493v1) | Retrieval plus visual verification for absent object queries | Removal since a supplied reference, with matched occlusion controls |
| [Geifman and El-Yaniv, 2017](https://arxiv.org/abs/1705.08500) | Selective classification trades answer coverage against error | A fair rejection baseline for this task, not a new abstention principle |

These distinctions are an interpretation of the cited methods. They do not prove an open publication opportunity.
The [2026-11-17 research check](../notes/2026-11-17-research-and-stress-plan.md) adds primary-text comparisons and narrows the claim.
The [VA-TAMP review](../notes/2026-11-17-va-tamp-and-simulation-scope.md) records the later reading and simulation-only decision.
This targeted search does not establish an exhaustive literature gap.

The review of Selective Commitment covered sections III and IV. Its defer action does not certify absence.
The earlier numerical confidence in novelty is withdrawn.

The researcher's slot rule remains: check the closest work before a freeze and before submission.
If a paper already tests the same matched task with scored UNKNOWN, stop this direction and select another topic.
Otherwise, state the exact remaining question before building a larger study.

## Next work

The [label audit](../notes/2026-11-16-development-support-audit.md) and [first development pilot](../notes/2026-11-16-development-pilot.md) are complete.
The model prompts failed an obvious retained-target control, despite valid numeric scoring.
The image-only control resolved all supported evaluation queries in the small cohort.
These findings do not establish a language-model method advantage.

The [full stress review](../notes/2026-11-17-scoring-and-label-review.md) is complete.
The guard controls recover correct answers. AI support judgments still need validation.
No support-based A/B/C decision follows from the sampled audit.

Locating the target before answering recovered two partial removals in MuJoCo and zero in Isaac Sim.
The [instruction comparison](../notes/2026-11-18-label-localization.md) ended under its cross-simulator rule.

1. Test repeatable local target changes at a fixed camera position against unchanged-target controls in both simulators.
2. Keep unresolved support judgments distinct from the complete physical-state scores.
3. Use the 200-pixel baseline and the completed RGB geometry comparison when assessing a new method.
4. Compare a justified method with credible geometric and confidence controls.
5. Plan fresh evaluation only after a repeatable benefit and a defensible contribution appear.

The [freeze policy](FREEZE_POLICY.md) blocks a new confirmatory claim until the required decisions and records exist.
Professor Liu wants to discuss the methodology. No fixed course deliverable or publication date is promised.
