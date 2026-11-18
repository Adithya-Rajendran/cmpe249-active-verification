# Development pilot and reviewer decision, 2026-11-16

The pilot completed, but its model verifier failed a basic capability check.
Do not freeze this method or use this run to claim a useful advantage over a competent model.
Repair and qualify the verifier before a larger experiment.

Terra executed the work on Ascent. The main assistant reviewed labels, implementation, readiness and results.
The run uses existing Isaac Sim images. It creates no new scene, confirmatory test or robot-memory deployment.

## Records and scope

- [Label audit](2026-11-16-development-support-audit.md): cohort, annotations and review limits.
- [Readiness approval](../runs/2026-11-16-support-audit-v1/readiness-approved.json): inputs, source hashes and settings before task inference.
- [Pilot results](../runs/2026-11-16-support-audit-v1/RESULTS.md): raw predictions, calibration grids and timing.
- [Post-hoc diagnostic](../runs/2026-11-16-support-diagnostic-v1/RESULTS.md): separate checks after the unexpected output pattern.
- [Reproduction procedure](../development/pilot/README.md): preparation, environment and execution.

The cohort contains 12 arrangements and 108 queries, split by arrangement into calibration and pilot evaluation.
Each partition has six arrangements and 54 queries.
The main assistant assigned support labels before task predictions, using one non-blind AI review.
The audit retained four disputed cases. It does not establish human ground truth.

Qwen3.5-4B and SmolVLM2-2.2B-Instruct each completed 216 model evaluations.
Each evaluation scores three complete answer continuations under one prompt.
The two prompts are the base prompt and the place-first prompt.
All 432 evaluations completed without a recorded call failure.
The pinned revisions and processor settings are in the run records.

The pilot uses 48 image-gate settings and four confidence thresholds.
Calibration requires useful yield within 0.10 of the base and both class resolutions at least 0.50.
These are development guards. They are not justified application requirements for a future test.
All candidate results remain available. No constraint changed after predictions.

## Main result

Both models chose REMOVED on every input under both prompts.
The following table therefore applies to both model families.
Rates average within each eligible arrangement, then across arrangements.
The raw support denominator excludes the one disputed evaluation query.

| Arm | Unsupported, raw | Useful, raw | Unsupported rate | Useful yield |
| --- | ---: | ---: | ---: | ---: |
| Base with UNKNOWN | 43/53 | 10/53 | 81.02% | 18.98% |
| Place-first prompt | 43/53 | 10/53 | 81.02% | 18.98% |
| Image-only verifier | 0/53 | 38/53 | 0.00% | 71.99% |
| Oracle support veto | 0/53 | 10/53 | 0.00% | 18.98% |
| Always UNKNOWN | 0/53 | 0/53 | 0.00% | 0.00% |

Neither model produced a supported PRESENT answer.
No full-veto setting or confidence threshold met the present-resolution requirement.
Those arms failed calibration. The removal-only arm also has no selected operating point.
Their evaluation results are unavailable, rather than fabricated from a relaxed rule.
No full-veto contrast or interval supports a method claim.

The image-only setting is quality 0.3, presence match 0.7, absence match 0.6 and obstruction 0.4.
It answered all 28 supported-present and 10 supported-removed evaluation queries correctly.
It returned UNKNOWN on both members of all seven unresolved evaluation pairs.

Coverage was 38/54. Treating the disputed case as UNKNOWN gives 0/54 unsupported and 38/54 useful answers.
Only five evaluation arrangements have a supported-removal denominator. The report retains this distinction.

The image-only result is a warning about attainable benefit in this small cohort.
Its usefulness cannot be improved on these support-labeled inputs by adding a language-model veto.
It does not establish zero risk, broad robustness or an unnecessary role for language models in other settings.

## Separate diagnostic after the pilot

The diagnostic uses calibration rows 72, 74 and 77 from TD5013.
They show a visible mug, a clear removal and a removed state behind a blocker.
The diagnostic plan precedes its execution. It does not replace the pilot or select a new official prompt.

For both models, the prefix-only next-token distribution exactly matched each teacher-forced distribution on these rows.
The main reviewer independently compared all 18 distribution pairs and the corresponding saved pilot scores.
This found no causal-scoring discrepancy on the selected inputs.

Ordinary deterministic generation with the exact pilot prompt also returned B on all three rows.
Both models returned B when the reference and query showed the same retained target.
Both single-image descriptions recognized the red cup or mug.
Thus, successful numeric scoring and image loading did not qualify this prompt as a useful verifier.

The separate literal question used full labels instead of answer codes.
SmolVLM2 still returned REMOVED on all three rows.
Qwen first reached the 48-token limit without final labels.
The diagnostic retained those outputs and extended only those three completions to 192 tokens.
Templates, input IDs and image tensors remained identical.

The completed Qwen answers were PRESENT, REMOVED and REMOVED, respectively.
The blocked answer is unsupported under the recorded UNKNOWN label.
This suggests prompt and response sensitivity. It does not isolate answer codes as the cause.
It also does not establish a general weakness of either model family.

## Implementation review

The review corrected image stretching, source metadata leakage risk and reuse of higher-resolution evidence.
The final image front end uses unmarked 1200-by-675 references with scaled boxes.
The model receives a red reference marker as a presentation aid.
The same underlying resized RGB supports both interfaces.
The initial marked-reference evidence run remains as a superseded artifact.

The scoring implementation derives each complete continuation from the model's assistant template.
It handles SmolVLM2's leading space and disables Qwen's thinking mode for this recorded comparison.
It checks the processed assistant prefix and scores the answer plus its terminator.
The confidence control uses normalized candidate scores, not calibrated probabilities.
All vetoes reuse the same cached base answers.

The main reviewer independently recomputed candidate probabilities, calibration choices, evaluation metrics and disputed-label sensitivity.
The checks also enforce the veto's useful-yield ceiling.
Evidence is in [Qwen verification](../runs/2026-11-16-support-audit-v1/qwen_parent_verification.json) and [Smol verification](../runs/2026-11-16-support-audit-v1/smol_parent_verification.json).
The [diagnostic verification](../runs/2026-11-16-support-diagnostic-v1/parent-verification.json) records the separate causal and completion checks.
The implementation has 14 passing tests. All 73 historical frozen files retain their hashes.

Qwen's median later three-candidate evaluation took 2.66 seconds. SmolVLM2's took 1.87 seconds.
These measurements include scoring under the recorded implementation, not an optimized deployment.
Image-evidence elapsed time was not recorded. End-to-end latency remains unavailable.

## Next design decision

1. Qualify the verifier on same-view, clear-present and clear-removed controls before fitting another gate.
2. Compare literal labels and answer codes under controlled wording on development inputs.
3. Define reasoning, response parsing, truncation and confidence scoring together before another pilot.
4. Obtain independent human support labels without access to model outputs.
5. Use a denser calibration-only confidence search before claiming an advantage over rejection.
6. Define a physically justified camera-displacement study and retain every declared severity.
7. Reassess whether the language model adds value beyond image-only verification.

Blocked cases must remain visible in the results. The capability checks must not select a model merely to create favorable gate errors.
Do not expand the present cohort just to obtain a favorable result.
If image-only verification remains sufficient, withdraw the language-model method claim or define a distinct question.
A new test still needs fresh arrangements, human label validation, a contribution check and a justified precision calculation.

## Later interpretation, 2026-11-17 UTC

The failed verifier still requires qualification before another model comparison.
The next experiment now stresses the image-only verifier first, under the [development protocol](../development/stress/PROTOCOL.md).
The [research correction](2026-11-17-research-and-stress-plan.md) separates agreement veto from a conditional cascade.
It also corrects threshold provenance and the later synthetic-yaw interpretation.
The original pilot outputs and provisional AI labels remain unchanged.
