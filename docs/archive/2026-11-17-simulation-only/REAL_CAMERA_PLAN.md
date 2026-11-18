# Real-camera baseline check

Adopted next development experiment, 2026-11-17. Collection and scoring remain pending.
All counts below are design choices, not results.

## Question

Does camera return in real scenes expose a repeatable failure beyond the declared image-only controls?
This is an engineering transfer check. It does not test the original unsupported-answer claim.

The completed simulator study separates physical truth from AI support judgments.
The next check uses recorded physical interventions for its primary scores.
It keeps visual support judgments separate.

## Cohort and captures

Use ten fresh physical arrangements, with five per environment.
Keep all five camera stops. This retains opportunities for partial occlusion.
Use the same target and background placement across each arrangement's states.

At each stop, capture one clear reference and these four query states:

| Query state | Target | Blocker |
| --- | --- | --- |
| Clear present | Present | Parked |
| Clear removed | Removed | Parked |
| Placed present | Present | Placed |
| Placed removed | Removed | Same placed pose |

The design contains 50 references and 200 queries.
The clear-removed state supplies an explicit absence control.
The placed states permit partial and heavy occlusion across the five stops.
Record the observed mix before model predictions.

Return the camera once before each four-state query sequence.
Keep that camera pose fixed within the sequence.
Keep lighting, focus, exposure and white balance fixed within each arrangement.
Record interventions separately from model inputs.

Before task capture, measure ten camera returns and document the measurement uncertainty.
Complete the capture manifest, object list and validity rules before scoring any task image.
Check the shared camera before capture. Use the existing capture environment.
Do not modify the separate visual-memory project.

## Inputs and controls

Use the existing 2304 x 1296 capture size and 1200 x 675 primary preparation.
Use the clear reference from the nominal camera pose and its target box.
Give each call only its reference, box and one query.
Hide intervention records, other states, poses and support judgments.

Run these fixed image-only controls:

1. Primary resolution with a 100-pixel shift guard.
2. Primary resolution with a 200-pixel shift guard.
3. Native resolution with a 192-pixel shift guard.

Retain the current evidence thresholds and all other effective parameters.
Record every configuration. Do not select settings from these outcomes and call them held-out results.
No task language model belongs to this experiment.

## Labels and scores

Record physical state from the intervention procedure.
Check image identity, target identity and intervention records before scoring.
Use the qualification instructions for a separate blind AI cue review.
Preserve disagreements and raw labels. Do not merge AI labels into physical truth.

Report physical errors, coverage, correct definite yield and clear-state resolution.
Correct definite yield divides correct definite answers by all scheduled queries.
Count UNKNOWN against clear-state resolution and coverage.
Report the placed pairs' outputs together, including both-UNKNOWN counts.
Do not assume every placed view is unanswerable.

Keep AI support metrics secondary and descriptive.
Do not treat correct guesses on blocked inputs as verified support.
The original stress A/B/C rules remain unavailable without their required support reference.

Use the arrangement as the analysis unit.
Report each environment separately, with raw denominators and arrangement summaries.
This small pilot does not establish a 1% error bound.

## Development decision

Stop the camera-return veto/cascade branch if one declared control meets both conditions in each environment:

- At least 90% correct resolution for both clear PRESENT and clear REMOVED inputs.
- No observed physical error across the complete four-state cohort.

This is an engineering stop rule for the observed pilot. It does not prove general equivalence or visual support.
Do not launch a language model merely because some placed views remain UNKNOWN.

A repeated failure needs the same observed failure mechanism in at least three arrangements within one environment.
Require that failure to persist across the applicable image-only controls.
Such a result motivates one declared comparison against a stronger geometric baseline.
It does not automatically authorize a support-based cascade claim.

If neither condition holds, record the pilot as inconclusive.
Do not add arrangements until a desired result appears.

## Scope

This plan changes the order of development. It does not change the completed simulator protocol.
All data remain development data. No confirmatory freeze exists.
Independent support validation and a suitable modern baseline remain requirements for a paper claim.
