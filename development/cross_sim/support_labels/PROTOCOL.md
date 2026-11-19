# Visual support qualification pilot

Adopted November 18, 2026, before selecting images or collecting judgments. This is a development screen.

## Question

Can Terra identify visual evidence when a blocker covers part of a remembered object?
The target has a fixed remembered pose. The only target interventions are retention and complete removal.

Use four new base arrangements, two shelf and two tabletop. Capture both simulators at nominal pose and at 6 cm and 3°.
Use the existing qualified capture adapters. Keep evaluator geometry and all counterfactual images outside the review packets.
Run no task verifier on these inputs.

## Image selection

Use native target and blocker masks to select camera positions before examining judgments.
For each clear view, require at least 64 target pixels and at least 80% of the isolated target silhouette.
Measure the fraction of the clear target mask that remains visible with the blocker placed.
A partial view must retain 10% to 90%, inclusive, in both engines.
In each engine the placed blocker must cover at least 8% of the clear target mask.

Select up to two partial camera positions per base arrangement. Prefer different stops, then different displacement levels.
Break ties by a fixed SHA-256 ordering with the seed `support-label-qualification-v1`.
Use one retained and one removed intervention, assigned in that order to the two hash-ranked positions.
If only one qualifies, alternate its intervention by the arrangement number.

Both engines supply the same physical intervention at a selected position. Keep the opposite intervention private for evaluation.

Select one clear control per base from unused positions. Alternate retention and removal within each environment.
Select one hidden control per base if the target has zero visible pixels in both engines under the placed blocker.
Require the blocker to cover at least 90% of the clear mask in both engines for this control.

Reserve hidden positions before choosing clear controls. Use the opposite state assignment to the clear controls.
Use the same fixed hash ordering within control categories. Preserve shortages and all eligible-location counts.
The design targets 16 partial judgments, eight clear controls and up to eight hidden controls across the two engines.

## Blind judgments

Use each engine's own nominal reference and its verified reference box.
Supply both full 1200 by 675 images and opaque IDs.
Derive the coordinate aid from the reference box alone.
Give fresh Terra sessions disjoint packets. Each session sees at most one case from a base arrangement.
Hide the engine, condition, intervention, weights, other cameras, twins, votes and evaluator files.

Ask the reviewer to name the target, locate its remembered region, describe the evidence and record an answer.
Require a query point on the identified target for PRESENT or on the exposed expected footprint for REMOVED.
Require a location anchor for REMOVED. A nearby blocker or a missing detection alone is insufficient.

UNKNOWN means the supplied images give insufficient evidence. DISPUTED records an unresolved interpretation of the task or image.
Record a reason for either answer. An optional point can locate the obstruction.

Main makes an independent pass on the anonymous images and locks it before opening Terra's labels or evaluator metadata.
Then main audits each Terra answer, its cue, and its location against the full evidence.
Both readers are AI systems. Common model biases can persist even after masking the metadata.

## Evidence checks and decision

The simulator intervention establishes physical correctness. Require three further checks to corroborate a definite support judgment.
First, both readers must independently give the same answer and a specific visible cue.

Second, each reader's query point must fall on evidence for the fixed target pose.
Use the matched retained object's visible mask for either intervention. Resize that mask with nearest-neighbor sampling.
At least half of a seven-by-seven patch centered on each point must fall in this mask.

Third, compare the matched-present and matched-removed RGB values within the masked part of the patch.
Require a mean absolute channel change of at least eight intensity units and at least four above the clear repeat difference.
Use the same point and mask for the repeat measurement. The repeat is a rendering-noise check for the clear state.

Inspect the named cue after unblinding. Record weak identities, unsupported absence claims, physical contradictions and point failures separately.
These checks corroborate a localized cue under the restricted interventions. They are a conservative working reference.

An identical delivered reference, numeric box and full query under opposite interventions certifies UNKNOWN for that case.
Zero mask area alone cannot establish this certificate. Retain other unresolved views and disagreements in every denominator.
Report certificate agreement only where the inputs pass the exact check. Report physical judgments and visual support separately.

For the partial-view screen, each engine needs eight partial cases and four clear controls.
Require corroborated judgments on at least six partial cases and three clear controls per engine.
Require at least one corroborated partial PRESENT and REMOVED per engine.
Any physically wrong definite answer or definite answer on an exact ambiguity fails the screen for that engine.

Incomplete case counts leave that engine's screen inconclusive unless a contradiction already establishes failure.
Passing permits assisted development labeling with main review in these conditions.
The small number of arrangements limits claims about the population. Keep the earlier support-based model decision on hold.
