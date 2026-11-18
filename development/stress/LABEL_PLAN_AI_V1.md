# Development labels: simulator measurements and blinded AI review

Adopted 2026-11-17 after the researcher authorized AI and programmatic labels.
This supplement replaces the requirement to wait for human labels during development.
It does not change the camera cohort, images, verifier settings or historical records.

The original protocol hash remains the provenance of the geometry and render inputs.
Bind new label reports to [this label specification](label_plan_ai_v1.json) as well.
The amendment follows the smoke predictions. It is development, not a prospective test claim.

## Separate the three label sources

| Source | What it establishes | Record type |
| --- | --- | --- |
| Scene intervention | Whether the target exists in the remembered region | Physical state |
| Simulator masks | Target visibility and obstruction of its region | Geometry measurement |
| Blinded AI inspection | A provisional judgment about evidence in the permitted image pair | AI support label |

Geometry does not establish visual identification from a fragment, reflection or shadow.
An AI judgment is not a human label. Keep those sources explicit in every file and result.
Never expose simulator truth, geometry or predictions to the blinded AI reviewer.

## Complete programmatic labels

Generate a geometry proxy for every query at each input size.
Use the paired simulator masks, with their saved hashes.

- Use PRESENT when the target is present and its visible fraction is at least 0.90.
- Use REMOVED when the target is removed and region obstruction is below 0.10.
- Use UNKNOWN otherwise.

These cutoffs reuse the historical diagnostic definitions. They are not newly fitted support thresholds.
Proxy UNKNOWN means that the geometry rule did not pass. It does not prove that the image is unanswerable.
Record the measurement, cutoff, proxy version and exact input binding for each label.

Use the source value `geometry_proxy`. Keep these files outside the verifier input folders.
Report proxy-reference scores separately from AI-reference and human-reference scores.
Do not use a geometry proxy alone to establish a support-based model go decision.

## Blinded AI audit

Use a fresh Terra reviewer for all 60 starter pairs.
Supply only the neutral review pack and the task definitions.
Show the exact primary-size query and the reference with its target rectangle.

The reviewer records PRESENT, REMOVED, UNKNOWN or DISPUTED, with the visual cue and uncertainty.
Store the source as `ai`, the model name, instructions and presented image hashes.
Keep judgments unchanged when they disagree with physical state or the geometry proxy.
Reconcile those differences in a separate audit record after the labels are complete.

For the full cohort, select a fixed 180-query audit before its predictions.
Take six queries from each environment, level and phase stratum.
Each stratum contains 60 queries, so each query has inclusion probability 0.10.

Rank neutral IDs by SHA-256 of `20261117|ID` within each stratum.
Select the first six. Resolve any digest tie by the neutral ID.
Retain overlapping starter labels only when their exact image and box bindings match.

The starter pack is a convenience sample with one base arrangement per environment.
Do not combine all starter labels with the sampled audit to estimate full-cohort performance.
Report the audited counts, strata and disagreements. Keep unreviewed AI support labels missing.

## Interpretation and next action

The geometry proxy supplies a complete development diagnostic.
The AI audit tests where that proxy misses distinguishing evidence or assumes too much.
Expand blinded AI review when those differences could change the next research decision.

Complete AI labels may support a provisional development go or stop decision under the existing numerical rules.
Label that decision AI-only. Require the existing strong-control comparison before model expansion.
An incomplete support audit does not become a full-cohort support result.

The 2026-11-17 full audit exposed disputed removal judgments in a nonblind main review.
See the [recorded disagreements](../../runs/2026-11-17-stress-dev-v1/reviewer/full_ai_label_spot_review.json).

Completeness alone does not qualify these AI labels for triage.
Before expanding AI review, qualify the reviewer with independently reviewed clear PRESENT and REMOVED controls.
Declare the qualification criteria before the check. Preserve the original judgments and record any revision separately.
This quality hold changes no rendered input, existing label or numerical threshold.

Keep the paper requirement for independent validation of support judgments.
That can use a declared human sample or a suitably validated external benchmark.
Specify its sampling, weights and uncertainty before using it for a paper claim.
