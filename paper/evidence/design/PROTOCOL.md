# A counterfactual audit of two-image robot memory updates

## Scope and release

This is a prospective development evaluation named `paper-audit-v1`. Inspect no final-test geometry. Keep the historical freeze intact.
The question is whether deterministic verifiers distinguish removal from occlusion when they revisit a remembered place.
Measure both definite commitments on identical inputs and correct removal decisions on clear views after camera motion.
Lock this document before generating the new arrangements. Record operational source hashes before each stage.
Date each correction. State which inputs or outcomes were visible at that time. Retain all prior receipts.
The record is local. It has no independent timestamp.

## Physical design

Use five camera stops, L1 lighting, and one nominal clear retained reference per stop and engine.
Each query crosses target retained or removed with blocker parked or placed. Record all four interventions.
Reuse the same physical layouts and camera transforms in Isaac Sim and MuJoCo. Keep each layout together in analysis.
Capture native RGB at 2304 by 1296. Treat both engines as paired measurements of one layout.
Use 20 fresh base layouts from disjoint 32-seed blocks. Block i starts at 3200 + 32i, counting from zero.
Exclude all prior development seed reservations, used seeds, exact canonical object-layout duplicates, and throwaway fixture seeds.
Do not apply a visual similarity filter. Check existing reservation metadata without opening reserved test layouts.
Within a slot, take the first valid seed and attempt in ascending order. Use attempts 0 through 20 inclusive per seed.
Require the generator's geometry constraints and its nominal and moved frame-margin checks.
Log every failed attempt. Record an exhausted slot as missing. Preserve all valid slots.

For fixed-camera calibration, order the first four slots by environment, then target.
Use shelf and tabletop, each with target 6 and M16-L or target 21 and M16-R.
Targets are the YCB tuna fish can and foam brick. Use nominal cameras only.
At each intervention and stop, render a base, two fresh repeats, and a distractor-only change.
Hide distractor_0 for that change. Record which object moved and whether its removal measurably changed the image.
Use the same target and blocker state throughout each negative control.

For evaluation, order 16 slots by environment, target, blocker family, then replicate, in that sequence.
Use shelf then tabletop, target 8 then 18, IND28 then white_box, and two fresh layouts per combination.
The gelatin box and large marker are new target identities for the two-method comparison. They may occur in training or as distractors.
Capture D0 and D6. D6 applies a 6 cm horizontal translation and a 3 degree yaw about world vertical.
Cycle translation and yaw signs through ++, +−, −+, −− by (5i+j) modulo 4, using split-local layout i and stop j.
The planned denominator is 640 queries per engine. Each engine supplies 80 clear removed queries at D6.

## Information available to verifiers

Give each verifier only the two decoded RGB images, the reference target box, and its fixed task.
Normalize RGB to 1200 by 675 with Pillow Lanczos. Use nearest-neighbor for evaluator masks.
Use the tight, half-open reference box. Scale its native minimum edges down and maximum edges up to enclosing integers.
Randomize opaque case names and processing order. Keep the private mapping outside the packet.
No path, ID, pairing, label, mask, engine, calibration, camera pose, or true query box enters a prediction.
Use deterministic inference. Fail closed to UNKNOWN on a declared abstention. Record execution failures separately and retry technical failures without changing settings.

## Methods

Compare the fixed 200-pixel image-registration baseline and the fixed VGGT registered-place geometry adapter.
Use the unchanged decisions and pretrained weights in the prior scoring and geometry protocols.
Both methods use the same RGB evidence rule after registration. Call an abstention UNKNOWN.
Keep always-UNKNOWN and always-REMOVED as analytic operating points.
Qualify a separate zero-shot registered-place adaptation using the frozen prompt and specification in the scoring release.
This adaptation uses VGGT and Qwen3.5-9B. Its qualification uses only archived controls disjoint from this study.
Require PRESENT on at least 34 of 36 same-view cases and correct answers on at least 36 of 40 cases in each clear class.
If any gate fails, report the failure and stop that adaptation. Do not tune on the gate or apply it to fresh queries.
If it passes, report a secondary comparison on the same inputs. Do not claim reproduction of the trained STILL verifier.

## Independent evidence

For each engine, layout, stop, level, and blocker position, compare the opposite-state inputs.
Hash decoded reference and query pixels including shape. Compare numerical reference boxes and all fixed instructions and fields.
Certify ambiguity only for equality of the entire delivered input. Separately audit native equality.
Any definite answer on such a pair is unsupported by that input under the two interventions.
A deterministic definite answer makes one correct and one wrong factual decision per identical pair.
Report pair and layout denominators, both-UNKNOWN pairs, commitment pairs, and any inconsistent outputs.
An empty certificate set gives an unmeasured commitment rate. Unequal images do not establish semantic answerability.

Use calibration only for a separate fixed-camera, local RGB witness.
Intersect the reference target mask with the retained intervention's exposed target mask at primary resolution.
Find the pixel farthest from the complement using Euclidean distance and an outer ring of false pixels.
Break ties by distance to the reference-box center, then row and column.
Use its centered 7-by-7 patch if wholly in frame and at least 25 pixels belong to the intersection.
Measure mean absolute RGB difference over those pixels and channels, on the 0–255 scale.
Set the noise bound to the larger base-to-first-repeat contrast from the two physical states.
Require target contrast of at least 8 and a margin of at least 4 over that bound.
Apply the same bound to the held-out second repeat and distractor-only change in each state.
Count each intervention pair once. Report eligibility, positives, all negatives, and native retained visibility versus the isolated target mask.
Require at least eight positive pairs among ten clear pairs per engine and environment.
Also require a positive placed pair with native visible fraction strictly between zero and one in every calibration layout and engine.
Describe raster-boundary cases and the magnitude of each distractor manipulation. Report weak or invisible manipulations.
Any triggering negative or missed yield ends the witness claim. Even a pass establishes local image change only.

## Analysis and interpretation

The primary paired method contrast is the difference in correct clear REMOVED decisions at D6, by engine.
Divide by all eligible-by-design clear removed positions, keeping abstentions in the denominator.
The co-reported safety endpoint is the number of certified pairs receiving definite answers, plus the total certified pairs.
For every method, also report factual correct, wrong definite, false removal, false presence, coverage, and abstention.
Break out environment, camera level, intervention, target and arrangement. Count paired gains and losses.
Keep an error on a retained object separate from a correct removal. Report empty sets explicitly.
Resample whole layouts 10,000 times with replacement within each environment, with seed 249.
Keep engines, methods, stops, camera levels and physical states together. Use descriptive 2.5th and 97.5th percentiles for paired contrasts.
Report layout counts. The fixed target panel defines the inference scope.
Call a tradeoff-free improvement only for a strict primary gain in each engine, positive gains on at least two tabletop layouts per engine,
zero newly wrong definite decisions across the full sample, and zero commitments on available certified pairs.
Report an engine with no certificates as unassessed on that criterion. Report each condition even if another fails.
Keep any other contrast descriptive. The paper can report a failed method, a failed witness, or no differences.
No criterion authorizes automatic home-robot updates or generalizes to physical cameras.
