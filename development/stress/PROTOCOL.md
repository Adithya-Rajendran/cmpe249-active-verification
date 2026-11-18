# Camera displacement pilot

Version: `support-stress-dev-v1`, 2026-11-17 UTC. This is development data.
The researcher authorized research, corrections and Terra execution on Ascent.
This specification precedes new task predictions. Changes require a dated amendment and a new manifest hash.

## Question and sequence

Does camera displacement cause unsupported image-only answers, or does it only reduce useful answers?
Run geometry validation first. Then run a smoke render and an independent review.
Expand to the declared cohort only after that review passes.

Run image evidence and prepare blind labels. Do not run a task language model in this stage.

## Cohort

- Generate 12 fresh arrangements per environment: shelf and tabletop.
- Use all five stops, lighting L1, and visible, occluded and removed query phases.
- Keep the occluder, distractors, lighting and query camera identical within each occluded/removed pair.
- Keep each base arrangement together across all levels and later data partitions.
- Use two slots per class: S8, M16-L, M16-R, IND28, white_box and ycb_box.
- Assign targets from the first two development slots of each class in the existing configuration.
- Generate new placements. Do not rename old layouts or use reserved test seeds.
- Reserve seeds 8000–8383 for shelf and 8500–8883 for tabletop, subject to an existing-artifact collision check.
- Allocate 32 consecutive seeds to each slot. Try attempts 0 through 20 in order.
- Stop each slot search after 32 valid candidates or exhaustion of its seed block.
- Require every candidate to pass the target and workspace frame checks at every specified pose.
- Rank those candidates by the number of nominal stops with analytic obstruction strictly between 0.3 and 0.9.
- Select the highest count. Break ties by seed, then attempt. Record every rejection and candidate.

The target is at least 20% middle-band stops among placed-occluder views at level zero, separately by environment.
This enriches partial occlusion deliberately. It does not estimate natural scene prevalence.
Analytic hull coverage selects the cohort before rendering. Render masks report the achieved mix separately.

The [first amendment](../../notes/2026-11-17-stress-amendments.md) increased the candidate limit after a geometry-only quota failure.

If the analytic target fails, stop before rendering and record a geometry-only amendment. Do not silently change the quota.
Never replace a rendered case because its evidence, model output or human label is inconvenient.

## Motion

| Level | Translation | Yaw |
| --- | ---: | ---: |
| D0 | 0 cm | 0 degrees |
| D1 | 1 cm | 0.5 degrees |
| D2 | 2 cm | 1 degree |
| D4 | 4 cm | 2 degrees |
| D6 | 6 cm | 3 degrees |

Let the nominal heading be `h`. Set horizontal camera-right to `r = (cos(h), -sin(h), 0)`.
Set the position increment to `s_translation * translation * r`.
Set the heading increment to `s_yaw * yaw`. Keep pitch, roll and height unchanged.

Use the frozen `perturb_camera` function. Check actual target hulls with the frozen `frame_check` function.
Use the fitted distortion in the conservative frame check. Render and diagnostic projections use pinhole cameras.

At each stop in each environment, assign three arrangements to each sign pair: `++`, `+-`, `-+`, `--`.
Use NumPy PCG64 seed 20261117, with environment index and stop index as SeedSequence components.
Permute the repeated ordered sign list separately for each stop. Save the complete sign table before generation.

Keep each sign pair unchanged across levels and query phases. Do not redraw a sign after a failed geometry check.
Report adding and cancelling sign groups separately, alongside the balanced result.

## Inputs and fixed settings

Render at 2304 x 1296 with the existing Isaac Sim 6.2 renderer and its default capture settings.
Use L1 only. Keep the historical renderer and every file in FREEZE unchanged.
Record source hashes, environment versions, assets, settings, commands, start times and exit codes.

Use new arrangement identifiers, including a base identifier and level, with separate output folders.
The planned total is 120 arrangement-level renders and 1800 query views. This is a design count, not a result.

Primary evidence inputs are unmarked RGB images at 1200 x 675, resized with Pillow LANCZOS.
Scale reference boxes with the v1 pilot rule: round each coordinate after multiplying by the size ratio.
Strip PNG metadata. Use neutral input names. Keep physical labels and scene data outside the verifier directory.

Use each base arrangement's D0 reference image and box at every displacement level.
Retain other rendered references for integrity checks. Do not introduce reference-render variation into the paired severity comparison.

Human reviewers see the same pixels and a reference-box overlay. The query has no overlay.

Use evidence v0.3 with image alignment, default `Params` and `V03Params`, and these `AlignParams`:

| Parameter | Value |
| --- | ---: |
| sigma | 1.5 |
| levels / stop_level | 4 / 1 |
| excl_f / border | 0.5 / 24 |
| iters / eps | 200 / 0.000001 |
| min_cc | 0.5 |
| max_shift_px | 100 |
| max_scale_dev | 0.10 |

The decision thresholds are quality 0.3, presence 0.7, absence 0.6 and obstruction 0.4.
These four thresholds came from v1 calibration on resized inputs. The alignment settings came from the earlier implementation.
Serialize every effective parameter and checkpoint hash in the run manifest. Refuse mismatched resumes.

Keep these additional controls separate from the primary result:

- Native 2304 x 1296 inputs with the unchanged 100-pixel guard.
- Native inputs with a 192-pixel guard, matching the primary guard's fraction of image width.
- Primary-size inputs with a 200-pixel guard, as a declared relaxed-guard diagnostic.

Do not choose a winning guard from evaluation outcomes. A later calibrated comparator needs its own development split.

## Alignment diagnostics

Save the estimated warp before rejection, when optimization returns a finite fit.
Keep the ordinary acceptance decision and its reason. Do not insert a true warp into the verifier.
Report optimizer failure, correlation refusal, shift refusal and scale refusal separately.

Flag large residuals separately from these refusal reasons. A residual is not an additional deployed guard.

Use corresponding world points projected at both actual camera poses.
Report background-plane transfer error and target-point residual separately.
For shelf diagnostics, use the front plane and target-depth plane from `sim/alignment.py`.

For tabletop diagnostics, use the table plane and target-height plane from that module.
These are declared plane references, not a claim that one homography fits every background depth.
Use the reference-box corners and centre projected through each plane homography as diagnostic points.

Convert half-pixel conventions explicitly. Record the direction of every warp.
Use 5 pixels at width 1200 and 9.6 pixels at width 2304 for descriptive residual flags.
Do not infer support from a small residual or an error from a large target residual alone.

## Labels and analysis

The researcher authorized AI and programmatic development labels after the smoke run.
Follow [the label supplement](LABEL_PLAN_AI_V1.md). The original JSON remains the provenance of the geometry and render settings.

Generate complete geometry-proxy labels and a separate blinded AI audit.
Store each source explicitly. Geometry alone does not establish visual answerability.
Use PRESENT, REMOVED, UNKNOWN or DISPUTED for the AI support judgment.

PRESENT requires identifying evidence. REMOVED requires a sufficiently exposed remembered region with distinguishing absence evidence.
Fragments, shadows and reflections count only when they support identification from the supplied pair.

Record the cue and uncertainty. Do not infer a label from the physical intervention or another query.
Randomize neutral IDs. Hide predictions, physical state, geometry, level and pair membership during labeling.
Keep missing labels missing. An AI judgment must never appear as a human label.

Use the supplement's fixed 180-query sample for the full AI audit.
Report agreement and disagreements with the geometry proxy after the blinded review.
Independent validation remains a paper requirement. Repeated AI judgments do not establish human agreement.

Keep AI-reference, geometry-proxy-reference and human-reference metrics separate.
Repeat support metrics with DISPUTED treated as UNKNOWN.
Keep disputed and unlabeled inputs in coverage and physical-state counts.
Report review judgments that conflict with physical state. Keep those judgments unchanged in the label record.

Secondary-resolution support comparisons require labels at that resolution. Do not silently reuse primary-size answerability labels.

Use the metrics in `docs/EVALUATION_PLAN.md`, separately for each environment and level.
Average within arrangement, then across arrangements. Report raw counts and eligible denominators.
Use 2000 paired arrangement bootstrap draws, seed 20261117, for differences from D0.

Carry all stops, phases and levels with their arrangement. Do not treat views as independent samples.
Report the achieved obstruction bands and sign groups without altering their weights after results.

## Development decision

Apply these rules to complete primary-size support labels, separately by environment.
AI-only decisions are provisional development triage. A geometry proxy cannot establish a support-based go decision.

| Outcome | Rule | Next action |
| --- | --- | --- |
| A | Unsupported rate exceeds 0.01 in at least three arrangements at one nonzero level. | A qualified model can test agreement veto. |
| B | Useful yield falls by at least 0.10 from D0 at D6, with no outcome A. | Drop veto as the main claim. Assess recoverable abstentions. |
| C | Neither rule holds. | Stop expansion on this simulation axis. Assess a small real development set. |

Outcome A must hold under both the primary labels and the DISPUTED-as-UNKNOWN analysis.
A recoverable useful answer requires a definite support label that agrees with physical state.
A cascade additionally requires answerable image-only abstentions covering at least 0.10 of support-labeled inputs at the affected level.
Check whether a credible image-only guard control removes that opportunity before crediting a model.

Do not start the task-model stage from geometry labels or alignment refusal alone.
These are development triage thresholds. They are not confirmatory margins or safety guarantees.

## Terra execution check

Terra owns simulator execution on Ascent. The main reviewer owns scientific decisions and output acceptance.
First render the first selected arrangement from each environment at D0 and D6.
Check every stop and query phase, source hashes, masks, cameras and matched scene interventions.

Rerun one smoke input with validated resume. It must preserve complete outputs and refuse changed input provenance.
Test invalid study IDs and reserved seeds before simulator startup.
The reviewer inspects actual images and independently recomputes geometry and artifact checks.

Only a passing review permits the remaining cohort render.
Passing establishes suitability for this bounded simulator role. It does not establish general scientific autonomy.
