# Isaac Sim and MuJoCo development plan

Adopted 2026-11-17 after the researcher selected simulation-only evaluation.
Both engines completed qualification and the paired pilot on November 17.
Read the [pilot results](../../notes/2026-11-17-cross-sim-pilot.md) for observed counts and the engineering decision.
The [RGB geometry comparison](../../notes/2026-11-18-rgb-geometry-comparison.md) ended on November 18 after two new false removals.
The original pilot design and [comparison rules](#rgb-only-alignment-comparison) follow.
The [research review](../../notes/2026-11-17-va-tamp-and-simulation-scope.md) records the rationale.

## Scope and question

Does a fixed image-only verifier preserve correct decisions across Isaac Sim and MuJoCo under controlled viewpoint changes?
Use shelf and tabletop environments in both engines.
This study tests rendered observations. Physical-camera transfer remains future work.
The first pilot uses static scenes. It does not test contact dynamics or manipulation success.

The ordinary input remains one reference RGB image, its target box and one query RGB image.
Use each engine's own nominal reference with that engine's query.
Do not compare an Isaac reference directly with a MuJoCo query in the primary experiment.
Hide simulator identity, state, masks, poses, filenames and pair membership from ordinary verifiers.

## Backend work before scoring

The separate development adapters now read the same textured shelf and tabletop arrangements.
Both passed the [dated qualification](../../notes/2026-11-17-cross-sim-qualification.md) on November 17, 2026.
Use the [capture checks](QUALIFICATION.md) and the declared limits for future adapter changes.
The legacy shelf renderer has a separate interface.

The adapter must preserve target identity, mesh scale, transforms, background geometry and camera calibration.
It must support the same target and blocker interventions as Isaac Sim.
Use matching texture assets where supported. Record asset hashes, UV conventions and material conversions.
Record lighting and renderer differences. Equivalent geometry does not require identical RGB values across engines.

Pin actual installed versions and dependencies before rendering.
Keep GB10 dependency changes in the project uv environment. Preserve Python 3.13.5 and the approved Isaac environment.
Do not copy current online documentation assumptions into an older runtime without a compatibility check.

## Terra qualification

Terra implements and operates the development adapters under main review.
The main reviewer checks each engine's rendered evidence before authorizing the paired pilot.

Use two declared development arrangements, one per environment, for the first smoke.
Include all five stops, D0 and D6, and all four query states defined below.
This gives 80 queries and ten references per engine.

Before task scoring, the main reviewer must accept these checks:

1. Compare compiled object transforms, camera transforms and projected landmarks against the shared manifest.
2. Check target identity, texture orientation, reference boxes and image dimensions visually.
3. Check clear presence, clear removal and matched blocker placement in both engines.
4. Check renderer-specific masks against an independent geometric projection.
5. Repeat captures to detect stale render buffers and order-dependent artifacts.
6. Demonstrate correct resume and refusal of changed inputs or reserved test identifiers.
7. Save versions, settings, source hashes, input hashes, output hashes and every failed check.

Declare numeric projection tolerances and mask-edge handling before this smoke.
Do not require mask identity across different rasterizers.
A failed check requires an adapter correction and a fresh qualification record.
These smoke arrangements remain development data and stay outside the paired pilot below.

## Paired pilot after qualification

Use ten fresh base arrangements, five per environment, with all five camera stops.
Use D0, D4 and D6 from the completed stress study's coordinate and sign convention.
D4 means 4 cm and 2 degrees. D6 means 6 cm and 3 degrees.
Use a common sign assignment across engines and states.

Balance sign combinations across arrangement-stop units, with count differences of at most one per environment.
Pin the full assignment and reserve list before rendering. Do not choose signs from model outcomes.

Render one nominal clear reference per arrangement and stop in each engine.
Render these four query states at each level:

| Query state | Target | Blocker |
| --- | --- | --- |
| Clear present | Present | Parked |
| Clear removed | Removed | Parked |
| Placed present | Present | Placed |
| Placed removed | Removed | Same placed pose |

Each engine supplies 50 references and 600 queries. The paired pilot supplies 100 references and 1,200 queries.
These are ten independent arrangements, not twenty. Engine copies do not increase the arrangement count.
Keep camera pose, lights, background and blocker placement fixed within each matched query group.
Record every intervention in evaluator-only metadata.

Use 2304 x 1296 renders and 1200 x 675 primary inputs.
Retain the fixed primary guards of 100 and 200 pixels and the native guard of 192 pixels.
Keep the existing evidence thresholds and all other effective settings fixed across engines.
Record every preprocessing and effective-configuration hash.
No language-model arm belongs to this first pilot.

## Labels and evidence records

Use simulator interventions for physical state. Export masks as separate geometric diagnostics.
Keep AI support judgments separate, with input hashes, cue reasons and disagreements.
A mask threshold does not certify support for a definite answer.

For a constructed unresolved control, test exact equality of all supplied inputs across opposite physical states.
Such equality certifies ambiguity only for that matched control.
Keep controls without that certificate outside the certified subset. Preserve them in physical-state scores.
Do not relax equality to a chosen pixel threshold after examining outcomes.

Record the verifier's estimated alignment validity, quality, target match, obstruction and refusal reason for each decision.
Keep those estimates separate from simulator diagnostics.
The [method](../../docs/METHOD.md) defines the evidence requirements.
The existing support-validation hold remains active. This pilot does not assign the original stress outcome A, B or C.

## Scores and decisions

Report physical errors, coverage, correct definite yield and clear-state resolution for each engine, environment and level.
Count UNKNOWN against coverage and clear-state resolution.
Show both members of placed pairs, including both-UNKNOWN frequency.
Use arrangement summaries and paired engine differences. Do not treat views or engine copies as independent samples.

Stop the added veto/cascade claim on this axis if one fixed control meets both requirements:

- At least 90% correct resolution in each clear class, for every engine, environment and declared level.
- No observed physical error across the complete pilot.

This is an engineering stop rule. It does not establish supported correctness on every blocked view or a population risk bound.

A wrong definite answer in any state can identify a failure. An unresolved clear control can also identify a failure.
Report abstention on the placed-blocker states as a separate outcome.

Count each distinct failure pattern within one engine, environment, level and state.
The same pattern must recur in at least three common arrangements across all three fixed controls.
Use the observed verifier gate to describe a refusal. Check the evidence before assigning a physical cause.

An effect seen in only one engine needs a renderer-specific explanation before any transfer claim.
An inconclusive pilot remains inconclusive. Do not add arrangements until a desired result appears.

## RGB-only alignment comparison

Only MuJoCo met the pilot recurrence rule. All three affected tabletop arrangements used the same tuna-can asset.
At the corresponding stops, the 200-pixel guard resolved two of the three clear removals in Isaac.
The new experiment tests whether a geometry estimate from the two RGB images improves target alignment.
The [November 18 protocol](geometry_protocol.json) fixes the input hashes, model, visibility rule, thresholds and acceptance criteria.

1. Compare the primary 200-pixel guard with the geometry stage from GeoSCD.
2. Use the same reference image, reference box and query image on all 1,200 pilot cases.
3. Pin the VGGT-1B checkpoint, upstream revision, precision, coordinate transforms and validity rules before scoring.
4. Estimate depth, camera pose and visible overlap from the two RGB images.
5. Map query pixels into reference coordinates with the predicted correspondences.
6. Pass the registered image and valid-pixel mask to the existing evidence calculations.
7. Preserve the target matcher, quality score, obstruction calculation and decision thresholds.
8. Return UNKNOWN if the map fails its declared visibility or validity rule.
9. Qualify the mapping direction, image padding, missing pixels and repeatability on separate fixtures.
10. Check the original registration path through the same interface against its saved decisions.

The original interface accepts one homography. The [geometry adapter](geoscd_geometry.py) computes one correspondence per reference pixel.
It interpolates displacements from the model grid. Each contributing grid point must have a usable projection.

The verifier requires usable projections for at least 60% of the reference target box.
This engineering threshold uses the complement of the existing 40% obstruction ceiling.
Predicted overlap measures the model's estimated coverage of the box.

The authors' geometry stage predicts correspondences and occlusion. Their complete change detector also uses SAM.
The adapter retains subpixel projections. It gives the scorer both linear and nearest-neighbor image samples.

The [shared scorer](geometry_run.py) uses the original appearance, quality and obstruction calculations.
The replay on Ascent matched every decision and every evidence score on all 1,200 original baseline cases.
Use the [pinned geometry implementation](https://github.com/ZilingLiu/GeoSCD/blob/dd31369654e96d6843cc4bbcebce854a8bc2159b/src/pixel_match.py) and the [pinned VGGT checkpoint](https://huggingface.co/facebook/VGGT-1B/tree/860abec7937da0a4c03c41d3c269c366e82abdf9).

Count correct clear removals on the MuJoCo tabletop at D6. The baseline resolved 16/25.
Require a higher total with a net gain in at least two arrangements.
Require at least 17/25 on the corresponding Isaac cell.

Require zero newly wrong definite answers across all cases.
Require UNKNOWN on both members of all 40 certified ambiguous pairs.
Report paired gains and losses in every cell, including clear presence and the other camera displacements.
Keep simulator poses and masks in the independent evaluation only.

A miss ends this adapter revision. A pass supports a development replication on fresh arrangements balanced by target identity.
This rule follows inspection of the pilot. A paper claim needs a separately fixed evaluation on fresh arrangements.

The [paired evaluator](geometry_evaluation.py) compares answers after the complete run.
It checks the saved inputs, both sets of case records and the dated release.
The [landmark diagnostic](geometry_residuals.py) measures target-position errors only where each method supplies a map.

### Completed decision on November 18

The geometry verifier resolved all 25 clear tabletop removals at D6 in each simulator.
It abstained on both inputs of all 40 certified ambiguous pairs.
It also declared a present shelf target removed in two related views. The 200-pixel control correctly retained it in both.
These new errors ended this revision under the declared rule.
The [comparison report](../../notes/2026-11-18-rgb-geometry-comparison.md) gives the paired outcomes, target-position errors and case images.

## Visual support on partly occluded views

The [blind label pilot](support_labels/PROTOCOL.md) used four fresh arrangements in both simulators.
Terra identified all four retained objects per engine. It abstained on all four partial removals per engine.
Both engines failed the qualification rule. The [case review](../../notes/2026-11-18-support-label-qualification.md) includes the matched images and all independent readings.

The [matched instruction comparison](localization/PROTOCOL.md) used four new layouts and 16 blind Terra sessions.
The two-step readers recovered two partial removals in MuJoCo and zero in Isaac Sim.
Only MuJoCo passed its screen. The tabletop sample also lacked a partial retained target at the original camera pose.
The rule ends this revision. The [casebook](../../notes/2026-11-18-label-localization.md) gives each reading and its evidence.

<a id="next-development-experiment"></a>

### Fixed-camera calibration

The [paper protocol](../paper_audit/PROTOCOL.md) fixed the patch rule, contrast threshold and minimum yield.
Four fresh layouts used the same panel across target states and repeats.
In each engine, the patch detected all 20 clear changes. None of the 132 eligible negative controls triggered.

The shelf can missed the required partial yield in both engines.
The [calibration measurements](../../paper/evidence/calibration.json) retain all pairs and the reasons for missing coverage.
General support judgments for partly hidden regions remain open.

## Paper evaluation after development

The [prospective paper comparison](../paper_audit/PROTOCOL.md) uses 16 fresh arrangements.
It reports physical decisions and exact-input ambiguity under fixed settings.
The following requirements concern a later confirmatory study.

Choose the primary mechanism and credible change-detection comparator from development evidence.
Keep method and calibration choices fixed before transfer evaluation.
Reserve fresh base arrangements and occluder identities, including their derivatives in both engines.
Seeing MuJoCo development images prevents a claim that the entire engine was unseen during development.
Describe the later evaluation as transfer on fresh scenes with fixed settings.

Choose final sample size from arrangement-level precision requirements. The pilot counts are not the final study size.
Report all declared engines, controls, ablations, failures and intervals.
Use programmatic support certificates or an independently validated reference for any support-based paper endpoint.
Human review can contribute, but the plan does not require physical images or human labels as the only validation route.

A later active-view method needs a separate protocol, useful decision headroom and equal view and movement budgets.
It must observe only requested views and use one stopping rule across policies.
This document does not start that extension or a confirmatory test.
