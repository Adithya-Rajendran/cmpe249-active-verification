# Isaac Sim and MuJoCo development plan

Adopted 2026-11-17 after the researcher selected simulation-only evaluation.
All counts and thresholds below are prospective development choices, not results.
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

The current MuJoCo backend renders the legacy shelf rig with flat materials.
It does not load the textured Isaac arrangement cohort.
Implement a separate development adapter for shared scene specifications before comparing scores.
Keep the historical renderers and frozen inputs intact.

The adapter must preserve target identity, mesh scale, transforms, background geometry and camera calibration.
It must support the same target and blocker interventions as Isaac Sim.
Use matching texture assets where supported. Record asset hashes, UV conventions and material conversions.
Record lighting and renderer differences. Equivalent geometry does not require identical RGB values across engines.

Pin actual installed versions and dependencies before rendering.
Keep GB10 dependency changes in the project uv environment. Preserve Python 3.13.5 and the approved Isaac environment.
Do not copy current online documentation assumptions into an older runtime without a compatibility check.

## Terra qualification

Terra implements and operates the development adapters under main review.
Its completed Isaac qualification does not automatically qualify the new MuJoCo adapter.
The read-only source review confirms the implementation gap. It is not a rendering result.

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
A continuing mechanism must recur in at least three arrangements within an environment and level.
It must persist across applicable guard controls before a stronger geometric comparator is justified.

An effect seen in only one engine needs a renderer-specific explanation before any transfer claim.
An inconclusive pilot remains inconclusive. Do not add arrangements until a desired result appears.

## Paper evaluation after development

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
