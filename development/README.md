# Development studies

Start with the [current method](../docs/METHOD.md) and [evaluation plan](../docs/EVALUATION_PLAN.md).
The active scope uses Isaac Sim and MuJoCo. No confirmatory test is frozen.

| Folder | Status and purpose |
| --- | --- |
| [Prospective paper audit](paper_audit/PROTOCOL.md) | Fresh counterfactual pairs, fixed-camera calibration, three verifiers and the layout-level comparison |
| [Shelf camera orientation](level_shelf/PROTOCOL.md) | Repeated shelf scenes with the original, edge-leveled and straight-on views |
| [Service corridor transfer](corridor_transfer/PROTOCOL.md) | Eight new floor layouts, two new target objects, and a sliding door in both simulators |
| [Furnished workshop](workshop_realism/DOOR_AMENDMENT.md) | Matched corridors with a sliding door, photographed surfaces and two robot camera mounts |
| [Workshop robot training](workshop_realism/ROBOTICS.md) | Drive a simulated robot in MuJoCo or use the native Isaac Sim facility |
| [cross_sim](cross_sim/PLAN.md) | Qualified adapters, completed paired pilot and completed RGB geometry comparison |
| [Blind label pilot](cross_sim/support_labels/PROTOCOL.md) | Fresh paired scenes, localized evidence checks and exact ambiguity controls |
| [Reader instruction comparison](cross_sim/localization/PROTOCOL.md) | Separate blind readers judged the same images, with or without an initial location estimate |
| [stress](stress/PROTOCOL.md) | Completed Isaac camera-displacement study and its fixed operational protocol |
| [pilot](pilot/README.md) | Completed first model pilot, retained for reproduction and diagnosis |
| [label_audit](label_audit/) | Builders for the first input and support-label audit |
| [diagnostic](diagnostic/) | Literal-generation diagnostics for the first model interface |
| [real_pilot](real_pilot/PLAN.md) | Superseded plan pointer, retained for earlier links |

The [completed scoring review](../notes/2026-11-17-scoring-and-label-review.md) records results and label limitations.
The [paired pilot](../notes/2026-11-17-cross-sim-pilot.md) identifies a recurring alignment refusal on three MuJoCo tabletop arrangements.
The [geometry comparison](../notes/2026-11-18-rgb-geometry-comparison.md) resolved these removals and introduced two new false removals at one shelf view.

Terra abstained on all eight partly occluded removals in the [separate label pilot](../notes/2026-11-18-support-label-qualification.md).
Locating the target first recovered two new MuJoCo removals and zero Isaac removals in the [matched instruction comparison](../notes/2026-11-18-label-localization.md).
The [research review](../notes/2026-11-17-va-tamp-and-simulation-scope.md) records the current contribution boundary.
The support-based model comparison remains on hold. A basic AI reviewer check does not validate the research endpoint.

Keep raw outputs in `runs/` and datasets in their configured data directories.
Those directories are excluded from Git. Dated notes retain evidence paths and interpretation.
Do not move historical source files merely because a newer experiment supersedes them.
Their tests, imports and saved run records depend on those paths.

Use a new study identifier and output directory for each revised experiment.
Keep simulator state and masks separate from ordinary verifier inputs.
Follow the [freeze policy](../docs/FREEZE_POLICY.md) before any confirmatory collection.
