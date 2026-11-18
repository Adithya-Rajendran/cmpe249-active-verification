# Development studies

Start with the [current method](../docs/METHOD.md) and [evaluation plan](../docs/EVALUATION_PLAN.md).
The active scope uses Isaac Sim and MuJoCo. No confirmatory test is frozen.

| Folder | Status and purpose |
| --- | --- |
| [cross_sim](cross_sim/PLAN.md) | Next work: qualify the MuJoCo arrangement adapter, then compare fixed controls across simulators |
| [stress](stress/PROTOCOL.md) | Completed Isaac camera-displacement study and its fixed operational protocol |
| [pilot](pilot/README.md) | Completed first model pilot, retained for reproduction and diagnosis |
| [label_audit](label_audit/) | Builders for the first input and support-label audit |
| [diagnostic](diagnostic/) | Literal-generation diagnostics for the first model interface |
| [real_pilot](real_pilot/PLAN.md) | Superseded plan pointer, retained for earlier links |

The [completed scoring review](../notes/2026-11-17-scoring-and-label-review.md) records results and label limitations.
The [research review](../notes/2026-11-17-va-tamp-and-simulation-scope.md) records the current contribution boundary.
The support-based model comparison remains on hold. A basic AI reviewer check does not validate the research endpoint.

Keep raw outputs in `runs/` and datasets in their configured data directories.
Those directories are excluded from Git. Dated notes retain evidence paths and interpretation.
Do not move historical source files merely because a newer experiment supersedes them.
Their tests, imports and saved run records depend on those paths.

Use a new study identifier and output directory for each revised experiment.
Keep simulator state and masks separate from ordinary verifier inputs.
Follow the [freeze policy](../docs/FREEZE_POLICY.md) before any confirmatory collection.
