# Does the hidden-object failure transfer to a service corridor?

The earlier audit found false removals behind shelf blockers. Sometimes the geometry check stopped those answers when the camera roll changed.
This prospective development study tests those observations in a new setting.

## Scene and inputs

Use eight new service corridors with a sliding door, four with a power drill and four with a bleach bottle.
Place the target on the floor beyond the doorway. Change the passage geometry and object positions across layouts.

Use identical scene coordinates in Isaac Sim and MuJoCo. Keep the source shelf and tabletop scenes outside this panel.
Reserve one additional corridor for engineering checks. Exclude that fixture from all reported rates.

In every layout, render all four combinations of target retained or removed and door open or closed.
Use an earlier clear image with a marked target and one later image. Keep the task, image dimensions and three scorers from the paper.
Only the evaluator receives masks, simulator identity, camera poses and intervention labels.

| View | Comparison |
| --- | --- |
| P1, oblique | View the floor from the side. Shift the revisit camera 6 cm and change its heading by 3°. |
| P2, edge leveled | Keep each P1 lens center and viewing direction. Rotate around the optical axis until a transverse floor marking appears horizontal. |
| P3, frontal | Look along the passage from the centerline. Translate the revisit camera 6 cm sideways with zero yaw change. |

Use the nominal reference and the moved-camera revisit for every decision. Each engine supplies 96 queries and 48 opposite-state pairs.
Never select a layout, camera or pair using a verifier answer.

## Checks before scoring

Bind the declaration, scene generator, input manifest and three original scorer definitions to the release.
Verify all eight layouts and their seeds against the earlier evaluation panels and development reservations.
Check requested poses against both simulators. Check identical P1/P2 centers and directions for each camera.

Check the projected marking separately for reference and revisit. Require an absolute angle below 0.05° in P2 and P3.
Require positive reference, isolated and clear retained mask areas. Require empty removed masks.

Check every complete retained/removed input for exact equality at native and delivered resolutions before scoring.
Include every declared query in the physical-state results. Record partially hidden and unequal pairs separately.
Never infer image equality from an empty target mask alone.

## Decisions fixed before predictions

The primary test uses MuJoCo P1 with the door closed. This matches the condition where the shelf finding occurred.
A pair qualifies when the native and supplied RGB inputs, reference box and task match across the two physical states.
The outcome is the registered Qwen system answering REMOVED twice, which falsely deletes the retained target.

- Two or more distinct layouts with that outcome establish recurrence across layouts on this panel.
- One layout establishes a single recurrence here.
- Zero such outcomes with qualifying pairs means the original failure did not recur on the observed opportunities.
- Zero qualifying pairs means the primary test had no opportunity. Report the remaining physical-state results.

Repeat the count separately in Isaac Sim. Report any definite commitment on exact inputs for all three verifiers.
Separate model answers from abstentions caused by the geometry check. Record technical failures separately.

For each primary failure, compare P2 only if its opposite-state inputs also qualify.
Count how many such pairs switch to two UNKNOWN answers at the geometry check. Report every discordant pair.
Also compare P1 and P2 across all layouts, including new errors introduced by leveling.

For P3 with the door open, count false removals on exposed retained targets in each simulator.
Define exposed as overlap with at least 95% of the isolated target mask. Report the eligible denominator.

For each method, engine and view, report clear removal detection, all false removals, and the share of definite answers.
Count UNKNOWN against useful yield. Show both object identities and preserve the layout as the unit of replication.
Treat this as a transfer test on one new scenario family. These scenes change objects, room geometry and occluder together.
