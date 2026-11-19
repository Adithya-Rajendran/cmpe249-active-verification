# Does the hidden-object deletion survive a furnished workshop?

The first corridor audit found seven false deletions on eight exact frontal MuJoCo pairs. This simulation-only rerun tests that finding under richer visual conditions.

## Design

Reuse the eight corridor layouts and both target meshes. Pair each new observation with its earlier layout, target, camera and intervention.
Keep the original floor marking, horizontal aperture, target pose and first three camera trajectories. Use the same focal length, resolution, prompt, checkpoints and thresholds.


Add photographed concrete, painted brick, wood and steel surfaces. Add storage racks, labeled cartons, a rolling cabinet, a clamp, service hardware and two ceiling lamps.
Use a 2.04 m sliding steel door with a face sheet, warning plate and handle. All door parts travel with the door.

Both simulators render each image from three-dimensional geometry. Use area lights in Isaac and spotlights at matching positions in MuJoCo. Allow target shadows in both engines.

| Camera | Configuration |
| --- | --- |
| P1 | Original oblique corridor view. The revisit moves 6 cm sideways and turns 3°. |
| P2 | Original leveled oblique view. Keep the P1 centers and viewing directions. |
| P3 | Original straight-ahead view. The revisit moves 6 cm sideways without turning. |
| P4 | A 95 cm mast looks down the approach at a fixed 13° pitch. Advance 6 cm and turn 1°. |
| P5 | A 48 cm camera looks across the aisle at a fixed 8° pitch. Advance 6 cm without turning. |

P4 and P5 use fixed mounting parameters. The target location does not set their aim. Mirror the approach side with the layout.

For each view, keep or remove the target with the door open or closed. Use a clear nominal reference and one moved-camera revisit.
Run all 320 queries through the same three verifiers. Each simulator contributes 80 opposite-state pairs. The eight layouts are the replication units.

## Review before scoring

Render a separate engineering fixture in both engines. Inspect both door positions from matched and approach cameras. Check material orientation, scale, lighting, hardware and visibility.
Save the accepted fixture images and the design hashes before releasing the panel. Exclude the fixture from every result.

Check the actual camera transforms against the source. Compare P1 through P3 with the earlier measured cameras.

Verify the target interventions, reference boxes and original image files. Check full inputs for equality at native and delivered resolutions before scoring.
Give scorers only the anonymous reference, target box, revisit and fixed instructions. Keep simulator and intervention data with the evaluator.

## Outcomes

Use the eight MuJoCo P3 closed-door pairs for the primary outcome. Require matching native RGB, delivered RGB, box and prompt across opposite physical states.
Count a false deletion when the registered Qwen system answers REMOVED on both members of an exact pair. Report the exact opportunity count.

Two or more distinct layouts establish recurrence on this panel. One establishes an isolated recurrence. Zero with opportunities means no observed recurrence.

With zero exact opportunities, report the primary outcome as untested. Report the physical-state errors regardless of image equality.

For the same eight layouts, report persistent, lost and new frontal failures against the earlier corridor. Mark lost equality separately from a changed answer.
Report oblique and leveled views, both approach views, and all Isaac results separately. Count definite answers, correctly detected open-door removals and all false removals by method.

Separate geometric abstentions from Qwen's own answers. For exposed controls, require at least 95% overlap with the isolated target mask.
Report fraction of visible pixels and every closed-door exact commitment. Preserve all abstentions and execution errors in the query denominator.

The richer treatment changes materials, lighting and geometry together. The matched layouts support a paired sensitivity result on these arrangements.
