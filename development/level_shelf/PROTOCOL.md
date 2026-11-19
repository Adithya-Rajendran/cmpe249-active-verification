# Level-shelf camera control

## Question

Does a verifier still report removal behind an opaque panel when the shelf looks level?
This is a follow-up on eight previously inspected shelf layouts.
The illustrated MuJoCo gelatin box at PAM01, original P5, is the primary example.

## Captures

Use all eight shelf layouts from the paper in both simulators.
Keep every object, blocker, shelf board, light and camera intrinsic from each source layout.
Render the following five camera slots. Render all four original target and blocker combinations.

| Slot | Camera |
| --- | --- |
| P1 | Replay original P4. |
| P2 | Level the front edge of the floor board at original P4. |
| P3 | Face the shelf straight on with world-up vertical. |
| P4 | Replay original P5. |
| P5 | Level the same front edge at original P5. |

For P2 and P5, rotate each camera about its existing optical axis.
Make the left and right endpoints of the front edge project to the same image row.
Keep the original camera center, heading, pitch, focal length and field of view.
Apply this roll correction separately to the reference and revisit cameras.

This rotation levels the selected edge. An oblique camera can still show perspective on other edges.

For P3, put the nominal lens on the shelf centerline at the original arc radius and height.
Set heading, pitch and roll to zero in both frames.
Use the original P5 translation sign for a 6 cm lateral shift. Keep yaw zero.
This view changes the line of sight and may reveal the target. Check the masks.

Use the nominal clear reference and the D6 revisit for every decision.
The four oblique slots retain the original signed 6 cm shift and 3° yaw.
Render original camera positions again to measure replay agreement.
Compare the new baselines with the corresponding original paper predictions and image hashes.
The eight layout clusters supply 320 queries across both engines. Use all queries.

## Checks before interpretation

Verify the camera and object transforms from each renderer and the visibility flags for each intervention.
Measure the projected edge with the observed camera matrix.
Require absolute slope below 0.05° for each corrected floor edge and each frontal edge.

Measure reference, retained and isolated target masks. Verify zero target pixels in every removed-state mask.
Record equality of the full native and delivered RGB images, reference box and task for every retained/removed pair.
Only complete equality establishes identical model inputs. Report partly visible and unequal pairs separately.

Give the three original scorers randomized opaque identifiers and the original inference inputs.
Keep the simulator, camera labels, masks and intervention labels outside the packet.
Score each record once with the original checkpoints, code, prompt and abstention rules.

## Outcomes

For the illustrated original P5 pair, report both answers in each camera treatment and both engines.
If both inputs still match and Qwen returns REMOVED twice after leveling, the failure survives the roll correction.
If it abstains twice on an equal pair, its behavior changed with image orientation.

If the frontal camera uncovers the target, report the newly available evidence separately.
If the original decision fails to repeat, disclose that before attributing differences to orientation.

Repeat the comparison on all four previously identified MuJoCo pairs with unsupported Qwen removals.
These four pairs span three layouts. Count each unique frontal layout once in aggregate.

Across all eight layouts, report clear correct answers, false removals, exact-pair commitments and coverage by engine and treatment.
Distinguish abstention from the geometric check and abstention from Qwen.
Report observed counts on this reused panel. A new result is a sensitivity check.
