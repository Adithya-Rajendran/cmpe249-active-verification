# Does locating the remembered object first improve removal judgments?

This development comparison uses four new layouts in Isaac Sim and MuJoCo. The independent unit is the base arrangement.
Both engines render each shelf and tabletop layout at the original camera pose and at 6 cm and 3°.
The study uses the captured poses, target interventions, repeats and image processing from the [previous qualification](../support_labels/PROTOCOL.md).
The evaluator keeps all masks, opposite-state images and simulator records private during annotation.

## Selection before review

Use only source geometry to choose the layouts. Exclude prior layouts and reserved seeds.
Measure eligible camera positions with the native masks in both engines. Use the original clear, partial and hidden eligibility thresholds.
Use one presented physical intervention per camera position. Both engines supply the same physical intervention at that position.

Select partial views separately for each environment. Each of its two arrangements can supply one retained and one removed target.
Maximize the number of available cases first. Next, minimize the total absolute deviation from one case in each camera-level and physical-state cell.
Then prefer different levels within each arrangement, followed by different stops.
Break equal scores by SHA-256 order using the study identifier, environment and sorted arrangement assignments.

Reserve hidden controls next, then take the clear controls from unused positions. Use one control of each type per arrangement when eligible.
The first layout in each environment has a retained clear control and a removed hidden control. Reverse those states for the second layout.

Prefer the original camera for the first shelf and second tabletop clear controls. Prefer the moved camera for the other clear controls.
Prefer the opposite camera level for each hidden control. Break ties by the same study-specific hash ordering.
Preserve every shortage and any imbalance. The selection uses no predictions or reader answers.

## Matched instruction comparison

Both arms receive the same 1200 by 675 image pairs, reference boxes and reference-based coordinate aids in the same packet order.
The original arm uses the earlier [reader instructions](../support_labels/REVIEWER.md) verbatim.
The two-step arm first [estimates the target location](LOCATE.md) from visible landmarks. The reviewer commits its estimate before receiving the [decision instructions](DECIDE.md).

Each fresh Terra session receives one instruction version and at most one case per base arrangement.
Randomly assign session order within each matched pair of packets. Use the same Terra model and reasoning setting in both arms.

The main reviewer sees the anonymous image pairs once in a separate shuffled order and uses the original task definition.
Lock its answers before opening either arm's results, first-stage boxes or case identities.
Lock each arm's votes and the first-stage records. Then measure physical correctness, complete-input equality and both readers' cited patches.

Mask arm names when the main reviewer evaluates each submitted cue. Unmask after recording the qualitative decisions.
The project verifier receives no inputs from this study.

## Fixed decision rules

Use the [previous evidence rules](../support_labels/PROTOCOL.md#evidence-checks-and-decision) without changing thresholds.
A definite judgment needs matching independent answers, specific visual evidence and passing patches from both readers.
An exactly identical full input under both physical states certifies UNKNOWN. Count all disagreements and uncertified abstentions in their original cells.

Apply the original qualification rule to each instruction version and each simulator.
Require six corroborated partial cases out of eight, including a retained target and a removal.
Require three corroborated clear controls out of four. A wrong definite answer or a definite answer on an exact ambiguity fails qualification.
Incomplete primary counts leave qualification inconclusive unless a contradiction establishes failure.

The primary comparison is the paired gain in corroborated partial removals for the two-step arm, separately by engine.
Report new and lost removals, physical errors, and commitments on exact ambiguities. Report both arms, all controls, and every case.
Advance the new instructions only if they qualify in both engines and gain at least one net corroborated partial removal in each.

An incomplete or unbalanced primary camera-by-state design cannot establish this advance.
If these requirements fail, stop this instruction revision. The support-based model comparison stays on hold.

For the two-step arm, report the overlap of each committed box with the isolated target box at the actual camera position.
Use intersection area divided by union area in full-image coordinates, with exclusive right and bottom edges.
Count missing estimates separately. This diagnostic cannot turn an abstention into a supported removal or change a decision.
The shared main review and the paired renderings limit independence. These results concern the selected development scenes.
