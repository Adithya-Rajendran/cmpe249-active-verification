# Image review instructions

Use only your assigned packet. Treat each case independently.
Each case contains an earlier reference, its target box and an unmarked revisit.
The camera can move. The remembered object can remain in its original location or leave the scene.

Review both images. The gold reference box identifies the object.
The detail panels enlarge the same coordinates in each image. The tick labels give full-image pixel coordinates.

Inspect the full frames when the object or its surroundings extend outside a detail panel.
You may zoom the supplied images. Use only these images to locate evidence.

Return PRESENT when a visible feature identifies the remembered object.
Return REMOVED when you can locate its remembered position and see enough of that space to establish absence.
Explain why the object would occupy the exposed space if it remained there.
Return UNKNOWN when the pictures give insufficient evidence. Explain what prevents a decision.
Use DISPUTED for an unresolved interpretation of the task or image.

For each case, name the target and describe how you located the remembered place in the revisit.
Describe one visible cue and any relevant uncertainty.
For PRESENT, give one point on the identifying part of the object.
For REMOVED, give one point on the exposed space the object would occupy.

For UNKNOWN or DISPUTED, give an optional point on the obstruction or use null.
Use integer revisit coordinates. The origin is the upper left of the 1200 by 675 image.

Write a JSON array to the requested output file. Include exactly one object per assigned ID.
Each object needs the fields `id`, `target`, `anchor`, `cue`, `uncertainty`, `answer` and `point`.
Write the point as `[x, y]` or `null`. Identify an uncertain observation in words.
