# Label reviewer qualification, 2026-11-17

The full audit exposed two removal judgments that the main reviewer disputes.
The target regions appear clear beside the blockers, but Terra returned UNKNOWN.
An incorrect UNKNOWN reference can falsely count a supported answer as unsupported.

Multimodal judges can show bias, hallucination and inconsistent judgments.
The project therefore checks the judge on its own task before using its labels for decisions.
This is a project inference from [Chen and coauthors, ICML 2024](https://proceedings.mlr.press/v235/chen24h.html).
That paper does not establish Terra's reliability on this task.
[The VizWiz task](https://vizwiz.org/tasks-and-datasets/vqa/) also separates answering from answerability assessment.
Its human annotations do not validate this simulator task.

## Declared qualification

The machine-readable specification is [label_qualification_v1.json](label_qualification_v1.json).
It fixes 48 controls before new reviewer predictions.
Each environment contributes eight PRESENT, eight REMOVED and eight UNKNOWN controls.
The controls cover D0, D4 and D6 camera levels.
The existing audit sample and smoke IDs are excluded.

Geometry selects candidates. It does not supply the final support reference.
The main reviewer inspects both images and records a visual cue before locking each control.
The main reference uses AI judgment with known simulator state. It is not blind human validation.

The main reviewer can replace at most 12 ambiguous or contrary candidates through the saved ranking.
A failure to obtain the declared controls stops qualification.

Two fresh Terra reviewers receive the same controls with different instructions.
One receives the original prompt. The other receives one declared revision.
The revision requests the target identity, query region, visible evidence and final label.
It distinguishes a nearby blocker from a blocker covering the remembered region.
Neither reviewer receives the reference labels, state, geometry, predictions or class balance.

A pass requires at least 15/16 PRESENT and 15/16 REMOVED controls correct.
It also requires at least 14/16 UNKNOWN controls correct.
Each environment must meet 7/8, 7/8 and 6/8 respectively.
Any opposite definite answer on a clear control fails the qualification.
DISPUTED counts as incorrect for these reviewed clear controls.

These are development screening criteria. They are not a statistical bound on label error.

## Consequences

If the revised prompt passes, use it for a separate review of the fixed audit sample.
Preserve the original labels. Compare the versions before interpreting apparent unsupported answers.
If it fails, stop using Terra as the sole support reviewer for this direction.
Do not tune another prompt on these controls.

Passing this screen does not establish label error below the study's 1% unsupported-error threshold.
It does not qualify ambiguous fragments, all scene types or native-resolution images.
The current analyzer retains its AI decision hold until the full support and strong-control requirements pass.
The factual and geometry-proxy reports continue independently.

## Completed basic check

Both instruction versions passed the declared criteria with 47/48 agreements against the locked main-review reference.
Each version matched 16/16 PRESENT, 15/16 REMOVED and 16/16 UNKNOWN controls.
Both versions called the same shelf removal UNKNOWN.
The revised instructions show no improvement on this control set.

The main reviewer inspected 50 candidate pairs before reviewer exposure.
Two ambiguous candidates received their next ranked replacements.
These exclusions affect qualification controls only. All stress cases remain in the scoring cohort.

The [original score](../../runs/2026-11-17-stress-dev-v1/label-qualification-v1/original-score.json) and [revised score](../../runs/2026-11-17-stress-dev-v1/label-qualification-v1/localize-score.json) preserve the counts and input bindings.
Both reviewers used fresh contexts. Neither received the locked reference labels or scoring results.
Each version has one reviewer run. This comparison does not isolate a repeatable prompt effect.

## Completed audit repeat

Two fresh Terra reviewers repeated all 180 judgments with the revised instructions.
The repeat includes new judgments for the four earlier starter cases.
All 360 presented images match the fixed audit inputs.
The original labels remain unchanged.

Seventeen judgments changed. Fifteen changed from UNKNOWN to REMOVED.
One changed from UNKNOWN to PRESENT. One changed from PRESENT to UNKNOWN.
The repeated labels contain 99 PRESENT, 21 REMOVED and 60 UNKNOWN judgments.

The main reviewer inspected all 17 changed pairs and agreed with 16 revised judgments.
The remaining disagreement concerns a dark cube visible beside a tilted box.
The main reviewer judges PRESENT. The revised Terra judgment is UNKNOWN.
This review was nonblind and selected only disagreements. It does not estimate overall label accuracy.

See [the comparison record](../../runs/2026-11-17-stress-dev-v1/reviewer/ai-repeat-comparison.json) for labels, cues and image bindings.
No raw judgment was replaced or merged.

Main also reviewed all four cases where the geometry proxy returned REMOVED and the repeated AI audit returned UNKNOWN.
Main judged all four REMOVED. Both AI versions retained UNKNOWN.
These cases span two shelf arrangements and remain unresolved reviewer disagreements.

The [additional review](../../runs/2026-11-17-stress-dev-v1/reviewer/remaining-removal-review.json) records the selection, cues and images.
Main knew the geometry and predictions. This review does not supply independent validation.

Terra passes the basic development screen. It remains an assisted support reviewer, with disagreement review.
The evidence does not qualify Terra as a sole reference for the study's 1% threshold.
Do not expand to 1,800 AI labels solely because the basic screen passed.
The full support decision remains unavailable. The AI decision hold remains active.
