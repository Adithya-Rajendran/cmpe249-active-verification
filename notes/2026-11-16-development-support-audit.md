# Development support audit, 2026-11-16

The audit separates the scene state from the answer that an image supports.
It uses existing Isaac Sim renders. No new scene or confirmatory test accompanies this audit.

The main assistant reviewed labels before task-model predictions.
Terra prepared the images and implemented the pilot on Ascent.
The [review ledger](../runs/2026-11-16-support-audit-v1/support_review.json) records each decision and its reason.

## Cohort and inputs

The cohort contains the first six complete random development arrangements from each environment.
The shelf IDs are SD1000 through SD1005. The tabletop IDs are TD5011 through TD5016.
Each arrangement contributes P1, P3 and P5 under L1, with visible, occluded and removed query phases.

Alternating arrangement indices define calibration and pilot evaluation before inference.
Each partition contains six arrangements and 54 queries. Both partitions are development data.
No image, state or view from one arrangement crosses this partition.
The earlier four-arrangement draft is superseded and retained for provenance.

The supplied images contain 1200 by 675 RGB pixels. Resizing preserves their aspect ratio.
The reference has a red box around the remembered target. The query has no marker.
The preparation step removes source metadata. Model messages contain no source filename, physical label or pair identifier.

The source PNG files contain simulator metadata, including state information.
The model boundary rejects metadata and verifies prepared-image hashes.
The image front end uses unmarked images at the same resolution and the scaled reference box.
The model marker is a presentation aid. It does not enter appearance matching.
The pilot does not reuse its old scores from higher-resolution images.

## Review result

| Partition | PRESENT supported | REMOVED supported | UNKNOWN | DISPUTED | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | 31 | 12 | 8 | 3 | 54 |
| Pilot evaluation | 28 | 10 | 15 | 1 | 54 |
| Both | 59 | 22 | 23 | 4 | 108 |

These counts come from `support_review.json`. They are annotations, not model performance.
The file hash is `99fcad2707f177c52aa3ac996b4ca8f63c010973b8d4c76a282e6914c27fe44f`.
The scheduled queries contain 72 physically present states and 36 removed states.
These benchmark weights do not estimate a deployment prevalence.

Four occluded P3 queries remain disputed:

- SD1002: a thin dark gold lid above the blocker.
- SD1003: a narrow blue edge beside the blocker.
- TD5011: a thin red and white strip beside the blocker.
- TD5015: a small red curved fragment beside the blocker.

These fragments could identify the remembered target, but this review cannot establish that confidently.
The main support metrics exclude these queries. A sensitivity analysis treats them as UNKNOWN.
Factual errors and coverage retain all queries.

TD5013 at P3 supports PRESENT despite heavy occlusion.
The visible red curved rim and ribbed side match the reference mug.
Its removed counterpart supports UNKNOWN because the blocker still hides the remembered place.
This asymmetry shows why the old visibility category cannot define answer support.

SD1001 has no supported-removal query in the selected views.
The arrangement remains in the pilot. Its removal-resolution denominator is undefined.
It remains eligible for other metrics. The analysis must report the eligible arrangement count for each metric.

## Limits and implications

One AI reviewer inspected the source contact sheets and difficult sanitized queries at their supplied size.
The ledger lists every query inspected at that size.
The reviewer also saw physical labels, paired images and geometric diagnostics.
This review is not blind, and no human agreement measurement exists.
The labels are suitable for a development probe. They are not established human ground truth for a paper.

The review found no additional decisive shadow or reflection cues in the inspected cases.
That observation does not prove their absence. Subtle rendering cues remain possible.
Pixel similarity and small mask fractions cannot establish that an image lacks usable evidence.

The next label study needs independent human review before model outputs are visible.
It must record agreement, disagreements and the adjudication procedure.
The rubric must permit supported presence from partial evidence and unresolved removal when the relevant region remains blocked.

The primary task assumes no relocation within the scene or indistinguishable replacement.
The pilot does not support claims about those cases.
All source evidence and the historical freeze remain unchanged.
