# Stress pilot execution and review, 2026-11-17

Terra passed the bounded Isaac Sim operator check on Ascent, with main-agent review.
The small development pilot and full rendering batch are complete.
The full batch passed independent validation. Input preparation continues, with scoring queued behind the final checks.
This is development evidence. It does not establish a method advantage or a new test freeze.

## What passed

The final geometry has 24 distinct base arrangements and five motion levels.
The geometry-only partial-occlusion target passed in both environments.
The main reviewer checked all 120 arrangement records, 600 frame checks and the balanced motion signs.
See [geometry review](../runs/2026-11-17-stress-dev-v1/reviewer/geometry_audit.json).

Terra rendered the first arrangement in each environment at D0 and D6.
The reviewer checked all 80 RGB files and 180 masks, including their camera and source metadata.
Every mask-derived label was recomputed and matched.
Same-input resume preserved outputs. Changed provenance was refused before simulator startup.
See [render review](../runs/2026-11-17-stress-dev-v1/reviewer/smoke_audit.json) and [role qualification](../runs/2026-11-17-stress-dev-v1/reviewer/terra_qualification.json).

The observer reproduced the frozen verifier exactly on two actual inputs, one accepted and one shift-refused.
The guarded repeat produced identical records.
The [amendments](2026-11-17-stress-amendments.md) retain the initial geometry failure, engineering fixes and checkpoint-cache exception.

## Stronger image-only controls

All four analyses completed 60 inputs. The main reviewer independently recounted their saved predictions.
The table reports definite answers out of 15 views. It does not report supported accuracy.

| Analysis | Shelf D0 | Shelf D6 | Tabletop D0 | Tabletop D6 |
| --- | ---: | ---: | ---: | ---: |
| Primary, width 1200, shift limit 100 | 10 | 10 | 11 | 7 |
| Native, width 2304, shift limit 100 | 10 | 5 | 12 | 7 |
| Native, shift limit 192 | 10 | 9 | 12 | 7 |
| Primary, shift limit 200 | 10 | 10 | 11 | 10 |

All definite answers matched physical state across these 240 evaluations.
That does not establish visual support for those answers.
Each environment has only one independent base arrangement in this smoke run.
Its bootstrap output does not establish population precision.

Evidence: [independent evidence review](../runs/2026-11-17-stress-dev-v1/reviewer/smoke_evidence_audit.json).
The four scoring runs took 18.66, 44.28, 44.93 and 18.77 seconds, respectively.
Those times include process startup and model initialization.

## Automated labels and AI review

The researcher authorized AI and programmatic labels instead of waiting for human labels.
The [label supplement](../development/stress/LABEL_PLAN_AI_V1.md) keeps physical truth, geometry proxies and AI judgments separate.

A fresh Terra reviewer inspected all 60 neutral image pairs without their predictions or simulator truth.
It supplied 37 PRESENT, six REMOVED and 17 UNKNOWN labels.
The reviewer checked all 120 presented-image hashes and reconstructed every reference rectangle.
See [AI labels](../runs/2026-11-17-stress-dev-v1/ai-review-smoke/ai_labels.csv) and [input-binding review](../runs/2026-11-17-stress-dev-v1/reviewer/ai_label_binding_audit.json).

The complete geometry proxy agreed with the AI on 47/60 pairs.
Ten proxy-UNKNOWN pairs received AI PRESENT labels. One received AI REMOVED.
Two proxy-REMOVED pairs received AI UNKNOWN labels.
These are disagreements between imperfect references, not measured AI accuracy.
See [the cross-tabulation](../runs/2026-11-17-stress-dev-v1/label-evaluation-v1/ai-vs-proxy.json).

Against the original AI labels, the relaxed shift limit increases tabletop D6 useful answers from 6/15 to 9/15.
Recoverable useful abstentions fall from five to two. The one AI-reference disagreement remains.
See [primary AI-reference scores](../runs/2026-11-17-stress-dev-v1/label-evaluation-v1/ai-primary.json)
and [relaxed-control scores](../runs/2026-11-17-stress-dev-v1/label-evaluation-v1/ai-relaxed_guard.json).

The main reviewer inspected the three primary answers that disagreed with AI labels.
At least one AI UNKNOWN judgment appears too conservative: the remembered can region is visibly clear beside the panel.
The other two need closer review. Their raw AI labels remain unchanged.
This outcome-selected, nonblind inspection is not an independent accuracy estimate.
See [the spot review](../runs/2026-11-17-stress-dev-v1/reviewer/ai_label_spot_review.json).

Terra's simulator qualification does not make its semantic labels ground truth.
Neither the geometry proxy nor these provisional AI disagreements justify a paper claim.
The stronger image-only control already removes much of the apparent recovery opportunity on this tabletop arrangement.
Do not expand the task-model comparison from this smoke alone.

## Completed full render

The researcher supplied Ascent's network address. The reviewer verified the simulator host against its existing SSH host key.
The full render started at 00:33 Pacific and finished at about 03:31 Pacific on 2026-11-17, with exit code 0.
The original running-state handover remains intact as a historical record.

Terra and the main reviewer independently checked all 120 configurations from the 24 base arrangements.
Both audits passed all 2,400 RGB images and 5,400 masks. They recomputed the mask-derived labels.
Their source, label, render, image and mask hashes agree. The original smoke files remain unchanged.
See [main acceptance](../runs/2026-11-17-stress-dev-v1/reviewer/full_render_acceptance.json)
and [the complete main audit](../runs/2026-11-17-stress-dev-v1/full-main-review-v1/full_audit.json).

The rendered nominal obstruction fractions retain 14 middle-band stops out of 60 in each environment.
These are geometric strata, not evidence of visual answerability.
See [the rendered mask counts](../runs/2026-11-17-stress-dev-v1/reviewer/full_mask_band_counts.json).

## Full AI audit and quality hold

The [preselected audit](../runs/2026-11-17-stress-dev-v1/label-evaluation-v1/full-ai-sample.json) contains 180 queries in 30 strata.
Its selection preceded full-cohort predictions. Two fresh Terra reviewers inspected 176 new pairs in separate neutral packs.
Four starter judgments transferred after exact image, target-box and presentation checks.
The main reviewer verified all 352 new presented-image hashes and preserved both raw review files.

The merged sample contains 99 PRESENT, six REMOVED and 75 UNKNOWN judgments.
See [the original AI labels](../runs/2026-11-17-stress-dev-v1/ai-review-full-results-v1/ai_labels.csv)
and [their provenance](../runs/2026-11-17-stress-dev-v1/ai-review-full-results-v1/provenance.json).
The final comparison with completed verifier inputs remains pending. Missing cohort labels remain missing.

The main reviewer inspected two selected removal cases after seeing their AI labels and simulator state.
In both, the main reviewer judged that the former target location is visibly exposed beside the blocker.
Terra marked both UNKNOWN. The raw judgments remain unchanged.
See [the nonblind spot review](../runs/2026-11-17-stress-dev-v1/reviewer/full_ai_label_spot_review.json).
This establishes specific reviewer disagreements, not a blind error-rate estimate.

An incorrect UNKNOWN reference can make a supported removal answer appear unsupported.
It can therefore create a false reason to add a veto.
The current analyzer withholds AI-based triage until the reviewer passes a declared qualification check.
Qualification must include independently reviewed clear PRESENT and REMOVED controls.
The [quality-hold record](../runs/2026-11-17-stress-dev-v1/reviewer/label_quality_hold.json) binds the source change and six passing checks.

Terra is qualified for the bounded simulator role under review.
Its support judgments are provisional and cannot currently release the next model comparison.

## Running jobs and next actions

At 10:32 Pacific on 2026-11-17, preparation contained 4,940 of 7,200 verifier images.
The general review pack and full scoring had not started.
The main reviewer verified that preparation, the conditional pipeline and its checks remained active on Ascent.
See [the live-state snapshot](../runs/2026-11-17-stress-dev-v1/reviewer/live_state_20261117T183229Z.json).

The [conditional pipeline](../runs/2026-11-17-stress-dev-v1/full-conditional-job-v1/handover.json) is detached and authorized.
It waits for preparation to finish and for the independent input audit to pass.
It then runs all four image-only controls, exports both geometry proxies and creates separate factual and proxy-reference reports.
Any failed gate stops the dependent stages and preserves their files.
The [sample-binding check](../runs/2026-11-17-stress-dev-v1/ai-binding-job-v1/handover.json) also waits for preparation.
It verifies pixels and boxes. It does not validate semantic judgments.

The run uses the preserved analyzer snapshot for factual and geometry-proxy reports.
The newer local analyzer adds the AI qualification hold. Neither source authorizes a task-model experiment here.
After completion, independently recount the saved results and inspect failures before choosing the next experiment.
Qualify the support reviewer before expanding AI labels for a decision.

Keep paper claims conditional on independent support-label validation, real-image evidence and the closest-work comparison.
The 85% shelf and 65% tabletop D4 forecasts remain untested predictions.
