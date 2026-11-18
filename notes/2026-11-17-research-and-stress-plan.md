# Research check and development revision, 2026-11-17 UTC

The researcher requested research before corrections, followed by Terra execution of Isaac Sim on Ascent.
This note records the reasoning before new simulator outcomes. It does not create a test freeze.

## Literature and the claim

- [CYWS-3D, sections 1 and 3](https://arxiv.org/html/2308.10417v2) already addresses two RGB images of a changing 3D scene. It distinguishes physical changes from viewpoint occlusion and uses estimated depth for feature registration. Our narrower reference-region task is not a new change-detection problem. Its implementation is a relevant future comparator.
- [SceneDiff, sections 3–5](https://arxiv.org/html/2512.16908v2) explicitly addresses verifiable changes, including apparent disappearance through occlusion or leaving the frame. It supports both two-image and multiview inputs. Its primary metrics score change segmentation and detection. A possible distinction here is scoring supported PRESENT, REMOVED and UNKNOWN decisions for a supplied target. That distinction is an inference from the reviewed sections, not proof of novelty. A method paper needs a suitable two-image change-detection baseline, not just a weak language model.
- [FOCUS, sections III and IV](https://arxiv.org/html/2610.05639v1) maintains persisted, replaced and removed object beliefs using aligned RGB-D sequences and a reference map. It models uncertain appearance and correlated observations. This further limits broad claims about new uncertainty-aware robot memory. Our current RGB pair experiment excludes replacement and temporal fusion.
- [Selective Classification](https://arxiv.org/abs/1705.08500) establishes rejection as an error–coverage tradeoff. Agreement veto and confidence rejection are controls, not novelty claims.
- [OpenCV's ECC specification](https://docs.opencv.org/doc/doxygen/html/dc/d6b/group__video__track.html) defines a geometric warp and warns that large displacement can require initialization. A pixel-shift guard is not the only source of failure. Alignment diagnostics must respect the warp direction and pixel coordinates.

**Position.** Continue one bounded stress experiment. Do not claim a new veto principle, a publication slot, or a capable memory system. Retain the model direction only if it improves a credible image-only comparator on supported decisions. A larger project is conditional on that finding and real-image evidence.

## Corrections to the earlier reviews

The dated reviews remain unchanged. These corrections supersede the affected interpretations.

1. The v1 pilot recalibrated four evidence thresholds on resized inputs. Registration parameters were inherited. See `development/pilot/calibrate.py`, `CALIBRATION.md` and the saved analyses in `runs/2026-11-16-support-audit-v1/`.
2. The refused cases in `runs/2026-11-16-claude-assessment/ecc_range_probe.py` retain displacement-length error only. Equal lengths do not establish correct direction. The percentages called correct rejected alignments in round 3 are unsupported.
3. Accepted synthetic-yaw errors compare against an estimated baseline homography. They measure consistency, not absolute scene accuracy. Artificial borders prevent a guaranteed performance bound for translated views.
4. The saved 1200-pixel sweep has an exploratory 7/8-degree extension. Its results must not enter a table described as the sweep through six degrees. The 2026-11-16 night LOG entry records the recount.
5. A projected target residual can arise from parallax under correct background alignment. It must not automatically count as a wrong answer or wrong background fit.
6. Primary support labels must have one consistent human basis. Geometry is a stratum and diagnostic, not a substitute for human labels on conveniently easy cases. Repeating one person's labels does not create independent agreement.
7. A cascade can recover only answerable image-only abstentions. A physically present object with no identifying cue supplies no such opportunity.
8. The existing scorer already converts logits to float32 before normalization. Repeating that cast cannot undo low-precision forward-pass quantization.

## Adopted development choices

The later [label supplement](../development/stress/LABEL_PLAN_AI_V1.md) supersedes the human-label wait described in this initial design.
The researcher authorized AI and programmatic development labels after the smoke run.

The [stress specification](../development/stress/PROTOCOL.md) fixes the next experiment before inference.
It uses fresh development arrangements, all five stops, L1, balanced motion signs and five severity levels.
The primary input size remains 1200 x 675. The original-size and shift-guard analyses remain separate controls.
Human labeling precedes the support-based decision. Geometry and operational results can run before those labels exist.

The forecast remains **85% shelf and 65% tabletop**, at 4 cm and 2 degrees, under the specified balanced sign rule.
These are predictions of supported correct resolution among answerable primary-size views. They are not measurements.

Terra must first pass a bounded execution check: provenance, actual renders, matched interventions, complete artifacts and fail-closed validation.
The main reviewer independently checks those records before expanding the run. This qualifies Terra for this role only.

## Stop rules

The development thresholds are engineering decisions, not power calculations or safety guarantees.
Unsupported answers above 1% in at least three arrangements at one nonzero level permit an agreement-veto comparison.
The rule applies separately by environment, on human labels and on the same labels with disputed cases unresolved.
A useful-yield loss of at least 0.10 permits a cascade investigation only with at least 0.10 recoverable answerable abstentions.
If reasonable image-only controls eliminate that opportunity, do not justify a model by the original guard alone.
If a qualified model cannot improve useful yield by 0.10 while limiting added unsupported answers to 0.02, stop this cascade design.
No informative stress result means stop expanding this simulation axis and assess a small real development set.
