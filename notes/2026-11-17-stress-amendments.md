# Stress development amendments, 2026-11-17 UTC

These are development changes. No simulator image or task prediction preceded amendment 1.
The researcher authorized plan corrections and changes as needed before bounded simulator execution.

## 1. Increase the geometry candidate limit

The first eight-candidate search selected 13/60 middle-band stops for shelf and 9/60 for tabletop.
The tabletop count missed the planned minimum of 12/60.
Evidence: `/mnt/bench-data/datasets/cmpe249-active-verification/stress-dev-v1-20261117/geometry/summary.json` and `generation_log.json`.
The main reviewer independently read both the counts and the per-slot candidate records.

Increase the valid-candidate cap from 8 to 32 for every slot in both environments.
Keep the seed blocks, attempt budget, targets, classes, signs, frame checks and ranking rule unchanged.
This is a single bounded geometry-only expansion, not a search over model errors.
The new search has not established that the quota is feasible.
If it still fails, record the failure before deciding on another design change.

Preserve the original failed geometry attempt and the already-started serialization revision as separate records.
Both precede rendering and inference. Neither supplies an eligible final cohort.

## Engineering corrections before first render

The main review found identifiers that would fail the existing episode schema and frame checks that used rounded margins.
The input reviewer also found missing revisit metadata required by the frozen renderer.
Use compact valid identifiers, unrounded margin decisions and explicit deterministic-pose metadata.
These corrections do not change frozen source files or the intended camera motions.
Regenerate the final cohort with the corrected builder and record its actual source hash.

## Amendment 1 result and smoke acceptance

The final geometry search met the target in both environments: 14/60 middle-band stops each.
The main reviewer independently checked all 120 level arrangements and 600 camera frame tests.
The smallest projected frame margin was 50.50420624996718 pixels.
The 24 base arrangements have distinct object layouts. Every motion sign has the declared count.
Evidence: `runs/2026-11-17-stress-dev-v1/reviewer/geometry_audit.json`.
These are geometry results. They do not establish image answerability.

Terra rendered SXS01D0, SXS01D6, SXT01D0 and SXT01D6 on Ascent with Isaac Sim 6.2.
The main reviewer checked all 80 RGB files and 180 masks against the declared inputs.
All mask-derived labels were recomputed and matched. Every removed view has zero target-mask pixels.
Representative shelf and tabletop images were inspected directly.
The same-input resume skipped rendering and preserved the output hashes.
A changed-provenance probe was refused before simulator startup.

Evidence: `runs/2026-11-17-stress-dev-v1/reviewer/smoke_audit.json`, `smoke-review/resume.log`,
`smoke-review/resume-integrity.log` and `resume-mismatch-probe/result.json` under the same run directory.

Terra is sufficient for this bounded simulator role with main-agent review.
That judgment follows actual execution, not only a code review or the agent's self-report.
Reviewer corrections were necessary before the first render. Scientific acceptance stays with the main reviewer.
The smoke run supplies no human support labels and no support-based go or stop decision.

## Metric consistency clarification

After the smoke render, source review found a mismatch in the headroom numerator.
The useful-yield metric requires the answer, human support label and physical state to agree.
The initial headroom implementation counted any definite support label, even when it conflicted with physical state.
Require the support label to agree with physical state for a recoverable useful answer.
Keep the human label unchanged and report conflicts separately.
This correction precedes the first new evidence inference and every human label.
It changes no cohort, image, threshold, motion or decision margin.

## Runtime cache correction

The first two-input parity check unexpectedly downloaded CLIP through Ultralytics.
The Hugging Face offline settings did not cover that loader.
Its actual weight directory came from settings loaded before the project override took effect.
The downloaded file has the frozen checkpoint hash, `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
It did not change the model choice or the original checkpoint.

Terra stopped before the 60-input runs and identified the actual path and installed loader.
The repeat used the existing pinned checkpoint and a guard against outbound Python IP sockets and DNS.
That guard does not constrain arbitrary native-code network calls.
It records the actual CLIP path and hash, loader source hash and guard source hash.
The two parity records exactly matched the first run, including the accepted and refused alignment cases.
Both frozen-versus-wrapper numerical differences were zero.

Evidence: `runs/2026-11-17-stress-dev-v1/parity-primary-v1.log`, `parity-primary-v2.json`
and `parity-primary-v2-guard.json`. Preserve the first log as the runtime exception record.
The larger smoke scoring runs use the corrected setup. No dependency or frozen source was changed.

## Researcher-authorized label amendment

After the smoke scoring runs, the researcher authorized AI or programmatic labels instead of waiting for human review.
The [label supplement](../development/stress/LABEL_PLAN_AI_V1.md) records the revised development procedure.
The geometry and render protocol JSON remains unchanged because it binds the existing cohort.
New label reports also bind the separate label-plan JSON.

Simulator intervention and masks produce exact physical-state and geometry records.
A complete geometry-support proxy remains distinct from a blinded AI support judgment.
Neither source is represented as human validation. Proxy UNKNOWN does not prove visual unanswerability.
The first AI audit covers all 60 starter pairs. The full audit preselects 180 queries in 30 strata before full-cohort predictions.
The [execution note](2026-11-17-stress-execution.md) records actual disagreements and the main review.

Complete AI labels can support explicitly provisional development triage under the existing numerical rules.
A geometry proxy alone cannot establish a support-based model go decision.
Independent validation of support judgments remains necessary for the proposed paper claims.
