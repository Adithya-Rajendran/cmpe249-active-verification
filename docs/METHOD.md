# Method: evidence-aware verification

**Version: 2026-11-17 Pacific. Development specification. No confirmatory study is frozen.**

The [first development pilot](../notes/2026-11-16-development-pilot.md) tested the agreement veto.
Its model verifier failed a retained-target sanity check. It does not establish a method advantage.

The [stress scoring and label review](../notes/2026-11-17-scoring-and-label-review.md) is complete.
The [next experiment](../development/cross_sim/PLAN.md) checks fixed image-only baselines across Isaac Sim and MuJoCo.
It uses physical-state scores for engineering decisions. It does not replace the support-validation requirement.
The active study uses simulation only. Physical-camera transfer remains future work.

The [stress specification](../development/stress/PROTOCOL.md) preserves the completed simulator settings.
A conditional cascade is a separate development hypothesis.

## 1. Task and input boundary

The task checks a remembered target at a defined region.
The primary scenes allow the target to stay or leave that region.
They exclude relocation within the scene and replacement by an indistinguishable instance.

Any later study of those cases needs an identity rule and revised labels.

Every ordinary verifier receives the same inputs:

- One reference RGB image that shows the target.
- One target box in the reference image.
- One revisit RGB image.
- The same target description, task assumptions and output definitions.

The reference contains no information about the later state.
Query localization must come from the images. A test annotator must not supply a query box to a verifier.
The runner must hide masks, true depth, physical state, pair membership, filenames, seeds and future images.

Each query starts a fresh model context. A model must not see the other member of its pair.

PRESENT means that the target remains in the remembered region.
REMOVED means that it is absent from that region under the task assumptions.
UNKNOWN means that this observation does not support either decision.

UNKNOWN is an output, not a physical state.

## 2. Physical state and answer support

The evaluator stores two distinct labels.

| Label | Values | Source |
| --- | --- | --- |
| Physical state | Target present or target removed | Scene intervention or recorded capture procedure |
| Answer support | PRESENT supported, REMOVED supported, unresolved or disputed | Blinded review of the supplied inputs, with reviewer type recorded |

The core cases are:

| Scene | Evidence in the supplied view | Supported answer |
| --- | --- | --- |
| Target present | Enough target evidence for identification | PRESENT |
| Target removed | The relevant region is clear and supports the absence judgment | REMOVED |
| Target present behind a blocker | No distinguishing target evidence | UNKNOWN |
| Target removed behind the same blocker | No distinguishing absence evidence | UNKNOWN |

A partly visible target can support PRESENT. A blocked part of the region can prevent a supported REMOVED answer.
Therefore, a single visibility cutoff must not define both answers.
Mask area defines geometric strata. It does not establish recognizability or absence.

Use one consistent label source for each support score. Report geometric-label agreement separately.
AI labels and geometry proxies are provisional development references. They must never appear as human judgments.

Completeness alone does not qualify an AI label set for a research decision.
Check the reviewer with clear PRESENT and REMOVED controls before using AI-based triage.
Resolve evidence of systematic labeling errors through separate review. Preserve the original judgments.

Qualification must cover the decision's precision and difficult cases.
A basic clear-control pass does not validate a 1% unsupported-error decision.
The [completed reviewer check](../development/stress/LABEL_QUALIFICATION.md) permits assisted development review only. The AI decision hold remains active.

The [label supplement](../development/stress/LABEL_PLAN_AI_V1.md) permits automated development analysis.
Paper claims still need independent validation of support judgments. Missing labels stay missing.

The audit must check shadows, reflections, target fragments, alignment, blur and contextual cues.
The evaluator records disputed cases before model outputs are opened.
They remain in factual scores and coverage. Support scores report their exclusion and a conservative sensitivity analysis.

The sensitivity analysis treats every disputed case as unresolved.

## 3. Matched scenes

A matched pair changes target presence while preserving the reference, camera, blocker and other scene factors.
Clear controls expose evidence in both states. Blocked controls test abstention in both states.
The generator must not use the physical state to choose appearance, lighting or filenames visible to the model.

An exactly identical input cannot identify which of two different physical states produced it.
A correct guess on that pair is still unsupported.
Near-identical renders need a cue audit. Pixel similarity alone does not establish that both views are unanswerable.

The evaluator must not label every nominally blocked pair UNKNOWN without this check.

Pair agreement alone is insufficient. A verifier that always answers REMOVED agrees on every pair.
The scorer therefore reports both-UNKNOWN frequency beside factual errors and useful answers on clear controls.

## 4. Base verifier

The base model chooses PRESENT, REMOVED or UNKNOWN under explicit output definitions.
It sees the full reference and revisit, with the same reference marker across arms.
The runner pins image preprocessing, prompt text, model revision, precision, decoding and label parsing.

The prompt states that a missed detection alone does not establish removal.
It also states that memory of prior presence does not establish current presence behind a blocker.
The model must return one allowed label.

Invalid outputs become UNKNOWN operationally, with a separate failure flag. They never count as successful abstention.

The full comparison requires conditional scores for all allowed answer strings.
A model without a valid confidence score is ineligible for that comparison.
These are model scores, not calibrated probabilities of physical state.

The tokenizer and complete answer strings determine their calculation.

### Qualification before calibration

Finite scores and a valid parser do not establish a competent verifier.
Before gate calibration, check same-view retention, clear presence and clear removal on declared development controls.
Inspect ordinary generation beside constrained scoring. Record response truncation and changes caused by answer format.

Verify that the model can describe the visible target and receives the intended images.

A constant answer that fails these obvious controls blocks a substantive gate comparison.
Keep the failed run as an engineering record. Repair the interface or justify another capable verifier before expansion.
Keep blocked cases in the results. Do not select a weak verifier to create favorable gate errors.

Fix the prompt, reasoning budget, answer strings, parser and confidence calculation together before the next run.

## 5. Separate evidence checks

The development implementation reuses the image front end as a starting point.
It estimates alignment validity, image quality, a target match score and an obstruction score.
These estimates use only the allowed images and the reference box.

They are fallible proxies for support.

Let `valid` mean that image registration and required inputs pass their checks.
Let `q`, `m` and `o` denote quality, target match and obstruction.
The proposed predicates are:

```text
presence_support = valid and q >= t_quality and m >= t_present
removal_support  = valid and q >= t_quality and o <= t_obstruction
                   and m <= t_absent
t_absent < t_present
```

Separate match thresholds create an unresolved interval.
This replaces the old complementary match and miss predicates for the new study only.
The old multi-view stopping rule remains unchanged in the historical implementation.

A positive match can pass despite partial obstruction. Removal needs both a weak match and a usable view of the region.
Registration failure makes both predicates false. The method does not interpret registration failure as absence.
No simulator warp or mask can rescue an ordinary verifier at test time.

### Evidence records for the next implementation

The [literature review](../notes/2026-11-17-va-tamp-and-simulation-scope.md) places evidence gating within established prior work.
For each decision, record input hashes, alignment validity, quality, target match, obstruction and the relevant threshold comparisons.
Record all failed requirements. Distinguish input failure, uncertain localization, poor quality, ambiguous match and obstruction.
Store evaluator geometry in a separate record. These image estimates are not support certificates.

Previous presence identifies the target. It does not establish the target's state at the revisit.
Registration and obstruction estimates must refer to the supplied query.
A blocked or invalid revisit cannot renew a previous definite state.
The current single-pair method keeps no persistent occupancy map.

This requirement adds diagnostic records to the next implementation. It changes no completed prediction or threshold.
Any later active-view method must target missing decision evidence using requested observations and permitted candidate geometry.
It needs a separate protocol and equal view and movement budgets across policies.
Simulator visibility for unrequested images remains evaluator-only information.

### Agreement veto and image-only comparison

The v1 agreement veto retains the base answer only when the image verifier agrees:

```text
if base == PRESENT and presence_support:
    return PRESENT
if base == REMOVED and removal_support:
    return REMOVED
return UNKNOWN
```

The rule never turns the base model's UNKNOWN into a definite answer.
It never swaps PRESENT and REMOVED. It therefore tests selective retention of the base decisions.
It does not prove that retained answers have support. That is an evaluation question.

All veto arms use the same cached base prediction for each input.
They must not call the model again and attribute a changed answer to the gate.

The gate can only discard base decisions. Its useful correct yield cannot exceed that base on the same inputs.
Any implementation that reports the opposite must explain a changed input, decoding rule or error.

The image-only verifier returns PRESENT when `presence_support` holds and REMOVED when `removal_support` holds.
It returns UNKNOWN otherwise. The predicates cannot both hold because `t_absent < t_present`.
At identical thresholds, the agreement veto answers on a subset of image-only inputs, with identical retained labels.

It cannot increase coverage or useful yield over that image-only verifier.
It can reduce unsupported answers if the model rejects an erroneous image-only decision.
Separate calibration changes the comparison. Always retain the matched-predicate control and a separately calibrated image-only control.

## 6. Conditional cascade

The cascade is a separate development method. It can add answers where the image-only verifier abstains.

```text
image_answer = image_verifier(inputs)
if image_answer != UNKNOWN:
    return image_answer
return qualified_model_with_rejection(inputs)
```

The model receives the same permitted input pair in a fresh context.
It receives no geometry, true pose, physical state or other query.
A confidence rejection or invalid output returns UNKNOWN, with failures recorded separately.

The cascade cannot correct a definite image-only error. Agreement veto addresses that different question.
The cascade can increase useful yield and can also add unsupported commitments.
Its relevant opportunity is answerable image-only abstention, not every refusal or physically hidden target.

Do not run this comparison until the support-label decision in the stress specification permits it.
Complete AI labels can support a provisional development decision under the label supplement.
This requires a reviewed validation and analysis revision. The present AI decision hold remains active.

Require an advantage over image-only verification, including a credible guard control.
A relaxed guard that aligns correctly does not necessarily answer correctly. Score its support and useful yield too.

## 7. Controls that identify the mechanism

| Arm | Purpose |
| --- | --- |
| Image-only verifier | Reference for the primary comparison |
| Image-only guard controls | Separate pixel units and refusal settings from model value |
| Qualified model with UNKNOWN | Tests model competence and displacement sensitivity |
| Conditional cascade | Tests recovery of answerable image-only abstentions |
| Agreement veto | Tests error reduction only after image-only errors appear |
| Confidence rejection | Tests ordinary rejection with the same cached model scores |
| Oracle support diagnostics | Describe veto retention and recoverable abstentions separately |
| Always UNKNOWN, PRESENT and REMOVED | Expose class imbalance and misleading scores |

Keep the v1 place-first and removal-only arms as historical controls.
A new reasoning arm must actually generate intermediate text before its answer.
Forcing the first answer token does not implement a reasoning comparison.

A forced binary prompt cannot serve as the primary model comparator.

The confidence control searches every distinct calibration score boundary, including accept-all and reject-all.
Record the score definition before calibration. Complete answer-string scores are not calibrated truth probabilities.
The existing scorer already converts logits to float32 before normalization.

A new precision condition requires a declared forward-pass or output-projection change, with runtime and parity checks.
Use the same cached base outputs in agreement-veto and confidence arms.

An oracle veto cannot repair a wrong label or answer a model UNKNOWN.
A recoverable-abstention diagnostic instead asks where a supported definite answer exists after image-only UNKNOWN.
Neither diagnostic is a deployable method or a guarantee that another model can attain it.

## 8. Calibration and limits

Development arrangements choose thresholds. All images, states and nuisance variants of one arrangement remain together.
The search grid and selection rule must exist before the calibration outcomes are inspected.
The [evaluation plan](EVALUATION_PLAN.md) defines the joint error and usefulness requirement.

If no setting meets it, calibration fails. The runner must not relax a constraint to create a winning setting.

The present specification adds no sensor depth, repeated revisits, active view selection or learned gate.
Those are separate hypotheses. They do not become fallback methods after test results are seen.
The development runner, AI label audit and calibration code now exist.

The pilot failed baseline qualification. Independent support validation and a credible comparator remain prerequisites for a support-based freeze.
Programmatic certificates or a validated external reference can supply that validation for their declared scope.
Physical-state labels alone cannot validate a visual-support endpoint.
