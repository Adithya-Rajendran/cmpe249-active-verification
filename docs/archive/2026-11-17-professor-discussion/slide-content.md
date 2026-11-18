# 01. When does a view justify an answer?

When does a view justify an answer?

Evidence-Aware Verification for Robot Scene Memory

Methodology discussion with Professor Kaikai Liu

PROPOSED NEXT STEP

Stress image-only verification before adding a model.

COMPLETED

View selection

The first study left little room for a better selector. Preserve its results and design limits.

CURRENT

Supported answers

Can the supplied view justify PRESENT, REMOVED or UNKNOWN for a remembered target?

CONDITIONAL

Model contribution

Add a model only if it improves useful answers or rejects unsupported answers beyond strong controls.

DEVELOPMENT

Full render complete. Full-cohort decision pending. No new test freeze.

Meeting aim: agree on the scientific question, the comparison, and the evidence needed to continue.

---

# 02. Physical truth and visual support are different

Physical truth and visual support are different

A correct guess about a hidden object can still lack support.

IDENTIFYING CUE

PRESENT

Physical state: present

EXPOSED REGION

REMOVED

Physical state: removed

NO DISTINGUISHING CUE

UNKNOWN

Physical state: present

NO DISTINGUISHING CUE

UNKNOWN

Physical state: removed

Conceptual examples. UNKNOWN describes the evidence, not a third physical state.

FROM INPUTS TO ONE DECISION

Reference image

Target box + description

One revisit image

No query box

One decision

PRESENT / REMOVED / UNKNOWN

Masks, physical state, true pose and paired queries stay with the evaluator.

TASK BOUNDARY

REMOVED means absent from the remembered region.

Relocation and indistinguishable replacement stay outside the current task.

---

# 03. A controlled stress test of the image verifier

A controlled stress test of the image verifier

Vary camera displacement while keeping matched hidden and removed scenes comparable.

SHELF

TABLETOP

12 fresh arrangements per environment

Actual reference renders from the smoke run.

DECLARED MOTION LEVELS

D0

0 cm

0° yaw

D1

1 cm

0.5° yaw

D2

2 cm

1° yaw

D4

4 cm

2° yaw

D6

6 cm

3° yaw

Camera-right translation + yaw. Four balanced sign combinations.

24 arrangements × 5 stops × 5 motion levels × 3 query phases

1,800 planned query views

CONTROLLED

L1 lighting. Fixed D0 reference. Matched blockers and distractors. Primary size: 1200 × 675.

PARTIAL OCCLUSION

Geometry selection targets ≥20% middle-band stops at D0 in each environment.

---

# 04. A stronger image-only control already matters

A stronger image-only control already matters

Smoke run only: one independent arrangement per environment. Each cell contains 15 query views.

DEFINITE ANSWERS / 15

These counts measure coverage, not supported accuracy.

Shelf

Tabletop

D0

D6

D0

D6

Primary

Width 1200 · shift limit 100 px

10

10

11

7

Native resolution

Width 2304 · shift limit 100 px

10

5

12

7

Normalized limit

Width 2304 · shift limit 192 px

10

9

12

7

Relaxed limit

Width 1200 · shift limit 200 px

10

10

11

10

TABLETOP · D6 · PROVISIONAL AI REFERENCE

Useful answers

Primary

6/15

Relaxed limit

9/15

Answerable abstentions: 5 → 2

One disagreement with the AI reference remains in each control.

Every definite answer in the four smoke analyses matched physical state. Visual support remains a separate question.

DECISION

Test strong image-only controls before attributing recoverable answers to a model.

---

# 05. Judge usefulness and unsupported answers together

Judge usefulness and unsupported answers together

Keep physical state, geometric proxies and support judgments separate.

PHYSICAL STATE

What happened?

The scene intervention records whether the target stayed or left the remembered region.

GEOMETRY PROXY

What is exposed?

Masks describe obstruction. They do not establish recognition or evidence of absence.

SUPPORT JUDGMENT

What can we infer?

A blinded reviewer sees only the allowed image pair. AI judgments remain provisional.

13 / 60

AI–geometry disagreements

The main reviewer disputes at least one AI judgment. Preserve original labels and report sensitivity.

Smoke audit, not an AI accuracy score.

Useful yield  Y

Supported, physically correct definite answers / support-labeled inputs

Unsupported definite rate  U

Definite answers that disagree with the support label / support-labeled inputs

UNKNOWN lowers coverage. More answers alone do not establish improvement.

The full AI audit preselects 180 queries. A support-based go decision needs complete labels under the current rule.

---

# 06. Let the result choose the next experiment

Let the result choose the next experiment

Apply the development rules separately in each environment. Do not pool motion levels to trigger a branch.

First: finish full review, run all four controls, and complete the required support labels.

Unsupported answers

U > 0.01 at one nonzero level, with events in at least three arrangements.

Permit a qualified agreement-veto comparison.

The rule must hold with undisputed labels and with DISPUTED treated as UNKNOWN.

Useful answers decline

No A, and useful yield at D6 falls at least 0.10 below D0.

Drop veto as the main claim. Assess answerable abstentions.

A cascade needs recoverable answers on ≥0.10 of labeled inputs, beyond credible image-only controls.

Neither rule holds

No A and no B under the declared rules.

Stop expanding this simulation axis. Assess real development images.

Keep the boundary finding. Do not add a model without an evidence-based reason.

AI-only decisions are provisional. Geometry alone cannot authorize the model stage.

These thresholds guide development. They are not statistical significance tests or deployment guarantees.

---

# 07. Use a model only where it can add evidence

Use a model only where it can add evidence

The v1 model interface failed retained-target controls. Qualify the next interface before a task comparison.

QUALIFY THE INTERFACE FIRST

Qwen first

≥34/36 same-view PRESENT

≥90% correct in each clear-state class

Image verifier

Same permitted image pair

UNKNOWN

Qualified model + rejection

No simulator truth or extra query

PRESENT / REMOVED / UNKNOWN

Rejected or failed calls return UNKNOWN

Keep a definite image answer

The cascade cannot correct a definite image-only error.

At matched thresholds, veto can discard errors. It cannot add useful answers.

CASCADE CONTINUATION RULE

Δ useful yield ≥ +0.10

Δ unsupported rate ≤ +0.02

BOUNDED MODEL WORK

Permit one larger replacement after qualification failure. Stop model claims if both fail.

Require both cascade thresholds per claimed environment, including disputed-label sensitivity, against credible image-only controls.

---

# 08. Progress through evidence milestones

Progress through evidence milestones

The schedule follows the results. The course has no fixed deliverable in the current record.

NOW

Finish the stress test

Review the full outputs. Score four controls. Resolve label-sensitive decisions.

Output: Y, U, refusals and recoverable answers by environment and level.

CONDITIONAL

Test the model

Qualify the interface. Split calibration and evaluation by whole arrangement.

Output: cascade or veto comparison with image-only and confidence controls.

AFTER C OR USEFUL STAGE 2

Check real images

Plan ten arrangements, three stops and three query phases. Measure ten camera returns.

Output: transfer limits, support labels and failure cases from real scenes.

PAPER READINESS

Prepare a fresh test

Compare closest work. Obtain independent support-label validation. Choose a generalization axis.

Output: fixed settings, justified precision, sample size and a new freeze.

THE INDEPENDENT UNIT IS THE ARRANGEMENT

Keep all views, states and motion levels together in splits and bootstrap draws. UNKNOWN counts against coverage.

Current snapshot: 120 render folders complete. Structural validation passed. Full scientific review and decision remain open.

---

# 09. Four decisions to leave the meeting with

Four decisions to leave the meeting with

The paper is a research aim. Its contribution remains conditional on stronger evidence.

01

Research question

Is target-level supported-answer evaluation worth pursuing beyond change detection?

Working position: keep the claim narrow. Do not claim a new abstention principle.

02

Baseline comparison

Which two-image baseline would make the comparison convincing?

Candidates: CYWS-3D and the two-image SceneDiff setting. Declare adapters before evaluation.

03

Real evidence and labels

What real-data scope and independent label review would support a workshop submission?

Working position: start with the small real pilot, then justify a fresh evaluation.

04

Stopping and contribution

If strong image-only controls solve the task, is the boundary finding worth pursuing?

Working position: retain negative evidence and stop a model direction that adds no value.

Meeting notes:  priority __________________     baseline __________________     next evidence __________________

---

# 10. Evidence behind this discussion

Evidence behind this discussion

Snapshot: 17 November 2026. Results shown here are development observations.

PROJECT DOCUMENTS

S1

Method, evaluation and current plan

docs/METHOD.md · docs/EVALUATION_PLAN.md progress/PLAN.md · progress/meetings/2026-11-17.md

S2

Stress design and label amendment

development/stress/PROTOCOL.md development/stress/LABEL_PLAN_AI_V1.md

S3

Smoke counts and interpretation

notes/2026-11-17-stress-execution.md [R]/reviewer/smoke_evidence_audit.json

S4

AI-reference useful yield

[R]/label-evaluation-v1/ai-primary.json [R]/label-evaluation-v1/ai-relaxed_guard.json

S5

Label disagreements and spot review

[R]/label-evaluation-v1/ai-vs-proxy.json [R]/reviewer/ai_label_spot_review.json

S6

Render completion and validation

[R]/reviewer/ipv4_and_render_completion.json [R]/full-review/cohort-validation.json

HISTORY AND READINESS

S7  Pilot failure and study limits

notes/2026-11-16-development-pilot.md notes/2026-11-12-posthoc-review.md

S8  Research context and freeze policy

notes/2026-11-17-research-and-stress-plan.md docs/FREEZE_POLICY.md

CLOSEST-WORK CANDIDATES

P1  CYWS-3D · Sachdeva & Zisserman

Two RGB images under viewpoint changes.

arxiv.org/abs/2308.10417

P2  SceneDiff · Wu and colleagues

Verifiable changes under occlusion and viewpoint shifts.

arxiv.org/abs/2512.16908

[R] = runs/2026-11-17-stress-dev-v1. Paths are relative to the project repository.

Evidence snapshot and file hashes accompany this PDF. AI assisted document preparation. No new experiment ran for this brief.
