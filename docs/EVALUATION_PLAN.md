# Evaluation plan for evidence-aware verification

**Revised 2026-11-17 Pacific. Development is open. Confirmatory testing is not ready.**

The [method](METHOD.md) defines inputs, labels and outputs.
This plan replaces the fresh-study proposal draft of 2026-11-12.
The pinned [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) remains the historical study plan.

## 1. Claims and comparisons

The completed simulation measures image-only performance under declared camera displacement.
Its [development specification](../development/stress/PROTOCOL.md) fixes geometry, preprocessing, controls and decision rules before predictions.
The [completed review](../notes/2026-11-17-scoring-and-label-review.md) leaves support-based triage unavailable.

The [next simulator comparison](../development/cross_sim/PLAN.md) uses recorded simulator interventions for its primary engineering scores.
It tests transfer before another language-model experiment. It does not claim an original stress outcome A, B or C.
The researcher selected Isaac Sim and MuJoCo. The active study includes no real-camera collection.

If recoverable abstentions justify a model, the primary contrast becomes cascade versus image-only verification.
A cascade claim needs more useful answers with a bounded increase in unsupported answers.
An agreement-veto claim instead needs fewer unsupported answers with a bounded useful-yield loss.

These are separate hypotheses. Neither succeeds through coverage alone.

The image-only guard controls and independently calibrated confidence rejection are required comparisons.
A weak model or forced binary prompt cannot establish a contribution.
The operating point comes from calibration arrangements. Report actual evaluation usefulness for every arm.

Matched-coverage curves remain descriptive. They cannot choose the evaluation operating point.
A method claim requires benefit beyond the strongest applicable controls and a suitable change-detection baseline.
CYWS-3D and the two-image SceneDiff setting are candidates for that later comparison, with input and output adapters declared first.
GeoSCD is another candidate. The completed review links its geometric correspondence and occlusion filtering method.

The [VA-TAMP review](../notes/2026-11-17-va-tamp-and-simulation-scope.md) also identifies established evidence-gating and active-perception principles.
These principles and the number of simulators are not contribution claims.
An adapted geometric comparator must disclose its inputs. True simulator depth requires a separate privileged diagnostic arm.

## 2. Data partitions and unit

The independent sampling unit is the base arrangement.
All targets, views, lightings, states, twins and camera changes from one arrangement stay in one partition.
Renaming a scene or changing its seed does not establish independence from a reused base layout.

| Partition | Allowed use |
| --- | --- |
| Old study images and inspected pilots | Development and diagnostics only |
| Development calibration groups | Threshold and operating-point choice |
| Fresh test arrangements | One declared evaluation after the new freeze |

If development data fit a new predictor, reserve calibration arrangements that do not fit it.
For a fixed pretrained model, retain a documented separation between prompt design and final calibration where feasible.
Neither subset becomes confirmatory evidence.

Every primary arrangement must support the declared state and visibility cells.
State balance must not depend on model predictions. Sample identities and weights must exist before inference.

Keep both simulator versions of each arrangement in the same partition.
Report each simulator separately, with paired differences across shared arrangements.
Renderer copies do not increase the independent sample count.
Freeze method settings before transfer evaluation. Report any simulator-specific adaptation separately.

## 3. Label and input audit

1. Export physical state and geometric diagnostics to an evaluator-only table.
2. Inspect candidate support labels using exactly the model's images and task description.
3. Record the cue audit and every disputed label before opening predictions.
4. Check that paths and metadata do not expose state or pair membership.
5. Verify that each model receives one query and no other pair member.
6. Record all exclusions with a model-independent reason.

The researcher authorized AI and programmatic development labels on 2026-11-17.
Follow [the label supplement](../development/stress/LABEL_PLAN_AI_V1.md) for complete geometry proxies and blinded AI review.
Keep their metrics separate. Neither source is a human label.

Independent validation of support judgments remains a paper requirement.
Use programmatic support certificates, a declared human sample or a suitably validated external benchmark for that purpose.
State which endpoint and input subset each source validates.
Record reviewer identity, blinding, exposure, cue reasons and disagreements.
A repeat by one person measures intra-rater consistency only.

Simulator interventions establish physical state under the task definition.
Masks measure geometry. A visibility cutoff cannot establish recognizability or supported removal by itself.
Exact identical inputs across opposite physical states certify unresolved evidence for that pair.
Near-identical inputs still require a cue audit. AI agreement alone does not certify support.

Missing labels remain missing. A geometry proxy alone cannot establish a support-based go decision.
Complete, qualified AI labels can support explicitly provisional development triage under the existing rules.

Qualify the reviewer with independently reviewed clear PRESENT and REMOVED controls.
Declare the qualification criteria before that check. Keep unresolved systematic labeling errors outside decision-making.

The [completed basic screen](../development/stress/LABEL_QUALIFICATION.md) does not release the current AI decision hold.
Its repeated audit remains descriptive. Label completeness cannot substitute for reference validation.

A later probability-sampled label plan must precede predictions and include known inclusion weights and corresponding uncertainty.
Secondary input resolutions require labels for those pixels before a support comparison.

The fresh test fixes its label rules on development. Adjudication remains blind to method outcomes.

Generator failures follow a fixed replacement rule based on validity alone.
No replacement depends on confidence, method disagreement or whether the example is difficult.
Partial visibility cases remain in the data. The support audit determines their analysis category.

## 4. Scores and denominators

Let `y` denote physical state, `s` denote the supported answer, and `a` denote the verifier output.
For an unresolved input, `s = UNKNOWN`.
Support scores use the undisputed set. Factual scores and coverage include disputed cases.

| Score | Numerator | Denominator |
| --- | --- | --- |
| Unsupported definite rate | Valid PRESENT or REMOVED outputs with `a != s` | Support-labeled inputs |
| False-removal rate | REMOVED when the target is present | Physically present inputs |
| False-presence rate | PRESENT when the target is removed | Physically removed inputs |
| Useful yield | Valid definite outputs with `a = s = y` | Support-labeled inputs |
| Present resolution | Supported correct PRESENT outputs | Inputs supporting PRESENT |
| Removal resolution | Supported correct REMOVED outputs | Inputs supporting REMOVED |
| Coverage | Valid PRESENT or REMOVED outputs | All scheduled inputs |
| UNKNOWN rate | Operational UNKNOWN outputs, including failures | All scheduled inputs |
| Output failure rate | Invalid, missing or failed outputs | All scheduled inputs |

Also report definite-answer error among answered cases. It is undefined when no case is answered, not zero.
Show unsupported PRESENT and unsupported REMOVED separately.
Report rates within each physical-state and support cell so class weights cannot hide a failure.

Balanced benchmark averages do not estimate an unknown deployment prevalence.

For each metric, average within arrangement first, using frozen cell weights.
Then average across eligible arrangements. Report the eligible arrangement and input counts for every denominator.
An empty denominator is undefined. A missing required cell is a validity failure, not an automatic zero.

The matched-pair summary includes both-UNKNOWN frequency and each direction of unsupported commitment.
Both-UNKNOWN success requires two valid model outputs or valid gate abstentions. Failed calls cannot satisfy it.
It does not treat pair agreement as verification success.

Latency includes image processing, model inference and the gate. Report cold and warm measurements separately.

Use a common cached base prediction for the veto and confidence arms.
Keep model-call failures separate from valid gate refusals, including registration refusals.
Fix retry and failure rules before evaluation. Missing outputs must not silently reduce an error denominator.

## 5. Development decisions and confirmatory success

The stress pilot applies the rules in its specification separately by environment and level.
Outcome A needs unsupported image-only answers above 0.01, occurring in at least three arrangements at one nonzero level.
The same rule must pass using undisputed support labels and DISPUTED-as-UNKNOWN sensitivity labels.

Do not pool severity levels to meet the rule.
Outcome B needs a useful-yield loss of at least 0.10 at the largest declared level, without outcome A.
A cascade additionally needs answerable image-only abstentions on at least 0.10 of support-labeled inputs at the affected level.
Count a recoverable useful answer only when the definite support label agrees with physical state.
Report conflicting review judgments separately. Keep the original labels unchanged.

Evaluate the declared guard controls before attributing this opportunity to a model.
Otherwise stop expansion on this simulation axis.
The simulation-only scope replaces the original real-development next step with the separately declared paired simulator pilot.
It does not assign a stress outcome or change completed scores.

For a permitted cascade pilot, require `Y_cascade - Y_image >= 0.10` and `U_cascade - U_image <= 0.02`.
Apply both requirements per environment claimed to benefit and in the disputed-label sensitivity analysis.
Require present and removal resolution of at least 0.50 during development calibration.

These development guards guide the pilot. They are not confirmatory margins, statistical significance or safety guarantees.
If a qualified model fails this bounded comparison, stop this cascade design. A later repair requires a distinct development revision.

A future cascade test requires the lower confidence bound for useful-yield gain above its justified minimum effect.
Its upper confidence bound for added unsupported answers must remain below the justified noninferiority margin.
A future veto test instead requires the upper bound for unsupported-rate change below zero.

Its lower useful-yield bound must exceed the negative justified loss margin.
Fix class-resolution floors, contrasts and multiplicity before test collection.
No confirmatory margin or sample size is adopted here.

Calibration uses a finite declared search and whole arrangements.
Choose the lowest unsupported rate among candidates meeting the utility and class-resolution requirements.
Break ties by higher useful yield, then fixed configuration order.

Confidence rejection searches every distinct calibration score boundary, including accept-all and reject-all.
Keep every candidate. If none qualifies, report calibration failure without relaxing constraints.

## 6. Statistical analysis and sample size

Use paired differences on the same arrangements.
Bootstrap whole arrangements within declared environment strata. Carry every nested record with its arrangement.
Never bootstrap views, twins or lighting conditions as independent samples.

Report point estimates, paired intervals, raw counts and zero-event warnings.
A degenerate interval does not prove zero risk or equivalence.
An exact paired binary test applies only to a declared binary outcome per arrangement.

It must not treat all views as independent trials.

Before freezing, select one primary model by a capability and runtime rule that excludes task error rates.
If a second model is declared, report it regardless of whether it supports the finding.
If both models become co-primary, fix a multiplicity procedure before test collection.

Secondary comparisons need explicit labels and a declared correction if they support formal superiority claims.

The sample size follows the paired arrangement variance and the minimum useful effect.
The planning calculation must cover both the error reduction and usefulness requirement.
It must also assess the planned contrasts with the stronger controls if the paper claims a method advantage.

Use conservative variance assumptions and sensitivity cases when the development errors are sparse.
An observed zero must not imply zero planning variance.
Save the precision or power calculation before fixing the test count.

Do not add arrangements until a comparison becomes significant.

## 7. Development sequence

| Step | Work | Required evidence |
| --- | --- | --- |
| Completed v1 | Preserve failed model pilot and provisional audit | Existing source, scores and dated notes |
| Stress design | Generate fresh layouts and check all poses | Hashed specification, sign table, rejection log and cohort manifest |
| Executor check | Terra renders two arrangements at D0 and D6 | Main-reviewer acceptance of actual artifacts |
| Stress pilot | Run image evidence on the full cohort | Complete outputs, guard controls and alignment diagnostics |
| Blind support review | Label the exact primary inputs | Typed review labels, cue reasons and disputed cases |
| Next simulator development | Qualify MuJoCo scene support and compare fixed controls across engines | Shared scene manifest, renderer audits, recorded interventions and separate support diagnostics |
| Conditional model stage | Qualify one model and test a justified combination | Validated support reference, capability checks, calibration and strong controls |
| Readiness review | Assess contribution and precision | Decision to freeze, revise or stop |

All current images and decisions remain development evidence.
Do not change camera levels, sample membership or thresholds after inspecting predictions.
A failed geometry check needs a recorded amendment before rendering, without outcomes guiding the change.

The qualification gate remains mandatory before any later task-model comparison.

## 8. Decisions after development

| Finding | Next action |
| --- | --- |
| Labels or matched inputs fail the audit | Repair the development data and repeat the audit |
| Strong controls already solve the task | Record the ceiling and stop this contribution claim |
| Oracle retention leaves no useful improvement | Stop the veto direction or define a distinct development question |
| Confidence rejection explains the full gain | Remove the claim that image support adds value |
| The image-only verifier matches the useful result | Treat the model as unnecessary unless another declared setting changes the finding |
| Evidence checks improve the joint tradeoff | Plan the fresh evaluation and test the contribution against prior work |
| The result remains imprecise | Report it as a pilot and revise the design before freezing |

These decisions use development evidence. They must not become rules for suppressing an unfavorable valid test result.
Once frozen, report every declared arm and comparison regardless of the outcome.

## 9. Fresh evaluation and scope

The final protocol must name a generalization axis and its held-out identities.
For the initial design, hold out occluder identities and arrangements under a fixed construction rule.
Keep target and background identities controlled, and limit the claim accordingly.

A single held-out family does not establish generalization to all unseen families.
The camera-displacement test measures robustness to that declared change, not general real-world robustness.

The active paper scope is simulation-only verification in Isaac Sim and MuJoCo.
Generalization claims must identify the tested scene, asset and renderer changes.
An engine used during development cannot later be described as wholly unseen.
The final evaluation uses fresh arrangements with fixed method settings.
Physical-camera transfer and deployed robot-memory performance remain future work.

Professor Liu wants to discuss the methodology. No fixed course deliverable is recorded.

Publication readiness remains a separate evidence decision.

The model identifiers, thresholds, weights, margins, sample size, uncertainty settings and collection manifest remain open freeze fields.
The [freeze policy](FREEZE_POLICY.md) requires all of them before a new test.
