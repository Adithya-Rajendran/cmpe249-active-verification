# Addenda to the freeze `simulation-freeze-1`

These are dated changes made after the freeze (SIM_PREREGISTRATION §13, FREEZE.md).
- Earlier text is never edited.
- No file in the freeze record changes.
- Each addendum says what changed, why, and whether any test result had been seen.

## Addendum 1 (2026-11-06, 10:35 PST): the timestamp went to the researcher's second mailbox

**What changed.** FREEZE.md (its header and §2, "Precondition for step 1") requires the researcher to send the freeze commit id and FREEZE.md's sha256 to the instructor as an outside timestamp before any test arrangement is generated (decision D2(a)).
- **What the researcher did instead.** They sent the two values to another mailbox of their own, for time keeping. They will send them to the instructor later, once the research direction is more concrete.
- **Approval.** The researcher approved the change on 2026-11-06 ("That is fine change the requirement").
- **The new precondition.** The researcher has confirmed that the timestamp was sent to their second mailbox. They confirmed it on 2026-11-06, before this addendum.

**Why.** It is the researcher's decision: the instructor email waits for a more concrete direction.

**What it changes about the evidence.** The outside timestamp is now an email the researcher holds, and its mail headers carry its time. That is weaker than a copy held by the instructor. An email sent to the instructor after the test is not a pre-test timestamp.

**Test results seen.** None. When this was written, no test arrangement had been rendered or scored. None had been generated either, apart from the one disclosed in `notes/2026-11-06-test-seed-incident.md`.

**Where the runbook runs.** FREEZE.md §2 says to run from a clean checkout at the freeze commit (6d0ed94a73a20ff41549a2383b07f0358bf69ad9). The runbook now runs from the commit that adds this file, a child of the freeze commit.
- That commit changes no file in the freeze record and no shipped code path. Three checks verify both:
  - `sim/freeze.py` `require_freeze` checks the committed bytes and the record's ancestry;
  - the pipeline's frozen-code check runs `git diff` against the record commit;
  - `tools/test_validity.py` checks the record's hashes.
- A generated arrangement records FREEZE.md's sha256, the freeze id and sim/testsplit.py's sha256, not a commit. So no arrangement changes.

**Unchanged:** the code, models, thresholds, stopping rule, scenes, seeds, sample sizes, primary comparison and analysis.

## Addendum 2 (2026-11-06, 10:35 PST): interpretation limits, recorded before any test outcome (reporting only)

**What changed.** Nothing in the analysis. On 2026-11-06, before any test arrangement was generated, the researcher instructed that the results be read with the limits that a post-hoc methodology audit found on the development data (`runs/2026-11-06-methodology-audit/audit.json` and `assessment.json`, untracked; no test data). The limits:
- **L1: A's history-only features cannot reorder candidates.**
  - Four features (obstruction_start, side_missing, match_max, quality_min) are constant across the candidate views of one history, so in A's additive logistic score they cannot change the order of the candidates.
  - Zeroing them changed 0 of 5,040 A replays, and zeroing A-nocue's two such terms changed 0 of 5,040 A-nocue replays.
- **L2: A's target.** A is trained to predict geometric diagnostic visibility, not terminal verification value.
- **L3: no remaining-budget input.** At b = 3 with a two-step travel cap, A resolved 0 / 240 development removal episodes. Dir resolved 139, and an evidence-aware ordering oracle 145.
- **S1: the generator restricts occlusion complexity.**
  - There is one designated occluder, and distractors never occlude the target.
  - Every scene but IND28 has two clear stops.
  - Every episode starts at P3.
- **S2: little room at the primary.** At b = 3 uncapped, over the four development environment × lighting cells:
  - the ordering oracle exceeds G by 6 of 240 removal episodes (196 vs 190);
  - on present episodes, G equals the oracle (114 / 120 per cell).

  The oracle reads future evidence. Its counts bound those development episodes only. They are not a test result, and they do not guarantee what a learned policy can attain.

**How the results are reported:**
- as results about this restricted A in these scenes;
- a weak or inconclusive A − G is not evidence that learned active perception is ineffective, and inconclusive is never equivalence (FREEZE.md §5);
- development findings stay apart from the test results.

**Test results seen.** None.

## Addendum 3 (2026-11-12): post-hoc analyses after the test (reporting only)

**What changed.** Nothing in the frozen analysis, and no reported estimate. On 2026-11-12, after every test outcome had been read and reported, an independent review of the completed study added analyses that the frozen plan does not contain. They are recorded in `notes/2026-11-12-posthoc-review.md`:
- **A replay check.** A separate implementation of the stopping rule and of F, G, G-lookahead, Dir, A and A-nocue gives the saved decision, path and travel in 172,800 of 172,800 runs.
- **Exact tests.** An exact sign test on the discordant arrangements of each paired contrast, beside the declared bootstrap interval. The note also states that the two lighting cells of an environment use the same arrangements.
- **Fixed orders.**
  - G's order on the test set, and all 24 fixed orders at the primary setting.
  - A fixed order chosen on the development data (Fix), as a control that reads no image to choose a stop.
  - A per-episode hindsight ceiling.
- **Visited stops.** The stops that A and Dir visit at the primary setting.
- **Safety margin.** The clear misses among all five views of the present episodes, and the separation of the match scores.
- **Calibration.** A recount of the frozen calibration, and the alignment refusal counts.

**Why.** The researcher asked for an independent review on 2026-11-12 and chose to record its analyses as a dated post-hoc addendum. The review found that three readings of the results needed a fixed-order control or an exact test:
- the shelf L2 interval of the primary comparison;
- the comparison with the fixed order F;
- the equal decisions of A and Dir.

**What it changes about the evidence.**
- **Every analysis here is post hoc.** The analyses to run, the rule that chooses Fix and the contrasts to report were all decided with the test outcomes known. The rule that chooses Fix reads development data only, but it was written after the test.
- **They are exploratory.** None is a confirmatory result, none replaces a declared comparison, and none is promoted to the primary.
- **The frozen results stand as reported** in `notes/2026-11-06-test-results.md`. Where the new note qualifies a reading of that note, its §8 says so.

**Test results seen.** All of them.

**Unchanged:** the code, models, thresholds, stopping rule, scenes, seeds, sample sizes, primary comparison and analysis, and every file in the freeze record.

## Addendum 4 (2026-11-16): close the historical freeze and separate the new method

**Request.** The researcher requested a repository cleanup, a revised method and plan, and a cleanup of the flawed freeze.
They also replaced the separate instructor report with the Canvas progress assignment.

**What changed.** The active research documents now describe a separate study of evidence for a memory update.
Its design is in [METHOD.md](METHOD.md), [EVALUATION_PLAN.md](EVALUATION_PLAN.md) and [FREEZE_POLICY.md](FREEZE_POLICY.md).
It has no active freeze, new test data or experimental result.
The older local reports and instructor packet moved to a dated archive with their source files and hashes.

**Status of this freeze.** `simulation-freeze-1` is closed to new research decisions.
It remains the historical record for reproducing the completed study.
Its code, manifests, tag, thresholds, seeds and results remain unchanged.
The old test data are development data for every later method.
A passing historical hash check does not authorize a new study's test.

**Why.** The completed study had little selector headroom and no attainable false removal under its frozen test evidence.
The ranker's history terms, training target and budget inputs further limit the interpretation.
Addenda 2 and 3 and the post-hoc review retain those limitations.
These are limits of the comparison, not evidence that an unreported successful method existed.
The seed incident and timestamp exception remain disclosed. This addendum does not repair their history.

**Test results seen.** All results of the completed study.
No new method was run during this document reset.
The new design is explicitly informed by the old outcomes.
Any confirmatory test of it needs a distinct protocol and fresh data.

**Unchanged.** Every file listed in the old freeze manifest and the original FREEZE.md bytes.
Earlier addenda remain intact. No new freeze tag, commit, push or external submission accompanies this addendum.
