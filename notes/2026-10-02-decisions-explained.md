# Decisions made on 2026-10-02, and why

*For the student. Written from the code as it stands today (`verify/`, `analysis/`) and the draft `docs/PREREGISTRATION.md`. Nothing here is final: every item is a* **proposed default — confirm***, and the place to confirm it is the slot named under each heading.*

**How to read the numbers.** Every number tagged **SYNTHETIC** comes from the synthetic generator (30 arrangements, seed 0, default regime unless stated). I re-ran each one today; the runs are listed at the end. They check that the code behaves as designed. They are **not** results and **not** threshold values. Numbers about the geometry (cm, breakpoints, score ties) and the interval bounds are exact and hold for real data at the design geometry.

**References** `[n]` point to the reference list of `notes/2026-10-02-caps-thresholds-literature.md` (the "lit report").

**Background in one paragraph.** The lit report found that our 50 cm example travel cap made F and G behave the same, that G ignored the remaining cap, and that we had no written rule for choosing thresholds. Today's work fixed G and the caps, built `verify/select.py` (the threshold-selection procedure) and `analysis/intervals.py`, checked the PBD-AG supplement, prepared bib entries and drafted `docs/PREREGISTRATION.md`. A reviewer then checked all of it, and a fixer applied the review findings.

---

## 1. Travel caps are measured in arc steps and reported as curves, not tuned

**What we decided.**
- A cap is a number of arc steps. The grid is {1, 2, 3, 4, 6} steps plus uncapped. cap_cm = steps × nominal step, and the nominal step comes from `configs/positions.json` (23.56 cm at the design geometry). That gives 23.56 / 47.12 / 70.69 / 94.25 / 141.37 cm.
- "Is this move affordable?" uses one tolerance everywhere: 1e-6 cm (`schema.TRAVEL_TOL_CM`). That covers the environment's `candidates()` and G's lookahead. No slack is added.
- The CLI takes `--cap-steps` only (`--travel-cap` in cm is gone), so cm slack can't creep back in. The old 50 cm example has been removed.
- Every metric is reported at every cap (the "Cap curve" table in `verify.run`). No cap is chosen from the data.

**Why.**
- 50 cm is 2 steps plus 2.9 cm. After a far first move (P3→P1 = 47.12 cm), nothing else fits, so F and G are forced onto the same path. **SYNTHETIC, re-run today at 50 cm, b = 3:** the old G (now called G-myopic) and F took different paths in **0 of 120** test episodes (0/60 on dev).
- In step units, each cap changes which view sets can be reached: 1 step reaches one neighbour, 3 steps reaches both neighbours (P2 then P4 = 70.69 cm), 6 steps reaches both extremes (P1 then P5 = 141.37 cm).
- Information-gathering work treats the budget as a constraint set by the application and reports results across budgets [14]. View-planning work reports performance against views or path length [6, 26, 27].
- Tuning the cap on dev "so that selection matters" (the old PLAN D4) would give us a free choice that favours G. A curve shows where selection matters without us choosing.

**Rejected alternatives.**
- *A cap tuned on pilot/dev* (EXPERIMENT_PLAN and PLAN D4): rejected for the free-choice reason above.
- *Caps set slightly above each multiple, e.g. +1 cm* (one option in the lit report §2): rejected in favour of an exact 1e-6 cm tolerance. Fixed cm slack is arbitrary, and +1 cm would let a mis-measured real arc pass silently.
- *A cm grid*: rejected, because it hides which view sets each cap makes reachable.

**Cost / risk.**
- **Run size.** Each run is bigger. Today's default run took 17 s and its `logs.jsonl` is about 200 MB. `runs/` is gitignored.
- **Uneven real stops.** If the measured stops are unevenly spaced, "N steps" no longer buys exactly N adjacent moves. `verify.run` warns when this happens. The grid is still defined in nominal steps.
- **Primary cap.** You still need one operating point for the primary contrast. The draft proposes **uncapped**, with the cap curves as secondary results.

**You confirm:** PREREGISTRATION §0 row **D-a** (the deviation from EXPERIMENT_PLAN/PLAN D4), and §6 (`confirm grid` and `confirm primary = uncapped / edit`).

---

## 2. G looks ahead ("reachability feasibility"); F and R stay naive

**What we decided.**
- G now keeps a candidate stop c only if REMOVED is still reachable after moving to c.
  - needed = k − (number of distinct clear-miss stops so far). c itself may supply one of these. The other needed − 1 must be distinct unvisited stops that are reachable within the remaining travel cap **and** the remaining view budget.
  - This is computed exactly by enumerating ordered paths over the 5 stops (`policies.can_visit_more`).
- If no candidate passes, G falls back to choosing among all affordable stops, because it can still find PRESENT.
- Under the k = 1 rule (`-fcm`), every candidate passes, so G-fcm never filters anything.
- G learns the cap, the view budget and k through an optional `begin_episode(limits)` hook. Only G defines this hook. G still sees no evidence it didn't request, and no ground truth.
- **F and R are unchanged.** F takes the first affordable stop in the order P1, P2, P4, P5. R picks uniformly among affordable stops. The lookahead belongs to G's selection intelligence. It is not part of the shared rule.

**Why.**
- `candidates()` already hides stops G can't afford, so masking alone didn't help. G could still spend the cap on a far stop and make REMOVED impossible, which leaves the episode stuck at UNKNOWN. The lit report §4 option 1 was to prune against the *remaining* budget, as RIG does [14]. Applying that to k distinct stops is our own adaptation.
- F and R stand for "no selection intelligence". Giving them the lookahead would change what those baselines measure.

**Evidence (SYNTHETIC, b = 3, test split, 120 episodes, placeholder λ = 0.5).** Episodes where G's path differs from F's:

| Cap | Path differs | Decision differs |
|---|---|---|
| uncapped | 58/120 | 20/120 |
| 1 step | 0 | 0 |
| 2 steps (47.12 cm) | 80/120 | 9/120 |
| 3 steps | 0 | 0 |
| 4 steps | 0 | 0 |
| 6 steps | 58/120 | 17/120 |
| old 50 cm cap | 80/120 (old G-myopic vs F: 0/120) | 9/120 |

- **At 2 steps, b = 3:** G − F is +0.200 recall [0.075, 0.350], +0.075 coverage [0.025, 0.133] and +0.013 FRR [0.000, 0.037] (paired arrangement bootstrap, B = 10,000). So the lookahead helps, but it adds a small FRR cost. On a bigger test set that cost would no longer be small.
- **At 3 and 4 steps, G = F.** This comes from the geometry, not the data. At the start, G's only reference is P3. Its first-move scores (computed today) are P1 = P5 = 1 − λ and P2 = P4 = 0.5 − 0.5λ. So for any λ < 1, G goes to P1, which is also F's first stop. A 3-step cap then leaves exactly one more step (to P2), which is F's second stop too.
- **That identity depends on λ.** At λ = 1.2, P2 scores higher (−0.1 against −0.2), so G's first move is P2. **SYNTHETIC check today (all 180 episodes, b = 3):** at λ = 0.5, 0.7 and 0.9, G vs F paths differ in 0/180 at 3 and 4 steps. At λ = 1.2 they differ in 121/180 at every cap. The synthetic selection picked λ = 1.2 (item 6), so whether G equals F at 3–4 steps really depends on which λ gets frozen.
- **6 steps.** P3→P1→P5 costs exactly 141.37 cm, which equals the cap, so it is affordable. The cap blocks the third move, P5→P4 (164.93 cm cumulative). An earlier chat note said "160 cm, over the cap"; that was wrong and has been corrected in the README and PREREGISTRATION.

**Rejected alternatives.**
- *A shared lookahead for all policies*: rejected, because it would make F and R smarter than "naive".
- *Receding-horizon enumeration ("G-h", lit report §4 option 2) [7]*: **not built.** To score a future step, G needs the obstruction flag of views it hasn't taken yet, because its reference set depends on them. That would mean inventing an observation model. The failure G-h was meant to fix (moves that make REMOVED unreachable) is already covered by the feasibility filter.
- *Greedy nearest-first reachability*: rejected, because it isn't exact. Example: from P2, nearest-first (P1, then P4) needs 4 steps, while P4 then P5 needs 3. A unit test covers this case. The reviewer also checked the enumeration against a brute-force oracle and found no mismatches.

**Cost / risk.**
- G is now "smarter" than in EXPERIMENT_PLAN, so the A − G contrast compares A against a stronger baseline. The draft proposes that A use the same filter (item 4 in your list).
- The feasibility check is optimistic: it assumes the future views will be clear misses. That is deliberate. The check asks "could REMOVED still happen?", not "will it?".

**You confirm:** PREREGISTRATION §0 row **D-c** (`confirm / edit: G as in EP, lookahead as ablation only`), and §4 (`confirm F/R naive`).

---

## 3. G-myopic is kept as an ablation (and G-nolook exists but isn't run)

**What we decided.**
- The pre-revision G is kept, unchanged, as the policy **G-myopic**. It has no lookahead, normalises travel by the fixed 94.25 cm span, and uses λ = 30° (`GMYOPIC_LAM_DEG`).
- G-myopic runs by default. G − G-myopic is a bootstrap pair.
- `G(lookahead=False)`, named **G-nolook**, also exists. It is not in the default run.

**Why.**
- G-myopic shows what the revision bought.
- **SYNTHETIC:** G and G-myopic are identical on every test episode when uncapped (0/120 path differences; the bootstrap rows are flagged `identical`). At 2 steps they differ in 80/120 paths and G − G-myopic has exactly the same numbers as G − F. So the cap is what makes the lookahead matter.

**Rejected alternatives.**
- *Dropping the old G*: rejected, because then nothing would show what the change bought.
- *Running G-nolook by default*: not chosen, to keep the run size down (it would add another policy to every cap × budget cell). Note the trade-off: G − G-myopic changes **two** things at once (the lookahead and the normalisation). G − G-nolook isolates the lookahead alone.

**Cost / risk.** If you report G − G-myopic as "the effect of lookahead", that is not quite right, because the normalisation changed too.

**You confirm:** PREREGISTRATION §4 (`confirm G-myopic as the only G ablation / edit: also run G-nolook`).

---

## 4. G's score is normalised per decision; λ is unit-free; the λ grid comes from the geometry

**What we decided.**
- score(c) = sep(c)/max sep − λ · travel(c)/max travel. Both maxima are taken over G's current pool (the feasible set, or the fallback set), and 0/0 = 0. Ties go to shorter travel, then to the lower stop index.
- λ is unit-free. `Thresholds.lam` defaults to **0.5**, which is a placeholder.
- **The λ grid is {0, 0.7, 0.9, 1.2}.** This is not the {0, 0.25, 0.5, 1.0} the main session first specified.

**Why.**
- Per-decision normalisation follows Delmerico et al. [16]. It makes λ unit-free, and it removes a scale that EXPERIMENT_PLAN would otherwise fit on dev data.
- Nobody derives the travel weight from first principles; it is tuned per environment [29, 30, 31]. So it gets a grid that includes 0 and a rule fixed in advance.
- **Why the grid changed.** G's choice depends on λ only at a few exact breakpoints. Today, `python -m verify.select --lam-breakpoints` gave **2/3, 3/4, 1, 4/3, 3/2 and 2** at the design geometry. Below 2/3, every λ gives exactly the same choices. That has two consequences:
  - The old grid had only **two** distinct behaviours: {0, 0.25, 0.5}, and {1.0}. Also, 1.0 sits exactly on a breakpoint, where the tie-break decides what G does.
  - The placeholder **λ = 0.5 behaves exactly like λ = 0** (no travel penalty) at this geometry.
- The new grid puts one value inside each interval below 4/3, with none on a breakpoint. The tool printed four separate behaviour classes, {0}; {0.7}; {0.9}; {1.2}. For the old grid it printed two warnings.
- `verify/select.py` checks the grid on every run and warns if two values collapse into one behaviour or one sits on a breakpoint.

**Rejected alternatives.**
- *EXPERIMENT_PLAN's dev-fitted normalisation*: rejected, because it adds one more quantity fitted on dev.
- *Normalising by the remaining cap* (lit report §4 option 3): rejected. It goes beyond what [16] does and is harder to explain.
- *The original grid {0, 0.25, 0.5, 1.0}*: rejected, because it can't actually select λ.
- *Values above 4/3*: left out, because travel would dominate G's choice.

**Cost / risk.**
- **The breakpoints move with the measured azimuths.** When the tape-measured arc replaces the template in `positions.json`, the grid has to be re-derived using the same rule, before any dev data are run. The selection code warns if the grid is stale.
- **Tie-break pulls λ up.** Because D10's tie-break is "lower mean travel", λ tends to drift to the largest grid value whenever coverage and recall tie. In the synthetic selection, λ = 0, 0.7, 0.9 and 1.2 all tied on coverage and recall, and λ = 1.2 won on travel (65.2 cm against 69.1 and 77.0). See the "Your decisions" list.
- **This changes your D-norm grid.** The fixer made the change and declared it as deviation D-f. It's your call.

**You confirm:** PREREGISTRATION §0 rows **D-b** (normalisation) and **D-f** (grid), and §7 (`confirm grid and re-derivation rule`, and the tie-break option `b / c`).

---

## 5. The D10 selection objective gets a recall constraint

**What we decided.** Among configurations whose FRR is ≤ 0.10:
1. Keep those with recall ≥ 0.80, if any exist. If none do, keep all of them.
2. Maximise coverage.
3. Break ties by recall, then by lower mean travel, then by grid order.

**Why.** The old D10 ("maximise coverage subject to FRR ≤ 10%, tie-break on recall") could freeze a configuration that satisfied D10 but missed the recall target (lit report §3, Step 0). Putting FRR first and then maximising a secondary quantity is the Neyman-Pearson pattern [19].

**Rejected alternatives.**
- *Recall only as a tie-break* (old D10): rejected for the reason above.
- *A weighted score*: rejected, because the weights would be another free choice.
- *A hard recall ≥ 0.80 requirement with no fallback*: rejected, because selection would then fail outright whenever no configuration reaches 0.80.

**Cost / risk.**
- The "if any" fallback means a frozen configuration can still miss recall. The report says so (`recall >= 0.8 available: False`).
- **SYNTHETIC:** no configuration reached 0.80. The best was 2/20, so this branch was only exercised by unit tests. The builder traced this to the synthetic generator, which yields very few removed episodes with enough diagnostic views. It doesn't come from the rule.
- **The FRR filter uses FRR measured on the arrangements selection runs on, not grouped-CV FRR.** That differs from the lit report's wording. Grouped CV is used only for the honest estimate (item 8).

**You confirm:** PREREGISTRATION §0 rows **D-d** and **D-e** (`confirm selection-set FRR / edit: inner-CV FRR`), and §7 (`confirm all grids, α_clear, FRR limit, recall target, order`).

---

## 6. τ_q and τ_occ come from per-view diagnostic labels (Neyman-Pearson style)

**What we decided (Step 2 in `select.py`).**
- **Null set:** the non-diagnostic views of occluded-present episodes. A view there that passes the gate is a "false clear".
- **Power set:** the diagnostic views of removed episodes.
- **Rule:** keep the gates whose false-clear rate is ≤ α_clear = 0.10, then maximise the clear rate on the power set. Ties go to the lower false-clear rate, then higher τ_q, then lower τ_occ (the more conservative gate). If no gate meets α_clear, take the lowest false-clear gate and log a warning.
- **Grids** are fixed in advance, not taken from data quantiles: τ_q 0.30–0.80 and τ_occ 0.10–0.50, in steps of 0.05.
- **Labels** use EXPERIMENT_PLAN's definition: a view is diagnostic if the annotator can verify the target, or verify that the expected region is unobstructed. Ambiguous views count as non-diagnostic. Labels are stored separately from policy inputs (`cmpe249.view_labels.v1`).

**Why.**
- These two thresholds decide whether a miss counts at all. They can be judged view by view, without running any policy. That makes them simpler and much better supported by data than an episode-level grid.
- Capping one error rate and then maximising the other is NP classification [19].
- Prior systems gate on visibility in the same way [8, 9, 13]. DGSG-Mind reports that a conservative gate "defers decisions under partial visibility and reduces false removals" [13].

**Rejected alternatives.**
- *Joint search over τ_q, τ_occ, τ_match and k on episodes*: rejected. It would multiply the grid and spend the scarce episode-level data on thresholds that the view-level labels already settle.
- *Per-lighting thresholds*: rejected, because there is too little data. We use one global set and break results down by lighting.

**Cost / risk.**
- You need to annotate every dev view.
- **SYNTHETIC:** the labels are a stand-in taken from the generator's hidden layout. They are not real annotation. Selected gate: τ_q 0.40 and τ_occ 0.15, with 0/84 false clears (CP upper bound 0.035, optimistic because views are correlated) and 16/16 clears.
- **τ_occ has two jobs.** It is the clear-miss gate in the stopping rule (obstruction ≤ τ_occ). It is also G's occlusion flag for its reference set (obstruction > τ_occ). So choosing τ_occ changes which views G picks. It also means the planned gate-off ablation (τ_occ = 1.0) is not a clean "gate off" for G (lit report §2 and §6 item 4).

**You confirm:** PREREGISTRATION §7 (`confirm the diagnostic-label definition and the Step 2 null and power sets`), and §4 "τ_occ has two jobs" (`i / ii / iii`).

---

## 7. τ_match and k come from a fixed-sequence walk

**What we decided (Step 3).**
- **Walk order** is fixed in advance: k = 3 first, then k = 2. Within each k, τ_match runs 0.40, 0.45, …, 0.70 ascending. A lower τ_match makes PRESENT easier, so it is the more conservative setting.
- **Pass rule:** a step is one (k, τ_match) pair. It passes only if its **worst** FRR over the whole λ grid is ≤ 0.10.
- **Accepted set:** the prefix of steps before the first failure. D10 (item 5) then picks among the accepted (k, τ_match, λ) configurations.
- **If step 0 fails,** the most conservative step is reported with `meets_frr_limit = false`. It is never accepted silently.
- **Monotonicity** is checked on the data, and any violations are written into the report.

**Why.**
- Learn-then-Test's fixed-sequence idea [17]: walking a pre-ordered grid and stopping at the first failure limits how many looks the data get.
- The arrangement is the unit, because LTT assumes i.i.d. calibration data [17].
- Taking the worst case over λ means the later λ choice can never be what rescues the FRR limit, so λ stays a secondary knob.
- We check monotonicity rather than assume it, because with an adaptive G it isn't guaranteed [18].

**Rejected alternatives.**
- *Full grid search with a max over everything*: rejected, because it gives the data many more looks.
- *Walking each k separately, or interleaving k with τ_match*: offered as an edit option but not the default.
- *A formal LTT p-value guarantee*: not possible with about 10 arrangements (item 9).

**Cost / risk.**
- Because the order is k-major, **one failing k = 3 step removes every k = 2 configuration**, even one with lower FRR. "Conservative → liberal" is an order we declared, not one we proved.
- If k = 3 is frozen, recall at b = 1 is 0 by construction. b = 1 allows only 2 distinct stops, and REMOVED needs 3.
- **SYNTHETIC:** all 14 steps passed, and the worst step was k = 2, τ_match 0.70, with 1/40. There were no monotonicity warnings. The candidate was k = 2, τ_match 0.40, λ = 1.2.

**You confirm:** PREREGISTRATION §7 (`confirm k-major / edit: interleave / walk each k separately`), and §6 (`confirm frozen k at every b / edit: k = min(k*, b+1)`).

---

## 8. Nested grouped cross-validation for the dev estimate

**What we decided (Step 4).**
- **Outer loop:** leave one arrangement out.
- **Inner loop:** the whole Step 2 + Step 3 rule, re-run only on the other arrangements. τ_q and τ_occ can therefore change between folds.
- **Dev estimate to quote:** the held-out decisions, pooled over folds.
- **Final candidate:** the same rule applied to all 10 dev arrangements, written to `frozen_candidate.json` with its sha256. The status is "candidate", and it becomes frozen only when you sign FREEZE.md.
- **Split refusal:** `select.py` refuses test or pilot episodes (`SplitRefused`), including any store or label file that contains one.

**Why.** Reporting the cross-validated score of the configuration you tuned is optimistic. On null data, nested CV was "almost unbiased" while the non-nested estimate wasn't [21].

**Evidence (SYNTHETIC).**
- On all dev data (resubstitution), the candidate scored recall 2/20 and coverage 40/60.
- Pooled over held-out folds, recall was **0/20** and coverage 38/60. The folds picked 3 different configurations.
- So the optimistic numbers really were optimistic. The builder traced both recall hits to a single arrangement.

**Rejected alternatives.**
- *Plain (non-nested) grouped CV*: rejected, because it is biased [21].
- *Leave-two-out as the outer loop*: offered as an edit.
- *A third CV level inside Step 3*: not used. Step 3 fits nothing beyond the grid choice.

**Cost / risk.** The estimate is noisy with 10 folds. It describes how well the *rule* does, not the exact frozen configuration.

**You confirm:** PREREGISTRATION §7 (`Outer CV … confirm / edit: leave-two-out`).

---

## 9. Intervals: Clopper-Pearson, Jeffreys and the cluster bootstrap, and what can't be guaranteed

**What we decided.**
- **Exact bounds:** one-sided 95% Clopper-Pearson (CP) bounds: upper for FRR, lower for recall and coverage. Jeffreys bounds are reported next to them. Both are in `analysis/intervals.py`, written with stdlib only.
- **Arrangement-level FRR:** also reported, counting an arrangement as failed if it has any false removal.
- **Bootstrap:** a percentile cluster bootstrap over arrangements for single rates, and a paired arrangement bootstrap for policy differences.
- **Integer counts only:** fractional counts (R's seed-averaged decisions) are refused, which is why selection uses a deterministic policy.

**Why.**
- CP is exact and conservative. Jeffreys is recommended for small n [22].
- The arrangement is the independent unit, so the arrangement-level bound and the cluster bootstrap are the honest ones [20].

**One-sided 95% bounds** (computed today):

| Count | CP | Jeffreys |
|---|---|---|
| FRR dev 0/40 | 0.0722 | 0.0466 |
| FRR dev 1/40 | 0.1132 | 0.0936 |
| FRR dev 2/40 | 0.1492 | 0.1316 |
| FRR dev 0/10 arrangements | 0.2589 | 0.1708 |
| FRR test 0/80 | 0.0368 | 0.0236 |
| FRR test 3/80 | 0.0941 | 0.0855 |
| FRR test 4/80 | 0.1108 | 0.1025 |
| FRR test 0/20 arrangements | 0.1391 | 0.0905 |
| recall dev 16/20 (lower) | 0.5990 | 0.6273 |
| recall test 32/40 (lower) | 0.6680 | 0.6821 |

**What can't be guaranteed.**
- **On dev, one false removal already rules out certifying FRR ≤ 10%:** 1/40 gives 0.1132. (The brief said 0.1129. That was wrong; the reviewers confirmed 0.1132 independently, and I recomputed it today.)
- **At the arrangement level, even a perfect dev run only bounds FRR at 0.259** (0/10).
- **No formal (α, δ) guarantee is possible on dev.** The NP order-statistic method needs about 22 *independent* units [19]. Conformal risk control gives only an expected-value guarantee [18].
- So dev is a **selector**, and FRR ≤ 10%, recall ≥ 80% and coverage ≥ 80% are **goals**.
- **On test,** "FRR ≤ 10%" is supported at one-sided 95% only with **at most 3 false removals out of 80**.

**Rejected alternatives.**
- *Wald or normal-approximation intervals*: they fail at 0/n [22].
- *LTT or conformal guarantees on dev*: not achievable at this sample size.

**You confirm:** PREREGISTRATION §9 (`confirm / edit`).

---

## 10. What is fixed now and what is set after the rig

**Fixed now, in writing:**
- the questions and hypotheses, and the "inconclusive" rule;
- the episode design and the split *procedure*;
- the policy and stopping-rule logic;
- budgets b = 0–4, with b = 3 primary;
- the cap grid in steps;
- the whole selection procedure: its grids, order, D10, CV and interval methods;
- k ∈ {2, 3}.

**Set after the rig, by the procedures above:**
- the measured stop positions (after the pilot, before dev capture). Then **re-derive the λ grid**;
- τ_q, τ_occ, τ_match, k and λ;
- the lighting definitions (D3), the Lux guard (D11), the pose thresholds (D17) and the detector/encoder (D9);
- the split seed, drawn once the layout cards exist.

**Why.** Score distributions depend on the camera, mount, lighting and crop. Prior work re-tunes per sensor [10, 11, 13], and constants carried over from simulation hold only partly [4, 7]. Fixing the *procedure* now and the *values* later is what keeps the values honest.

**Cost / risk.** Nothing synthetic may enter FREEZE.md; the candidate JSON records its synthetic provenance so this can be checked. The split function is **not stratified**, although PLAN asks for strata. That is your choice to make.

**You confirm:** PREREGISTRATION §8 (table), §2 (`split seed` and `confirm unstratified / edit: add a stratified wrapper`), and §15 (G1–G6, D1, D2, D2b, D15, D16, which are now in the draft).

---

## 11. PBD-AG verdict: the distinction holds (supplement not found)

**What we decided.** The project's distinction from PBD-AG [2] holds, based on its public frozen protocol. That protocol is in a repository that is most likely the authors' but isn't confirmed. The supplement PDF itself could not be found on arXiv, the project page, OpenReview or by web search.

**Why.** The paper's "paired acquisition experiments" rank *which fixture to inspect next* and are scored on mapping F1 in one simulated scene. They do not choose views to decide whether a remembered object is present, removed or unknown. They have no UNKNOWN output, no occluded-present controls, no shared stopping rule and no per-policy false-removal metric.

Two further results:
- Re-reading PInVerify [1] confirmed that its final answer is binary ("Unsure" appears only mid-episode). That strengthens the UNKNOWN part of our distinction.
- Build Once, Monitor Continuously [10] is new to NOVELTY_CHECK. It supports the motivation: its authors say it "cannot tell a missed detection apart from a real removal".

**Not claimed.** Comparing policies on paired seeds under matched budgets is standard practice (PBD-AG does it too), so it is not part of our contribution. H1 is described as a *transfer* of PBD-AG-style gated removal to a monocular 2D cue, not a new mechanism.

**Cost / risk.** The supplement's numbers are unread, and the protocol may not match what was finally submitted. Re-check arXiv and the AAAI-27 proceedings before the final report.

**You confirm:** PREREGISTRATION §15 **G1**. The addendum is in `docs/NOVELTY_CHECK.md` ("Addendum 2026-10-02").

---

## 12. Other things builders and fixers decided that you should know

- **Thresholds are selected under G, then shared by every policy.** Selecting them on G's performance could favour G. F is the neutral option, but under F λ is never selected: G and A would run at the placeholder 0.5, which behaves like λ = 0. Choosing F therefore also needs its own λ rule. → §7 (`confirm G / edit: F, with λ chosen by: ____`).
- **A (not built yet) should use G's normalisation, λ and reachability filter.** Otherwise A − G would mix the learned utility with lookahead. → §4.
- **A-nocue** (from EXPERIMENT_PLAN's comparison matrix) is kept in the draft and marked not implemented. → §0 D-g, §4.
- **H1 contrasts now exist:** paired bootstraps X − X-fcm for F, R, G and G-myopic. **SYNTHETIC, uncapped:** G − G-fcm FRR −0.075 [−0.138, −0.025], recall −0.475 [−0.675, −0.300]. This is the expected trade: the k = 2 rule makes fewer false removals but also finds fewer removals.
- **"Inconclusive"** now means only "the 95% paired bootstrap CI contains 0". The vague "wider than the effect" clause is gone. No equivalence claim will be made unless you declare a margin. → §1.
- **λ validation:** NaN, inf and negative λ are now refused. Before this fix, NaN silently turned G into a shortest-travel policy.
- **Provenance:** the candidate JSON records the synthetic generator settings, so the default and block-anywhere synthetic runs no longer share a settings hash.
- **Pre-registration hygiene:** §0 lists every change from EXPERIMENT_PLAN/PLAN. The document governs *only* for those listed items. Every choice has a `[ ] confirm / edit:` slot, and an unticked item is not pre-registered.
- **Bib entries:** 14 entries are ready as a patch that hasn't been applied (`git apply progress/references-additions.patch`). Four are arXiv-only preprints. Lee et al. [20] is not in the patch.
- **Specified, not yet implemented:** A, A-nocue, diagnostic-view coverage (§11.3), cue false-clear rate (§11.4), the gate-off ablation (§11.5), risk-coverage curves, and the optional angular-FPS selector (D8).

---

## Your decisions to make (most important first)

1. **Which policy the shared thresholds are selected under:** G (default) or F. If F, also say how λ is chosen. (§7)
2. **The λ tie-break** (§7 option b/c). With "lower travel" ahead of grid order, selection drifts to the largest λ, and λ = 1.2 changes G's very first move (P2 instead of P1). Choose whether the smaller λ should win ties. Also confirm the new grid {0, 0.7, 0.9, 1.2} and the rule for re-deriving it once the arc is measured. (§0 D-f, §7)
3. **The primary cap:** uncapped (proposed) or a step cap. At λ < 1, the 3- and 4-step caps cannot separate G from F, which is a geometric fact. (§6)
4. **The deviations table:** caps as curves (D-a), per-decision normalisation (D-b), G's lookahead (D-c), the D10 recall filter and worst-over-λ walk (D-d), selection-set FRR versus inner-CV FRR (D-e). (§0)
5. **τ_occ's double role:** (i) keep one threshold, (ii) give G its own occlusion flag (a small code change), or (iii) run the gate-off ablation for F and R only. (§4)
6. **The A design:** A shares G's normalisation, λ and filter; keep or drop A-nocue. (§4)
7. **Step 2 definitions:** the null and power sets, ambiguous views as non-diagnostic, the grids and α_clear = 0.10. (§7)
8. **Walk order:** k-major (default), interleaved, or each k separately. Also whether k stays frozen at every b, or k = min(k*, b+1). (§6, §7)
9. **The split:** the seed, and stratified or unstratified. (§2)
10. **Ablations to run:** whether to add G-nolook to the runs. (§4)
11. **G1 and the rig/gate items:** G1–G6, D1, D2, D2b, D15, D16. (§15)
12. **Housekeeping:** apply the bib patch, add Lee et al. [20], decide whether to add Build Once [10] to the survey, and push PREREGISTRATION.md once you have ticked it.

---

*Runs behind the numbers in this note, all re-run on 2026-10-02 in the session scratchpad, all SYNTHETIC:*
- *`uv run --no-sync python -m verify.run --synthetic` (default caps), and the same with `--cap-steps 2.122065907891938` (= 50 cm);*
- *`uv run --no-sync python -m verify.select --synthetic` (candidate sha256 `f295a899…c6c13`, identical to `runs/select-synthetic-n30-s0`);*
- *`uv run --no-sync python -m verify.select --lam-breakpoints`, with and without `--lam-grid 0,0.25,0.5,1.0`;*
- *a short script that counts F vs G path differences for λ ∈ {0.5, 0.7, 0.9, 1.2} at b = 3;*
- *the bounds in item 9, computed with `analysis/intervals.py`, and G's first-move scores with `G.scores`.*
