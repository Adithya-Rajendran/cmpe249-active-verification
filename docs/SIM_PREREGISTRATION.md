# Pre-registration: the simulation study

**Status: FROZEN with docs/FREEZE.md (simulation-freeze-1, 2026-11-06).**
- The design choices were confirmed by the researcher on 2026-11-02 / 11-03.
- The development loop, including A, is finished. Its record is §14 and the notes cited there.
- The researcher confirmed this document in its final review on 2026-11-06: every open slot as recommended, with the
  review's corrections (the test-seed incident record, the zero-width interval flags, pointwise and unadjusted
  intervals, and the development record's wording). It is frozen with FREEZE.md before any test arrangement is
  generated for the study.
- One shelf test arrangement (slot 0) was generated temporarily by a unit test on 2026-11-05 and deleted unseen: it was
  not rendered, scored or viewed (§14; `notes/2026-11-06-test-seed-incident.md`). The seed schedule is kept.

It succeeds the earlier physical-study draft, which was never confirmed and is archived in the researcher's local
progress folder.
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) holds the design.
- [`SIMULATION_STUDY.md`](SIMULATION_STUDY.md) is the amendment.
- This document fixes the analysis rules.

**Order of work:**
1. Finish the development loop: generator quotas, dev renders, evidence, threshold calibration, the S/F/R/G replay, and building and training A. **Done (2026-11-06; §14).**
2. Review this document. **Done (2026-11-06).**
3. Freeze it (§12). **Done: docs/FREEZE.md (simulation-freeze-1).**
4. Only then generate any test scene.

**Naming.**
- *Simulated* means images rendered by Isaac Sim from the rig replica (`sim/`); every course result is a simulation result.
- `verify.run --synthetic` means randomly generated evidence with no images. It is used only for pipeline tests and never appears in results.

## Decisions in force (the researcher, 2026-11-02 / 11-03 / 11-05; also in the amendment and DECISIONS)

- **Bounded completion** (2026-11-05):
  - The study is finished as declared and is not expanded to make A beat G.
  - A - G stays the primary at b = 3, uncapped, k = 2, lam = 0.
  - A - G-lookahead is reported across the caps.
  - A deterministic direction rule Dir, built on the same descriptor as A, is an explanatory control. Its origin as
    a post-hoc development idea is documented (`notes/2026-11-05-side-descriptor.md`).
  - No favourable cap or subgroup is promoted to the primary endpoint.
  - New models, registration methods, scene redesigns and physical studies are outside this study.
- **Registration** (Amendment 2 of `notes/2026-11-04-registration-repair-plan.md`, 2026-11-05):
  - The registration search is closed, and development continued with C0 unchanged in both environments.
  - C0's failure of its own tabletop refusal criterion stands: 12 / 165 views refused in L1 and 9 / 165 in L2 on the
    first 11 arrangements.
  - A refused view cannot give a clear miss, but it can still give PRESENT through the strong-match rule.

- **Simulator:** Isaac Sim 6.2 on the GB10 is the single simulator; MuJoCo is a geometry cross-check only.
- **Environments:** shelf (the measured rig replica), then tabletop. **No third environment.** Claims concern **new arrangements within these two environments**, not generalisation to unseen environment types.
- **Cameras:** the option-A arc with per-stop pitch, so compartment #3 stays in frame. The five-stop arc is kept in both environments; no extra cameras (confirmed by the researcher, 2026-11-06).
- **No fiducials** (2026-11-03). The corner ArUco tags are removed from every simulated scene, dev and test, because a home robot's shelf has none.
  - Evidence must align each query view onto its reference from the images alone (evidence v0.3), with no masks, labels or poses.
  - Frozen v0.2.1, the baseline, then runs unaligned.
  - The simulator's exact camera poses give a ground-truth query-to-reference alignment per view. It is used only to score the alignment, never as an input.
  - The v1/v2 check sets, rendered with tags, are kept as records.
- **References:** queries at L1 and L2 are compared with the **L1 reference** at their stop.
- **Reporting:** every result is reported **separately for L1 and L2**. A pooled number appears only as a labelled secondary line.
- **Policy G:** primary G has the lookahead **off**; G-lookahead and G-myopic are ablations.
- **Thresholds:**
  - τ_match, τ_q and τ_occ are calibrated **per view** on dev, **jointly** over the full grid, with no policy run (select-v4, §7; joint search since 2026-11-03).
  - k = 2 and λ = 0 are fixed, with k = 3 and the other λ classes as sensitivity runs.
  - FRR ≤ 10 % is a dev check, not an objective.
- **Primary operating point:** **uncapped travel, b = 3 additional views.** This is described as *view-budgeted verification*, with travel reported alongside accuracy and coverage. The finite-cap curves are secondary. b = 1 and b = 2 are secondary and are **not** promoted on the basis of dev results; the old D14 trigger is withdrawn.
- **Counts:**
  - Per environment: **60 dev + 100 test random arrangements**, with DES0–DES3 as separate dev checks outside the 60.
  - This is a practical allocation, **not a power guarantee**.
  - **Render time.** The first planning estimate was about 5.4 h for the 320 random arrangements, at about 61 s each (`runs/2026-11-02-sim-prereg-review/planning_checks.json`). That was measured before the ghost fix, which renders every capture with fresh render products. With the fix, the same 10 shelf dev arrangements took 120.9–126.4 s each, median 124.4 s (`data/raw/2026-11-03_sim_pilot/set_checks.json`). That gives about 11 h of rendering for 320. With revisit noise (checkpoint 2), the 10-arrangement check batch took 124.5–130.6 s per arrangement, median 128.6 s, and 1300 s in all including boot and shader compile (`shelf_dev_v1_check/render_report_20261103T000139.json` on the share). That is about 2.2 h for 60 and about 11.6 h for 320.
  - **Measured development throughput** (median render time per arrangement): shelf 128.6 s (the check batch above) and 127.9 s (the renderer check of 2026-11-06, FREEZE.md §4), tabletop 176.0 s (the 49, `notes/2026-11-06-tabletop-dev49-transfer.md`). So 100 + 100 test arrangements take about 3.6 h + 4.9 h of rendering; evidence takes minutes. The counts stand (confirmed by the researcher, 2026-11-06).
- **Occluder quotas** (fixed before rendering, §2):

  | Family | Share | Dev (per env.) | Test (per env.) |
  |---|---|---|---|
  | S8 | 20 % | 12 | 20 |
  | M16 (left/right balanced) | 30 % | 18 (9 + 9) | 30 (15 + 15) |
  | IND28 (intended to stay hidden) | 10 % | 6 | 10 |
  | White box | 20 % | 12 | 20 |
  | YCB box | 20 % | 12 | 20 |

- **α_clear = 0.10:**
  - It is a predefined **empirical per-view calibration constraint**. It does **not** guarantee that any policy's FRR is ≤ 10 %, and requiring two clear-miss views does **not** make the error α².
  - If no setting meets it, calibration fails loudly (§7).
- **Occlusion-gate-off ablation:** reported for **F and R only**.
- **Revisit noise** (2026-11-03):
  - **Query captures only.** At query time, cameras get small, bounded, seeded pose perturbations. References stay at the nominal poses.
  - **One draw per (arrangement, stop).** That revisit pose is reused across visible / occluded / removed, L1 / L2, and every policy, so matched episodes differ only in the scene.
  - **Recorded:** the ranges are documented in the generator config and in FREEZE.md.
  - **Labels:** geometric labels are computed from the actual perturbed renders.
  - **Checked first:** a small dev batch is checked before the full dev batch.

---

## 1. Questions and hypotheses

- **Primary question.** Under view-budgeted verification (b = 3, uncapped), does the learned ranker **A** improve on the geometric selector **G** in FRR, removal recall and coverage? The primary contrast is A − G on test, reported separately for L1 and L2 and for each environment. Confirmed by the researcher, 2026-11-06.
- **Baselines.** A − F, A − R, G − F, G − R, and each multi-view policy against S (at its actual effort).
- **Added 2026-11-05 (secondary):**
  - A − G-lookahead along the travel caps;
  - A − A-nocue (does the occlusion cue matter);
  - A − Dir (explanatory: does learning add anything over the deterministic rule on the same descriptor).

  None of them replaces the primary. Confirmed by the researcher, 2026-11-06.
- **Hypotheses:**
  - **H1:** for the same selector, k = 2 gated removal gives lower FRR than first-clear-miss removal (`-fcm`), possibly at lower coverage.
  - **H2:** A improves the error / coverage / effort trade-off over F and R.
  - **H3:** any A − G advantage depends on the occlusion family and on lighting.

  Confirmed by the researcher, 2026-11-06.
- **Inconclusive.** A contrast is inconclusive when its 95 % paired cluster-bootstrap interval (§8) contains 0. Inconclusive is never reported as equivalent, and no equivalence margin is declared. Each interval is pointwise and unadjusted (§8), so a conclusion is stated per environment, lighting and metric. Confirmed by the researcher, 2026-11-06.

## 2. Unit of analysis, seeds, quotas and split

- **Episode** = (arrangement, condition ∈ {visible, occluded, removed}, query lighting ∈ {L1, L2}), so 6 per arrangement. Truth is PRESENT or REMOVED. UNKNOWN is an abstention.
- **The independent unit is the arrangement.** Resampling, CV folds and the split are by arrangement. R's seeds are averaged within an episode and add no samples.
- **Seed blocks per environment:**
  - **Dev:** seeds 0–9999, plus DES0–DES3.
  - **Test:** a contiguous block per environment: shelf from 100000, tabletop from 110000. Each has per-family reserve
    lists (shelf 101000+, tabletop 111000+), as in the generator configs.

  No test seed is rendered, scored or viewed before the freeze. (A unit test generated the shelf test arrangement of slot 0 temporarily on 2026-11-05 and deleted it without rendering, scoring or viewing it: `notes/2026-11-06-test-seed-incident.md`. The seed schedule is kept.)
- **Quotas are slots, not weights.**
  - **Slot list.** Before rendering, a fixed slot list per split and environment assigns every arrangement its occluder family, and its M16 side. The list comes from a seeded permutation of the quota table above.
  - **Retries and replacements.** If a seed's placement fails, the generator retries deterministically within the same family: seed, attempt 1, 2, …, up to a fixed maximum. After that it takes the next seed from that family's reserve list. **It never falls back to another family.**
  - **Retry maximum: 20** attempts per seed, as proposed (generator config, 2026-11-03). Each family has its own reserve seed list; a replacement never changes the family, the M16 side or the target.
  - **M16 is placed asymmetrically.** M16 is the physical board, 16 cm wide and 12 cm tall (`configs/sim/arrange_shelf_v3.json`; a taller simulation-only variant, v2, was tried and dropped on 2026-11-03). The board is shifted to one side of the target's P3 sight line, with left and right balanced across arrangements (dev 9 + 9, test 15 + 15). **Side rule (v3):** the P3 view is hidden, the adjacent same-side stop is covered ≥ 0.95, the nearest opposite-side stop is covered ≤ 0.10, and at least 2 stops are clear. The far same-side stop is unrestricted, so partial patterns such as pHHCC and CCHHp are accepted (researcher, 2026-11-03). Pure patterns are CHHCC (left) and CCHHC (right).
  - **Targets are slots too** (researcher, 2026-11-03). Within each slot class (and each M16 side) the targets come from a seeded, balanced list, fixed per slot through every retry and replacement. A (target, family) pair is left out of a class's list only when **no valid placement was found in the tested generator search** (0 of 100 tries; not a proof that none exists). Under v3 that leaves: S8 without the tomato can; M16 (both sides) and the YCB box with the foam brick and tuna can only; IND28 and the white box with all five. The target composition per class is reported with every result (§9).
  - **Logging.** Retries and replacements are logged.

## 3. Conditions, occluders and geometric labels

- **Revisit frame check** (tag-free, 2026-11-03). A revisit pose draw is accepted only when compartment #3's outline and the target's projected hull both stay inside the frame (margin ≥ 0 px), at the nominal and the revisit pose; otherwise it is redrawn, deterministically, and the redraw is logged. Ranges: ±10 mm translation and ±0.5° rotation per axis.
  - **Tabletop (2026-11-04 design).** The tabletop has no compartment: the outline is the **workspace outline** (24 points on the 0.20 m circle around the workspace centre, at z = 0 and z = 0.12 m) together with the **target's projected hull**. Both must stay inside the frame with margin ≥ 0 px under the cameras' fitted k1 distortion, at the nominal and the revisit pose; the same ranges, deterministic redraws and logging as the shelf. Confirmed by the researcher, 2026-11-06.
- **True alignment, evaluation only.** For every query view the simulator writes the exact query-to-reference homography at the shelf front plane and at the target's depth (`true_alignment.json`, `sim/alignment.py`); on the tabletop at the table plane z = 0 and the horizontal plane through the target's centre. It scores the evidence step's image alignment and is never an input to evidence, calibration or a policy.
- **Matched occluder rule.** Occluded and removed share one occluder layout at both lightings. In the reference and visible phases the occluder is parked: present, but blocking nothing.
- **Objects.** YCB-Video models only (no Google Scanned Objects were needed), with the target from a fixed list that fits the space. The object list and its licence note are in the frozen generator configs. Confirmed by the researcher, 2026-11-06.
- **Geometric diagnostic labels** (from the instance masks):
  - a view of a present target is *geometrically diagnostic* when its visible fraction is ≥ 0.90;
  - a removed-episode view is diagnostic when the occluder covers < 10 % of the target's region at that view's camera pose. Queries use revisit poses, so the region is the visible-phase target mask at the same revisit pose, not the nominal reference mask (2026-11-03). On the check batch this differs for one view: SD1000 P4, 0.088 against 0.225 with the reference mask.
  - **These are geometric labels, not recognisability.** A target with some visible pixels below 0.90 is labelled non-diagnostic even if a detector might still recognise it. SD0001 is such a case: occluded visible fraction 0.022 at P1 and 0.005 at P5 (`data/raw/2026-11-03_sim_pilot/set_checks.txt`).

  Confirmed by the researcher, 2026-11-06.
- **Intended-hidden subset.** The IND28 family and DES3 are *designed* to stay hidden. Episodes in which no stop is geometrically diagnostic are reported as a **geometric no-diagnostic-view share**, per split, environment and lighting. That share is context for coverage. **It is not a theoretical maximum coverage**, and coverage is never reported against it as a ceiling. The code reports it as `geometric_no_diagnostic_view_share`, computed per episode and lighting from the view labels. It was renamed from "max achievable coverage" on 2026-11-03.

## 4. Policies

| Policy | Selection | Role |
|---|---|---|
| S | none (P3 only), k = 1 | passive reference, at its actual effort |
| F | first affordable of P1, P2, P4, P5 | naive baseline |
| R | uniform over affordable stops, 20 seeds, condition-free salt | naive baseline |
| **G** | D-norm parallax, lookahead **off** | geometric baseline (primary G) |
| G-lookahead | the same scorer plus the reachability filter | ablation |
| G-myopic | the pre-revision G (different normalisation) | legacy ablation |
| A | learned informativeness; same λ (= 0) and candidate set as G | proposed; built during dev; frozen model files in `frozen/` |
| A-nocue | A without the side descriptor and every other obstruction-derived feature | ablation |
| Dir | direction rule on P3's left/right descriptor (below); G when the descriptor is missing or 0 | explanatory control (post-hoc development origin) |
| FULL | all stops, uncapped | higher-effort reference, not an upper bound |
| `X-fcm` | same selector, k = 1 | H1 ablation |

- **A.** A regularised logistic regression on observed evidence and candidate geometry, never on unrequested images.
  - **Training:** grouped (by arrangement) CV on dev only, with geometric diagnostic labels (§3) as training targets.
  - **Freeze:** features and regularisation grid are fixed in FREEZE.md.
  - **Pinned design** (`notes/2026-11-05-ranker-a-design.md`, f4d4fe5, fixed before any fit on rendered evidence;
    implementation notes appended at 088e1dc):
    - Model: L2 logistic regression, fit by Newton's method, with C in {0.01, 0.1, 1, 10}. C is chosen by five
      inner arrangement folds on the training label's log-loss, never on a policy outcome.
    - Rows: every state after the start view and 0-2 further stops at which the stopping rule continues. Each
      arrangement carries weight 1.
    - Eight features: az_from_start, sep_visited, sep_occluded, obstruction_start, side_toward, side_missing,
      match_max and quality_min. A-nocue drops sep_occluded, obstruction_start, side_toward and side_missing.
    - Selection: G's candidate set and tie-break.
  - **Frozen models.** `frozen/model_A.json` and `frozen/model_A-nocue.json`, trained on all 120 development
    arrangements with the frozen thresholds (`python -m verify.ranker`).
  - **Isolation.** In the held-out check, each fold trains its own A and A-nocue in its own process on that fold's
    training arrangements.
- **The left/right descriptor** (`notes/2026-11-05-side-descriptor.md`, c41ecff; the M16 check's result appended at
  f4d4fe5):
  - Definition: d = u_left − u_right of v0.3's own obstruction map, over the target's context window split at the box
    centre. d > 0 points to larger azimuth.
  - It is missing on a refused registration or when either half has less than half its pixels valid.
  - Each requested view reveals only its own stop's value (`verify/side.py`).
- **Dir.**
  - If P3's descriptor is missing or 0, Dir is primary G at every step.
  - Otherwise, at each choice, it takes the affordable unvisited stop on the descriptor's side of the start stop that
    is nearest the current stop (ties: lower stop index). When there is none, it chooses as primary G.
  - Dir has no fitted parameter. It was defined after the headroom analysis of 2026-11-05, as a control for what A
    can gain from the descriptor alone.

  Confirmed by the researcher, 2026-11-06.
- **τ_occ** is both the clear-miss gate and G's occlusion flag. So the occlusion-gate-off ablation (τ_occ = 1.0) is **reported for F and R only**, where it changes the gate and nothing else.

## 5. Shared stopping rule

The rule is applied in order:
1. **PRESENT** if any view has match ≥ τ_match.
2. **REMOVED** if at least k = 2 distinct stops are clear misses (quality ≥ τ_q, obstruction ≤ τ_occ, match < τ_match).
3. **CONTINUE** if another stop is affordable.
4. Otherwise **UNKNOWN**.

Scores are never multiplied, and views are not assumed independent. k = 1 is used only by S and `-fcm`.

## 6. Budgets and travel

- **Primary:** uncapped, b = 3.
- **Secondary:** b = 0–4 curves, and the finite-cap curves over {1, 2, 3, 4, 6} arc steps, never tuned.
- **Travel:** reported alongside accuracy and coverage at every point.
- **Known geometry** (2026-11-02, every evidence pattern, both geometries), declared now so no setting is picked afterwards to make selection "matter":
  - With λ < 1, primary G and F make identical runs under the 1- to 4-step caps. Such contrasts are flagged `identical`, and their [0, 0] intervals are not evidence.
  - Uncapped, G and G-lookahead are always identical.

## 7. Threshold calibration (`verify/select.py`, select-v4; dev only)

- **Calibration.** τ_q, τ_occ and τ_match are calibrated together per view, on dev views of both lightings, Neyman-Pearson style. **One joint search** runs over the full predeclared grid τ_q × τ_occ × τ_match. A setting is feasible only if it meets **both** limits:
  - the false clear-miss rate on **geometrically diagnostic** views of present-target episodes is ≤ α_clear = 0.10;
  - the false-clear rate (views passing the τ_q / τ_occ gate) on **non-diagnostic** views of occluded-present episodes is ≤ α_clear.

  Among feasible settings, correct clear misses on diagnostic removed views are maximised. The tie-break order is complete:
  1. lower false clear-miss rate;
  2. lower false-clear rate;
  3. lower τ_match;
  4. higher τ_q;
  5. lower τ_occ.

  The choice therefore never depends on the order in which the grid is searched. The same joint search runs inside every CV training fold. **No policy is run before the thresholds are fixed.**
- **Infeasible α.** If no grid setting meets both limits, calibration **fails loudly**: there is no usable candidate. The report says which limit binds and gives the best achievable rates. Nothing is relaxed automatically. Any change (grid, α, or evidence v0.3) is a dated addendum made on dev before the freeze.
- **Procedure revision (2026-11-03, before any dev render from the revised generator and before the freeze).**
  - **What changed.** The first select-v4 implementation was sequential: it fixed the most permissive gate that met its limit, then searched τ_match only under that gate. The PR #4 review showed this can declare infeasibility while other grid settings meet both limits. In a hand-built example, 315 of 693 settings were feasible, but the sequential search failed (`runs/2026-11-03-pr4-review/select1/`; independently reproduced in `runs/2026-11-02-joint-calibration-review/`). That figure comes from constructed scores, not rendered Isaac data.
  - **What stays.** The joint search keeps the grid, both α limits, their label populations and the objective. The intent of this section is unchanged; the implemented method changed.
- **Feasibility is not lighting robustness.** In that example, every feasible setting rejects all L2 views at the quality gate. The calibration report therefore gives gate pass rates per lighting, with counts, at the chosen setting and in every fold. It covers diagnostic removed views, diagnostic present views and all views. It flags any lighting whose pass share is ≤ 0.05, and reports an empty population as "no views"; that cut-off is a reporting flag, not a procedure parameter. Held-out recall and coverage per lighting, from the held-out policy check, stay the evidence for robustness.
- **Fixed settings:** k = 2, λ = 0. The sensitivity runs are reported, never selected.
- **Dev check.** Every policy's FRR / recall / coverage is reported on dev per lighting. A policy above 10 % is reported as such, and the thresholds are not retuned.
- **Honest dev estimates** come from grouped (by arrangement) CV:
  - **Grouping.** All views, conditions and lighting variants of a held-out arrangement are left out together.
  - **Per fold.** Each fold's thresholds are calibrated without the held-out arrangement, using the same joint search. **Every policy is then run on it with that fold's thresholds** (the held-out policy check). The policies are S, F, R, G, G-lookahead, G-myopic, FULL and the `-fcm` variants, at the primary budget, uncapped and under each cap. Results are reported per lighting with counts.
  - **A.** When A is included, its training and model selection also exclude the held-out arrangement. Each fold builds A afresh from a factory and trains and runs it in its own process, on a training subset that holds a snapshot of the fold's training arrangements only. Nothing reachable from that subset leads to the held-out arrangement's evidence, truth, labels or input files, and no state carries from one fold to the next. A trainable that writes and rereads its own files is outside this guarantee and is checked by review.
  - **Thresholds.** One global threshold set is frozen.
- **Freeze eligibility.**
  - **Required.** A candidate is eligible only if calibration succeeds on all dev data **and** in every planned CV fold.
  - **Incomplete candidates** may be saved for investigation, but they are marked `eligible_for_freeze: false` with the reasons.
  - **Exit status:** `verify.select` exits 0 for eligible, 3 for infeasible on all dev data (no candidate) and 4 for ineligible.
  - **Freeze check:** the freeze step accepts a candidate only through `require_freeze_eligible`. It takes the candidate file only, checked against its `.sha256` sidecar, and refuses ineligible and unmarked candidates. It refuses a synthetic one if any mark says so: the flag, the label or the synthetic provenance.
  - **Not required:** every held-out arrangement meeting α. Those held-out rates are reported checks.
- **Flat threshold curves are acceptable.** Both the correct and the false clear-miss curves over the grid are inspected and reported. The declared tie-break is kept when it decides, and no variation is forced.
- **Result (2026-11-06; `notes/2026-11-06-pooled-calibration.md`).** The final shared calibration ran on both
  environments' 120 development arrangements: shelf 60, and tabletop 60 = the first 11 plus the 49 of Amendment 2. The
  held-out check included A, A-nocue and Dir.
  - **Eligible:** calibration succeeded on all development data and in all 120 folds.
  - **Thresholds:** tau_q 0.3, tau_occ 0.4, tau_match 0.7, k 2, lam 0. The candidate is `frozen/frozen_candidate.json`,
    sha256 3a45b23c...
  - **Subgroups** (reported, not adjusted): the gate's false-clear rate on shelf L1 views is 21 / 176. The pooled rate
    is 50 / 632, which meets alpha_clear.

Confirmed by the researcher, 2026-11-06.

## 8. Uncertainty

- **Primary:** the **paired arrangement-level cluster bootstrap**, resampling whole arrangements so their related views and episodes stay together.
  - It is **stratified by occluder family**, so each replicate keeps the design's quota counts. A stratum is a quota slot class: S8, M16-left, M16-right, IND28, white box or YCB box. Design cases (DES0–DES3) are strata of their own. If any arrangement lacks family information, or any stratum has fewer than 2 arrangements (a single arrangement would only redraw itself and add no variance), the bootstrap falls back to unstratified, and the output says so.
  - B = 10,000 on test; per lighting and per environment.
  - Percentile 95 % intervals for contrasts and for single rates (FRR, removal recall and coverage, per lighting), from the same draws. For R, arrangements are resampled but its seeds are not, so R's interval is for its expected rate over its own randomisation.
- **Secondary, descriptive only:** Clopper-Pearson and Jeffreys bounds on integer counts. They treat episodes as independent, which they are not. Integer counts do not remove the dependence between views and episodes of one arrangement.
- **FRR** is reported per episode and per arrangement (any false removal). For R, "any false removal" is computed per seed and then averaged over seeds, because R's matched episodes share a random stream.
- **Pointwise and unadjusted** (made explicit at the final review, 2026-11-06). Every interval is a pointwise 95 % interval for one metric of one policy or pair, in one environment, lighting and setting. None is adjusted for the number of comparisons, and no simultaneous coverage is claimed. A conclusion is stated only for the environment, lighting and metric it was computed for; there is no omnibus claim that A wins in some cell or in every cell.
- **Zero-width intervals** (added at the final review, 2026-11-06; verify.run real-v7). An interval whose bootstrap replicates all give one value is flagged in results.json and table.md (`zero_width`, a `degenerate_note`, and the discordant episode and arrangement counts of a contrast), and so is a pair with the same decisions on every episode but different effort (`same_decisions`, `effort_note`). The typical cases are zero observed errors (an FRR of 0 [0, 0]) and two policies with the same decisions: on development data A and Dir gave an A − Dir removal recall of [0, 0] although their travel differs, so the runs were not identical. The declared bootstrap and the raw estimates are kept unchanged. Such an interval is never presented as evidence of zero risk, of a perfect rate or of equivalence; its counts are reported with it.

Confirmed by the researcher, 2026-11-06.

## 9. Metrics and reporting

- **Metrics:** FRR, removal recall, coverage (UNKNOWN in the denominator), resolved accuracy, mean views, mean travel, confusion counts. Every rate comes with its counts and with n_arrangements / n_episodes.
- **Context, not a ceiling:** the geometric no-diagnostic-view share (§3).
- **Breakdowns,** separately for L1 and L2:
  - per environment, per occluder family and per condition. Every per-family result carries the family's target composition (targets are assigned per slot within the targets each family supports, §2), because a difference between families can reflect their targets as well as their occluders;
  - curves over b and over the cap grid;
  - risk-coverage curves sweeping τ_match, with the frozen point marked.
- **Paired contrasts on test** (per lighting and environment):
  - A − G (primary), A − F, A − R and A − A-nocue;
  - G − F, G − R and each policy vs S;
  - G − G-lookahead and G − G-myopic;
  - `X` vs `X-fcm`.

  Identical-run contrasts are flagged, and so are contrasts with the same decisions but different effort and every zero-width interval (§8).
- **Secondary, predeclared: A − G on common targets.** This is the A − G contrast restricted to the foam brick and tuna can episodes, the two targets every family supports, across all families. It uses consistent target weighting: within each family the two targets carry equal weight, so the comparison across families is not a difference in target mix. It is reported per lighting and environment, with the same stratified paired cluster bootstrap (§8). This addresses the target–family confound (2026-11-03); it is secondary and does not replace the primary A − G. Implemented in `analysis/common_target.py` and `verify/run.py` (abd99d6). Confirmed by the researcher, 2026-11-06.
- **View level, on test (Amendment 2).**
  - Reported by environment and lighting:
    - refusals;
    - both per-view error directions: present-labelled clear misses and removed-labelled strong matches;
    - the plane geometry of accepted views: on the shelf the target plane, with the front plane beside it; on the
      tabletop the horizontal plane through the target's centre, a proxy, with the table plane beside it;
    - target-alone silhouette IoU.
  - It uses `tools/registration_compare.py --report-only` with the frozen thresholds.
  - The first 11 tabletop arrangements, the remaining 49 and the test are kept apart.
  - A plane disagreement is not an error against a true whole-scene alignment.
- **Labels and the real-photo check.** Every result is labelled "simulation". Pilots 1–2 (real photos) are a separate sim-to-real check of score ranges and are never pooled.
- **As implemented (verify.run real-v6, 2026-11-06, with the zero-width flags of real-v7; confirmed by the researcher, 2026-11-06).**
  - **Families.** The breakdowns use two levels:
    - the occluder family (S8, M16, IND28, white box, YCB box), with M16's two sides pooled;
    - the quota slot class, which splits M16 into M16-left and M16-right.

    Each level gives every policy's metrics per lighting, with the family's target composition. Policies run at the
    primary budget, uncapped and under every cap; S and FULL run at their own effort. On test, A − G per family and
    lighting is computed at b = 3 uncapped with the paired cluster bootstrap (B as §8). Within an occluder family the
    bootstrap is stratified by slot class: for M16 that is by side; the other families are a single stratum. If one M16
    side has fewer than 2 arrangements, the M16 bootstrap is unstratified and flagged, as in §8. Each family and
    lighting has its own derived seed. A family with fewer than 2 arrangements gets no interval. These intervals are
    descriptive evidence for H3, not new endpoints. The occluder-family level, with M16's sides pooled, is secondary
    reporting; the slot-class rows (M16-left and M16-right) and the side-stratified resampling are kept beside it.
  - **Each multi-view policy against S:** A, A-nocue, Dir, F, R, G, G-lookahead and G-myopic minus S, at S's actual
    effort (one view).
  - **The τ_match risk-coverage sweep.** It is run at b = 3, uncapped, with the other thresholds frozen, for every
    multi-view policy, S and FULL. The values are the calibration grid below the frozen value (0.4-0.65) and a
    predeclared extension above it (0.75, 0.8, 0.85, 0.9), so the curves show both sides of the frozen point at 0.7.
    Each curve gives FRR, removal recall, coverage, resolved accuracy and the false-present rate per lighting. It is
    reported only, never selected. The extension is fixed at those four values and is secondary reporting.

Confirmed by the researcher, 2026-11-06.

## 10. Predeclared diagnostics (reported whatever they show)

1. Distinct view sets reachable and used per (b, cap).
2. Path and decision disagreement: A vs G, G vs G-lookahead.
3. Geometric diagnostic-view coverage: occluded episodes with at least one geometrically diagnostic non-start view, and removed episodes with at least k such stops. As implemented: occluded-present episodes, k = 2, per lighting, at both family levels.
4. Occlusion-cue false-clear rate against the geometric labels, with counts. As implemented, at the frozen thresholds and with verify.select's definitions, it comes with three more per-view counts:
   - the false clear-miss rate on diagnostic present views;
   - the clear-miss rate on diagnostic removed views;
   - the false strong-match rate on diagnostic removed views.

   Beside these sits the strong-match count over all views of removed episodes, since each of those decides PRESENT.
5. The occlusion-gate-off ablation for F and R (§4). As implemented: F-gateoff and R-gateoff (τ_occ = 1.0, so the gate keeps only the quality cut) at b = 3, uncapped and under every cap, with the paired contrasts F − F-gateoff and R − R-gateoff at b = 3 uncapped.

## 11. Exclusions, retries and failed renders

- **Mechanical failures.** A render that fails mechanically (missing file, empty frame, no target in the reference mask) is re-rendered from the same seed, deterministically, and logged.
- **Revisit redraws** (frame check, §3) are logged per stop; they change no family, side or target.
- **Placement failures** are retried within the same family and then replaced from that family's reserve list (§2). This happens **before rendering**, never after evidence exists.
- **No outcome-based exclusion.** Nothing is dropped for a wrong or UNKNOWN decision.
- **Counts reported:** retries, replacements and exclusions, per split, environment and family.

## 12. Freeze (`docs/FREEZE.md`, recorded with sha256 **before any test scene is generated**; one exception, a unit test's temporary arrangement, is recorded in §14)

**Frozen before test rendering:**
- **Code and render setup:**
  - the code commit and tag, and `uv.lock`;
  - the Isaac Sim version and install recipe;
  - the renderer settings: mode, settle frames, subframes, tonemap, and `tags_enabled = false` (the render-settings hash);
  - `sim/alignment.py` (evaluation only).
- **Generator:**
  - its code and config hash, the object list with licences, and the quota table;
  - the **slot lists and test seed block, including the reserve lists**;
  - the retry maximum and the intended subsets (IND28 family, design cases);
  - the per-slot target lists and the excluded (target, family) pairs with their tested-search counts.
  - Current generator config: `configs/sim/arrange_shelf_v3.json` (12 cm M16, v3 side rule, tag-free scenes). v1 and v2 and the check sets made with them are superseded records.
- **Scene geometry:** `configs/rig_geometry.json`, the tabletop geometry, and the positions file per environment with per-stop pitch.
- **Evidence and calibration:**
  - the evidence code (frozen v0.2.1, or a v0.3 developed on dev), with the detector, prompts and encoder;
  - `frozen_candidate.json` from select-v4.
- **Policies:** A's coefficients and features; the budgets, the cap grid, the start stop and F order, and the R seeds.
- **Analysis:** this document's commit hash, and the analysis rules in §8–§10.

**Produced after test rendering, by the frozen procedure:** the mask-derived geometric labels, the visible fractions, and the no-diagnostic-view flags. They are outputs of the frozen code, not inputs to the freeze.

**How (2026-11-06).**
- `docs/FREEZE.md` carries one machine-readable record (`tools/make_freeze.py`): the frozen commit and the sha256 of
  every file the test path runs or reads, including `frozen/` (the candidate and the A / A-nocue model files) and the
  test pipeline specs.
- It also gives the runbook and the validity requirements that are checked before any outcome is read.
- Test seeds open only against that committed file (`sim/freeze.py`):
  - generation through `sim/testsplit.py`, with `sim/arrange.py` byte-identical;
  - episodes through `sim/episodes.py --freeze`;
  - rendering through `sim/render_test.py`, with `sim/isaac_batch.py` byte-identical, in a GB10 run directory of its
    own.

## 13. Deviations

- **Dated addenda.** Any change after the freeze is a dated addendum: what changed, why, and whether any test result had been seen. Earlier text is never edited.
- **Where addenda go.** They go in a new file, `docs/FREEZE_ADDENDA.md`, never in a frozen file. This document, the
  pinned notes and every file listed in FREEZE.md's record are checked byte for byte whenever a test seed opens, so
  editing any of them would close the gate.
- **Changes after test rendering** are reported as deviations next to the results they affect, with the pre-registered analysis alongside where feasible.

## 14. Development record before the freeze (CONDITIONAL DEVELOPMENT DIAGNOSTICS)

These findings come from development data. Several come from analyses chosen after looking at that data. They are
recorded before any test scene was rendered or scored, so that the test cannot be read in their light afterwards. (One
shelf test arrangement was generated temporarily by a unit test and deleted unseen; see the last item.) They predict
nothing about the test: new arrangements can move every number below, and A may tie with or lose to G.

- **Headroom** (2026-11-05, as Amendment 2 lists it):
  - On shelf dev at b = 3 uncapped, G removes 45 / 60 (L1) and 43 / 60 (L2) against any-order ceilings of 46 and
    44, and the present episodes are at their ceiling.
  - G's choices equal the fixed route P3, P1, P5, P4, P2 on all 360 shelf episodes.
  - Under the two-step cap, the any-order ceiling (25 and 24) exceeds the best one-side route (15 and 13). 18 of those
    ceiling removals are M16 arrangements, each resolvable only on the side away from its designed board offset
    (`runs/2026-11-05-headroom-check/side_audit.txt`, untracked).
  - The ceilings use views no policy may see.
- **The 49 tabletop arrangements** at the fixed shelf thresholds: `notes/2026-11-06-tabletop-dev49-transfer.md`.
- **The pooled calibration and its held-out check:** `notes/2026-11-06-pooled-calibration.md`. Held out, with no false
  removal anywhere:
  - At b = 3 uncapped, A and G differ on 5 of 240 removed episodes: A gains 3 and loses 2.
  - **Decision agreement, at the stated settings.** A's held-out decisions equal Dir's in every episode at b = 3
    uncapped and under the 3- and 4-step caps, and A-nocue's equal G's uncapped. This is an observed agreement, not
    a causal attribution: A-nocue drops four features together (sep_occluded, obstruction_start, side_toward and
    side_missing), so the side descriptor's own contribution is not isolated. Any mechanism is exploratory.
  - **Same decisions, different effort.** Uncapped, mean travel per episode (shelf L1 / L2, tabletop L1 / L2) is A
    156 / 158 / 71 / 74 cm, Dir 94 / 96 / 36 / 37 cm and G 145 / 146 / 68 / 69 cm, and mean views are A 2.14 / 2.16 /
    2.12 / 2.17, Dir 2.24 / 2.26 / 2.15 / 2.17 and G as A. Under the 3- and 4-step caps both A and Dir take 2.03 views,
    and A travels 85 / 85 / 40 / 40 cm against Dir's 51 / 51 / 25 / 25 cm
    (`runs/2026-11-06-pooled-dev120/summary/heldout_b3.txt`, untracked).
  - Under the 3- and 4-step caps, A (= Dir) removes 28 / 26 / 44 / 41 (shelf L1 / L2, tabletop L1 / L2) against G's
    15 / 15 / 28 / 29. At each of these caps A and G disagree on 58 removed episodes over the four cells: A gains 55
    (35 M16, all M16-left, and 20 outside M16: S8 12, YCB box 8) and loses 3 (YCB box)
    (`runs/2026-11-06-pooled-dev120/summary/discord_A_G_cap3_cap4_recount.txt`, untracked).
  - Under the 2-step cap A removes 0, as G does.
- **Resolution at the test size.** If test arrangements disagree like development ones, the SD of A − G removal recall
  at 100 test arrangements per environment is 0.012-0.017. So an inconclusive primary is likely. This is conditional
  on the sparse development disagreements (5 discordant episodes in 4 arrangements). Tabletop L1 had no discordant
  episode, so its SD of 0 reflects that sample, not perfect precision. The 2.8 SD figure is a normal-approximation
  heuristic, not a verified power for rare, discrete paired differences.
- **Limitations kept visible, not retuned.**
  - The shelf L1 subgroup's gate false-clear rate is 21 / 176 (0.119), above alpha_clear 0.10. The declared
    constraint is on the pooled views (50 / 632), §7.
  - In L2, the tabletop design cases TDES0-TDES3 fail the gate on their visible-phase views: 0 / 20 pass at the
    fixed shelf thresholds and 1 / 20 at the frozen ones, against 20 / 20 in L1. Their removed-phase diagnostic views
    still give clear misses, 10 / 10 in both lightings and at both threshold sets
    (`notes/2026-11-06-tabletop-dev49-transfer.md`; `runs/2026-11-05-tabletop-dev49/transfer/regcompare_design/` and
    `regcompare_design_frozen/`, untracked).
- **The renderer check** in FREEZE.md: the current renderer reproduces the shelf development renders within
  run-to-run render noise.
- **The test-seed incident** (`notes/2026-11-06-test-seed-incident.md`). On 2026-11-05 at 18:21 PST a unit test,
  under a mocked freeze, generated the shelf test arrangement of slot 0 (primary seed 100000) in a temporary
  directory. It read four fields (split, seed range, id format, freeze record), printed nothing, and the directory was
  deleted. Nothing was rendered, scored or viewed. Three runs of the rewritten test wrote the shelf test slot list,
  unread, to temporary directories. The gate was repaired the same evening (12b56f5, 28c7e96). The seed schedule is
  kept (the researcher, 2026-11-06).

None of this changes the design. The primary endpoint, k, λ, the caps, A's pinned design and the analysis rules stand
as written above.
