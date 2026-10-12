# Travel cap, λ and thresholds: when to set them, how, and what prior work says

*CMPE249 Occlusion-Aware Active Verification. Report date: 2026-10-02.*

## 1. Short answers

1. **How to set them.**
   - **The travel cap is an operating condition, not a tuned parameter.** Predeclare a small set of caps in units of arc steps (23.56 cm each) and report every metric as a curve over those caps and over b = 0–4.
   - **τ_q and τ_occ come from views with diagnostic labels.** Annotate each dev view: if the target were present, would it be visible here?
   - **τ_match, k and λ are chosen on multi-view dev episodes** using grouped-by-arrangement CV. Use a Neyman-Pearson-style rule: keep the configurations whose false removal rate (FRR) satisfies the limit, then pick the one with the best coverage and recall.
2. **Should they be set after the rig is ready?**
   - **After the rig:** every value that depends on scores, i.e. τ_match, τ_q, τ_occ and λ. Detector and embedding score distributions depend on the camera, mount, lighting and crop. Prior work re-tunes these per sensor or dataset [10, 11, 13], and hand-set constants carried from simulation to hardware hold only partially [4, 7].
   - **Now, in writing, before the rig:**
     - the procedure, the objective, the grid and the search order;
     - the cap grid;
     - the k range;
     - the CV protocol.
3. **Is there prior research?** Yes, though no paper matches the whole setup.
   - **PInVerify** [1] is the closest task: budgeted active verification over pre-captured views. It has no UNKNOWN answer, and its smart selectors did no better than random.
   - **PBD-AG** [2] is the closest removal rule: removal only after ≥3 visibility-admissible misses. Turning visibility gating off raised false absence from 0.014 to 0.262.
   - **INSPECT** [3] is the closest structure: 5 poses, a normalised motion cost and a defer option.
   - **Atanasov et al.** [7] is the classic precedent for jointly choosing movement cost and stopping time.
   - **The gap:** no paper I found combines an explicit UNKNOWN, view and travel budgets, removal only after clear misses from k distinct stops, and a risk-coverage evaluation on uncalibrated open-vocabulary scores.

## 2. What to set when

Order of work:
- **(A)** Fix the protocol now.
- **(B)** Fix geometry once the rig is measured.
- **(C)** Single-view "catalog sweep" with the camera and objects only.
- **(D)** Multi-view dev episodes under grouped CV.
- **(E)** Freeze everything before any test capture.

| Knob | Depends on rig? | Depends on perception? | Set from | Recommended procedure | Fix now? |
|---|---|---|---|---|---|
| Camera spacing / 5 stops | Yes (physical) | Indirectly (must create diagnostic views) | Pilot (G4 criteria) | Measure the arc, check G4 (i)–(iii). Allow one redesign if it fails. Must be fixed **before dev capture**, not at the freeze. | Spacing plan yes; final value after the pilot |
| Travel cap(s) | Geometry only (Rig A1: travel is modelled) | No | Geometry (B) | Predeclare caps in steps, e.g. {1, 2, 3, 4, 6, ∞} = 23.6 / 47.1 / 70.7 / 94.2 / 141.4 cm / uncapped. Set each value slightly above the multiple (e.g. +1 cm) to avoid floating-point boundary effects. Report all of them. Optionally mark one primary cap chosen on dev by a predeclared knee rule. | **Yes** |
| λ and normalisation | Weakly | Yes (scale of G's and A's utility) | Dev episodes (D) | Normalise utility and travel per decision over the feasible candidates, Delmerico-style [16], so the weight is dimensionless and A and G can share it. Sweep a small grid that includes 0. Pick by a predeclared rule. | Grid and rule yes; value no |
| τ_q | Yes (exposure, blur, resolution) | Yes | Catalog sweep + diagnostic labels (C) | NP-style: cap the false-clear rate on occluded-present views, then maximise clear rate on removed views. | Procedure only |
| τ_occ | Yes | Yes (obstruction cue) | Catalog sweep + labels (C), checked in (D) | As for τ_q. Note it is **also G's occlusion flag**, so re-check G's behaviour after choosing it. | Procedure only |
| τ_match | Yes (lighting, viewpoint) | Yes (DINOv2/CLIP, uncalibrated) | Per-view histograms (C), final in (D) | Final choice in (D), under the FRR constraint (Sec. 3). | Procedure only |
| k | No | Via per-view miss correlation | Dev episodes (D) | k ∈ {2, 3}, bounded by b+1 views. Choose under the FRR constraint. Requiring k ≥ 2 is fixed by design. | Range yes |
| Budget b | No | No | Fixed by design | b = 3 primary; b = 1 and 2 secondary (D14); report curves for 0–4. | **Yes** (already decided) |

On per-lighting thresholds: keep one global threshold set, because there is too little data to split them. Report all results broken down by lighting, and treat a large per-lighting gap as a finding.

## 3. Threshold procedure with about 10 dev arrangements

**Step 0 (now, before Oct 11): fix the D10 objective in a pushed `docs/PREREGISTRATION.md`.**
- PLAN requires this document before pilot capture, and it does not exist yet.
- Recommended objective: among configurations with grouped-CV FRR ≤ 10%, require recall ≥ 80% where possible. Then maximise coverage, and break ties by recall and then by mean travel.
- The current D10 draft treats recall only as a tie-break, so a frozen configuration could satisfy D10 and still miss the joint recall target.

**Step 1 (catalog sweep, single view): look at per-view score distributions before running any grid.**
- Plot match score for target-visible, target-occluded (at several levels) and target-absent-with-distractors views.
- Plot quality and obstruction for occluded-present views and for removed views.
- Atanasov et al. note that detection scores "affected by noise and occlusions … are not used directly". Instead they learn per-viewpoint, occlusion-conditioned score histograms offline [7]. This is the same kind of data.
- Detector calibration varies with box position and scale [24]. Break the plots down by stop (edge stops at ±60° versus P3) and by lighting.

**Step 2: choose τ_q and τ_occ from diagnostic labels.**
- Label each dev view "diagnostic" (the target would be visible if present) or not.
- Choose (τ_q, τ_occ) so that the false-clear rate on occluded-present views stays below a predeclared level. The project's R3 trigger is 20%; a stricter target is better. Subject to that, maximise the clear rate on removed views. This is a Neyman-Pearson trade [19].
- Prior systems gate on visibility in the same way: POCD's θ_vis [9], DGSG-Mind's ">50% visible" [13], Khronos's ray test [8].
- DGSG-Mind reports that a conservative gate "defers decisions under partial visibility and reduces false removals" [13]. Expect UNKNOWN to rise as the gate tightens.

**Step 3: choose τ_match and k on multi-view dev episodes.**
- Use a small grid, about 5–7 τ_match values × k ∈ {2, 3}, with τ_q and τ_occ held at their Step 2 values.
- Order the grid from most to least conservative and walk it in that fixed order. This is Learn-then-Test's fixed-sequence idea [17]. Keep the last configuration that still meets the FRR limit, then apply the D10 objective.
- Fixed-sequence testing needs a one-dimensional ordering decided in advance. LTT also assumes i.i.d. calibration data [17], so the **arrangement** should be the unit, not the view.
- Raising τ_match makes PRESENT harder and can *add* false removals. Parametrise the grid so it really is monotone. With an adaptive G, check monotonicity empirically rather than assuming it [18].
- Rough sanity check for k: if per-view clear-miss probability on present targets is p and views were independent, false removal would scale roughly like p^k. Views of one arrangement are correlated, so measure this rather than assume it. The SPRT analogy [23] holds only loosely: in the adapted SPRT the early decision is the one with the bound, which maps onto PRESENT, not REMOVED.

**Step 4: get an honest dev estimate with nested grouped CV.**
- Outer loop: leave-one- or leave-two-arrangements-out.
- Inner loop: the Step 3 rule, run only on the remaining arrangements.
- The CV error of the tuned setting, reported without nesting, is optimistic. On null data, 18.5–38% of tuned-model CV estimates fell below 30% error, while nested CV was "almost unbiased" [21].
- Freeze the configuration the same rule picks on all 10 dev arrangements.

**What guarantee is possible.** Dev has 40 present episodes (10 arrangements × visible/occluded × 2 lighting) and 20 removed episodes. All figures below are one-sided 95% Clopper-Pearson upper bounds that I computed:

| False removals | Upper bound on FRR |
|---|---|
| 0 / 40 | 7.2% |
| 1 / 40 | 11.3% |
| 2 / 40 | 14.9% |

- **A single dev false removal already rules out certifying FRR ≤ 10%.**
- The Neyman-Pearson order-statistic method needs n ≥ log δ / log(1−α) = 22 *independent* present units at α = δ = 0.1 [19].
- Under hierarchical (arrangement-clustered) data, the effective sample size is closer to the 10 arrangements than to 40 episodes. This is my inference from Lee et al. [20], whose results are about prediction sets. At the arrangement level, 0/10 gives a 25.9% bound.
- Conformal risk control gives only an expected-value guarantee, and with n = 10 arrangements it needs empirical FRR ≤ 1% [18].
- **Conclusion: no formal (α, δ) guarantee is achievable on dev.** Treat dev as a selector and state the targets as goals.

On test (80 present, 40 removed episodes):

| False removals | Upper bound on FRR |
|---|---|
| 0 / 80 | 3.7% |
| 2 / 80 | 7.7% |
| 3 / 80 | 9.4% |
| 4 / 80 | 11.1% |

So "FRR ≤ 10%" is credible on test only with at most 3 false removals.

For recall on dev, 16/20 = 80% has a one-sided 95% lower bound of 59.9%. Report recall with intervals, prefer Wilson or Jeffreys intervals for small n [22], and add a cluster bootstrap over arrangements.

**Reporting.**
- Show risk-coverage curves [15] for each policy and budget by sweeping τ_match, and mark the frozen operating point.
- Never tune on test.
- Every frozen value goes into FREEZE.md together with the rule that produced it. DovSG, for example, gives no values for its removal thresholds [12].

## 4. Travel cap and λ

**Why 50 cm degenerates.**
- 50 cm is 2 steps plus 2.9 cm of slack.
- After a far first move (P3 to P1 = 47.1 cm), no second move fits, so F and G follow the same path.
- Caps should be defined in step units so that each one changes which view sets can be reached:
  - 1 step: one neighbour.
  - 2 steps: one extreme, or two stops on the same side.
  - 3 steps: both neighbours (P2 then P4 = 70.7 cm).
  - 6 steps: both extremes (P1 then P5 = 141.4 cm).

**Report curves, don't tune the cap.**
- Information-gathering work treats the budget as a hard, application-given constraint and reports results across budget values: "100 random scenarios … for each budget" [14].
- Active recognition and view planning likewise report performance against views or path length [6, 26, 27].
- If one primary cap is wanted, pick it on dev at the knee of the coverage-versus-travel curve at FRR ≤ 10%. This adapts a device that Patten et al. [25] and Pan et al. [28] used on evaluation curves; it is the project's adaptation, not their protocol.
- Pan et al. found that extra views beyond k = 1 bought little coverage for much more movement, but k = 4–5 did improve coverage [28]. The knee is a trade-off, not a plateau.

**Fixing "G ignores the remaining cap".** `candidates()` already masks unaffordable stops, so masking alone won't help. G needs to look ahead. Options, cheapest first:
1. **Reachability-aware feasibility.** Allow a candidate only if, after moving there, at least k−1 further distinct stops (enough to complete REMOVED) are still reachable within the remaining cap. Otherwise REMOVED becomes impossible and the episode is doomed to UNKNOWN. RIG prunes nodes whose cost exceeds the budget, and notes that goal-reachability against the *remaining* budget can be used the same way [14]. Applying this to k stops is the project's adaptation.
2. **Short receding-horizon enumeration.** With 5 stops and b ≤ 4, enumerate every affordable sequence, score each by the sum of G's utility, and execute the first move. This makes G non-myopic, as in Atanasov's nonmyopic planning [7], at negligible cost.
3. **Normalise travel by the remaining cap**, or by the sum over feasible candidates [16], instead of the fixed 94.25 cm span. This is the project's extension; Delmerico normalises per iteration over candidates.
4. **INSPECT-style move gate:** move only if net gain minus the weighted motion cost exceeds a margin. INSPECT uses λ_d = 0.05 and τ_V = 0.02, both hand-set [3].

Also report "distinct view sets per budget and cap" and the A/G disagreement rate as predeclared diagnostics of whether selection can matter at all. PInVerify found random, angular FPS and LLM-NBV "within 0.2 pp of each other" [1], so G ≈ R ≈ F is a plausible real outcome and should be reported honestly.

**λ.**
- Nobody derives the travel weight from first principles:
  - Gonzalez-Banos and Latombe set it to stop the robot oscillating between regions of similar gain [29].
  - Selin et al. note that tuning λ "has to be done for every environment and it typically requires several attempts" [30].
  - Naazare et al. dropped the travel penalty because motion costs were "hard to determine" for a robot mixing arm motion and driving. In their comparison, the linear penalty gave lower coverage (74.4% versus 82.0%) [31].
- Recommendation: use the dimensionless normalisation, sweep a grid that includes λ = 0 on dev only, and pick by a predeclared rule. For example: the lowest mean travel among configurations meeting the D10 targets at b = 3.
- Use the hard cap as the primary travel control and λ as a secondary tie-break.

## 5. Related research

★ marks papers new to the project, i.e. not in `references.bib`, the survey, or NOVELTY_CHECK.

**Closest active verification**
- **PInVerify** [1]: offline sector graph with T = 6 steps and YES/NO answers. Trap views "consume a step", and low-visibility confidence is down-weighted (c̃ = 0.2c when c < 0.3). **Differs:** no UNKNOWN (ties go to a conservative NO), no travel cost, and it verifies identity rather than removal. It is the model for capturing all 5 stops so every policy can be replayed offline.
- **INSPECT** [3]: 5 poses, motion cost λ_d·d with normalised angles, a defer option, and "insufficient" claims. The grouped inner-fold selection applies to its isolated visual-verifier study, and its travel constants are hand-set. **Differs:** it verifies assembly claims rather than presence or removal.
- **Atanasov et al.** [7] (T-RO 2014; arXiv title "…Active Object Detection"): minimises E{J_M(τ) + λJ_D} over viewpoints *and* stopping time. Baselines stop heuristically at p ≥ 0.6. On the PR2 it reached 76% TP and 98% TN with about 2.5 measurements. The framework allows asymmetric costs, but the experiments use 75 for every wrong decision, and future views are planned assuming no occlusion. **Differs:** no abstain answer, and it classifies rather than declaring absence.
- **Li et al. 2021** (arXiv 2107.12095, IROS 2021) is already in NOVELTY_CHECK and was not re-verified in this pass.

**Removal versus occlusion in scene memory**
- **PBD-AG** [2]: depth-based visibility test, removal at p_exist ≤ 0.12 "after at least three visibility-admissible misses". Constants are hand-set and it was evaluated on 12 events. **Differs:** a mapping setting with no UNKNOWN and no statistical validation.
- **Khronos** [8] names the problem "evidence-of-absence vs. absence-of-evidence". It needs 60% of rays within 5 s to count absence. It is passive, its constants are unjustified, and its quantitative evaluation is in simulation.
- **POCD** [9]: a frustum-only "unobserved" test with a forced removal pseudo-change and no occlusion check, i.e. the failure our gate targets. Its parameter values are in a supplement I did not open (UNVERIFIED).
- **★ Build Once, Monitor Continuously** [10]: removes on an FoV miss with GroundingDINO. Its authors concede it "cannot tell a missed detection apart from a real removal". Their future work is "interactive perception, where the robot repositions itself to confirm uncertain detections", which is this project. Occlusion is one of two causes they name.
- **★ Panoptic Multi-TSDFs** [11]: three states {persistent, unobserved, absent}, analogous to PRESENT/UNKNOWN/REMOVED. For real-world data they doubled the SDF tolerance "to account for the increased sensor noise", which is direct evidence that tolerances get re-set for the real sensor.
- **★ DGSG-Mind** [13]: a ">50% visible" gate before a fused change score with δ = 0.35. Its per-dataset 0.4/0.3 values are the *association* threshold, not the removal threshold.
- **★ Fehr et al.** [32]: free versus unobserved space in TSDFs. Most parameters are shared across datasets, but the change threshold θ differs.
- **★ DovSG** [12]: single-view depth/colour deletion, with no reported threshold values.
- **★ Langer** (thesis Ch. 6; published as Frontiers Robot. AI 2022) [33]: says the match threshold "is selected depending on the uncertainty in robot localization, the reconstruction, object detection …".
- **DynaMem** (already cited): its "not found" output means moved/absent, not a calibrated UNKNOWN.

**View planning with movement cost or budgets** (all ★ except Atanasov)
- **Hollinger and Sukhatme** [14]: hard budget, prune against the remaining budget, results over budgets.
- **Delmerico et al.** [16] and **Isler et al.** [34]: normalised gain minus normalised cost. In Isler there is no cost weight; in Delmerico it is γ ∈ [0, 1].
- **Gonzalez-Banos and Latombe** [29], **Selin et al.** [30], **Naazare et al.** [31]: λ is behavioural and depends on the environment.
- **Patten et al.** [25]: occlusion-aware active classification with clockwise and random baselines (like F and R). Travel cost is left as future work.
- **Zaenker et al.** [27]: U = IG − α·C for occluded fruit, evaluated against plan length.
- **Pan et al.** [28]: sweep over the number of NBVs with paired tests.
- **Yang et al.** [26] and **Jayaraman and Grauman** [6]: fixed step budgets with no learned stop or abstention. Yang's performance-versus-steps curve (on hard, occluded samples) saturates.
- **GenMOS** [5]: labels occluded regions Unknown rather than absent. It focuses on finding objects, not on declaring absence.
- **Ren et al.** [4]: conformal stopping for embodied QA. Falls back to argmax at the horizon. Its hardware thresholds (ε = 0.5) were carried over from simulation.

**Thresholds under risk constraints** (all ★): selective classification and SGR [15] (bounds selective risk, i.e. resolved error, not FRR); Learn-then-Test [17]; conformal risk control [18]; NP classification [19]; hierarchical exchangeability [20]; nested CV [21]; binomial intervals [22]; SPRT in vision [23]; calibration [35, 24]. The project currently cites nothing on risk control or threshold selection.

**Abstract-only, UNVERIFIED for any claim beyond its abstract:**
- OASIS-Map [36], whose abstract names occlusion as a key association challenge.
- Bircher et al., receding-horizon NBV (ICRA 2016), whose formula and tuning details I could not open. Not cited for any claim.

## 6. Next steps

**Now, before the rig:**
1. Write and push `docs/PREREGISTRATION.md`, with these contents:
   - the D10 objective, including a recall constraint;
   - the cap grid in steps;
   - k ∈ {2, 3};
   - the λ grid including 0;
   - the τ grid order;
   - nested grouped CV;
   - the interval methods (one-sided Clopper-Pearson or Jeffreys, plus a cluster bootstrap);
   - the diagnostic-label definition for τ_q and τ_occ;
   - the "selection matters" diagnostics.
2. Implement cap-aware G (reachability feasibility plus receding-horizon enumeration) and per-decision normalisation. Unit-test both on synthetic data (logic only) and confirm F ≠ G at the 2- and 3-step caps.
3. Build `verify/select.py` with the fixed-order grid walk, nested CV and bound reporting, and test it on synthetic data. Do **not** carry synthetic values forward.
4. Decide whether to split the double-duty thresholds: separate "strong match" and "no match" bands, and a separate G occlusion flag.
5. Update `references.bib` and the stale survey with [1–3, 7, 10, 11, 14–19].
6. Read the PBD-AG supplement.

**After the rig and perception stack are frozen:**
1. Run the pilot and check G4. Fix the spacing before dev capture.
2. Run the catalog sweep: per-stop, per-lighting score histograms. Set τ_q and τ_occ from labels.
3. Capture dev with all 5 stops per episode and replay all policies offline.
4. Run the Step 3–4 selection, sign FREEZE.md (target Oct 18), then capture test.
5. Report curves over b and caps, risk-coverage plots, per-condition and per-lighting breakdowns, and an ablation with the occlusion gate off (τ_occ = 1.0) as a PBD-AG-style replication [2].

## References

1. Y. Jiang. PInVerify: An Offline Embodied Benchmark for Active Instance Verification. arXiv 2605.30639 (2026), CVPR 2026 FMEA workshop poster. https://arxiv.org/abs/2605.30639
2. S. Bao et al. PBD-AG: Persistent Baseline-Delta Active Graphs with Uncertainty-Aware Inspection for Long-Horizon Service Robots. arXiv 2608.10449 (2026). https://arxiv.org/abs/2608.10449
3. D. Wen et al. INSPECT: Learning Robot View Selection from Assistant Use. arXiv 2609.20615 (2026). https://arxiv.org/abs/2609.20615
4. A. Z. Ren et al. Explore until Confident: Efficient Exploration for Embodied Question Answering. RSS 2024; arXiv 2403.15941. https://arxiv.org/abs/2403.15941
5. K. Zheng, A. Paul, S. Tellex. A System for Generalized 3D Multi-Object Search. ICRA 2023; arXiv 2303.03178. https://arxiv.org/abs/2303.03178
6. D. Jayaraman, K. Grauman. Look-Ahead Before You Leap: End-to-End Active Recognition by Forecasting the Effect of Motion. ECCV 2016; arXiv 1605.00164. https://arxiv.org/abs/1605.00164
7. N. Atanasov et al. Nonmyopic View Planning for Active Object Classification and Pose Estimation. IEEE T-RO 30(5), 2014, DOI 10.1109/TRO.2014.2320795; arXiv 1309.5401. https://arxiv.org/abs/1309.5401
8. L. Schmid et al. Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments. RSS 2024; arXiv 2402.13817. https://arxiv.org/abs/2402.13817
9. J. Qian et al. POCD: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes. RSS 2022; arXiv 2205.01202. https://arxiv.org/abs/2205.01202
10. S. H. Allu et al. Build Once, Monitor Continuously: Persistent Semantic Mapping via Autonomous Exploration and Open-Vocabulary Object Updates. arXiv 2409.15493 (v4, 2026). https://arxiv.org/abs/2409.15493
11. L. Schmid et al. Panoptic Multi-TSDFs. ICRA 2022; arXiv 2109.10165. https://arxiv.org/abs/2109.10165
12. Z. Yan et al. Dynamic Open-Vocabulary 3D Scene Graphs for Long-term Language-Guided Mobile Manipulation (DovSG). IEEE RA-L 2025; arXiv 2410.11989. https://arxiv.org/abs/2410.11989
13. L. Ge et al. DGSG-Mind: Dynamic 3D Gaussian Scene Graphs for Long-Term Scene Understanding and Grounding. arXiv 2605.29879 (2026). https://arxiv.org/abs/2605.29879
14. G. A. Hollinger, G. S. Sukhatme. Sampling-based Motion Planning for Robotic Information Gathering. RSS 2013. https://www.roboticsproceedings.org/rss09/p51.pdf
15. Y. Geifman, R. El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017; arXiv 1705.08500. https://arxiv.org/abs/1705.08500
16. J. Delmerico et al. A Comparison of Volumetric Information Gain Metrics for Active 3D Object Reconstruction. Autonomous Robots 42, 2018. https://rpg.ifi.uzh.ch/docs/AURO17_Delmerico.pdf
17. A. N. Angelopoulos et al. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. Annals of Applied Statistics 19(2), 2025; arXiv 2110.01052. https://arxiv.org/abs/2110.01052
18. A. N. Angelopoulos et al. Conformal Risk Control. ICLR 2024; arXiv 2208.02814. https://arxiv.org/abs/2208.02814
19. X. Tong, Y. Feng, J. J. Li. Neyman-Pearson Classification Algorithms and NP Receiver Operating Characteristics. Science Advances 4(2), 2018; arXiv 1608.03109. https://arxiv.org/abs/1608.03109
20. Y. Lee, R. F. Barber, R. Willett. Distribution-free inference with hierarchical data. arXiv 2306.06342. https://arxiv.org/abs/2306.06342
21. S. Varma, R. Simon. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7:91, 2006. https://doi.org/10.1186/1471-2105-7-91
22. L. D. Brown, T. T. Cai, A. DasGupta. Interval Estimation for a Binomial Proportion. Statistical Science 16(2), 2001. https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full
23. J. Matas, O. Chum. Randomized RANSAC with Sequential Probability Ratio Test. ICCV 2005. https://cmp.felk.cvut.cz/~matas/papers/chum-waldsac-iccv05.pdf
24. F. Küppers et al. Multivariate Confidence Calibration for Object Detection. CVPR 2020 Workshops (SAIAD); arXiv 2004.13546. https://arxiv.org/abs/2004.13546
25. T. Patten et al. Viewpoint Evaluation for Online 3-D Active Object Classification. IEEE RA-L 1(1), 2016. https://opus.lib.uts.edu.au/handle/10453/89769
26. J. Yang et al. Embodied Amodal Recognition: Learning to Move to Perceive Objects. ICCV 2019; arXiv 1904.04404. https://arxiv.org/abs/1904.04404
27. T. Zaenker et al. Combining Local and Global Viewpoint Planning for Fruit Coverage. ECMR 2021; arXiv 2108.08114. https://arxiv.org/abs/2108.08114
28. S. Pan et al. Integrating One-Shot View Planning with a Single Next-Best View via Long-Tail Multiview Sampling. IEEE T-RO 2024; arXiv 2304.00910. https://arxiv.org/abs/2304.00910
29. H. H. González-Baños, J.-C. Latombe. Navigation Strategies for Exploring Indoor Environments. IJRR 21(10–11), 2002. https://ai.stanford.edu/~latombe/cs326/2002/nbv.pdf
30. M. Selin et al. Efficient Autonomous Exploration Planning of Large-Scale 3-D Environments. IEEE RA-L 4(2), 2019. https://www.ida.liu.se/divisions/aiics/publications/IRAL-2019-Efficient-Autonomous-Exploration.pdf
31. M. Naazare, F. Garcia Rosas, D. Schulz. Online Next-Best-View Planner for 3D-Exploration and Inspection With a Mobile Manipulator Robot. IEEE RA-L 7(2), 2022; arXiv 2203.10113. https://arxiv.org/abs/2203.10113
32. M. Fehr et al. TSDF-based change detection for consistent long-term dense reconstruction and dynamic object discovery. ICRA 2017. https://tisl.utm.utoronto.ca/publication/201705-icra-tsdf_change_detection/icra17-tsdf_change_detection.pdf
33. E. Langer. Object Change Detection for Autonomous Indoor Robots in Open-World Settings. PhD thesis, TU Wien, 2023 (Ch. 6; published as Langer, Patten, Vincze, Frontiers in Robotics and AI 2022, DOI 10.3389/frobt.2022.828732). https://repositum.tuwien.at/handle/20.500.12708/176591
34. S. Isler et al. An Information Gain Formulation for Active Volumetric 3D Reconstruction. ICRA 2016. https://rpg.ifi.uzh.ch/docs/ICRA16_Isler.pdf
35. C. Guo et al. On Calibration of Modern Neural Networks. ICML 2017; arXiv 1706.04599. https://arxiv.org/abs/1706.04599
36. H. Oh et al. OASIS-Map: Object-Level Change Detection in Multi-Session Mapping using Semantic Correspondence Matching. arXiv 2607.14899 (2026). Abstract only, UNVERIFIED beyond the abstract. https://arxiv.org/abs/2607.14899