# Novelty Check for the Week-2 Gate

**Project:** Occlusion Aware Active Verification for Robot Scene Memory
**Check date:** September 26, 2026 (Week 3; the week-2 gate was due around September 21)
**Gate question** (from `docs/PROPOSAL.md`, "Milestones and feasibility gate"): "If the closest work already contains this comparison ... I revise the scope or the topic with the instructor."
**Prepared with:** Claude Code (AI assistance), for the student to review. This is a literature check. It contains no experimental results.

## Verdict

1. **The planned comparison is not already in the literature I could find.** No paper I found compares a learned view selector with a geometric one, plus fixed and random orders, under one shared stopping rule and matched view and travel budgets, for deciding whether a *remembered* object instance is present, removed, or undecidable after an unobserved change, with occluded-but-present controls. This holds for the three works the proposal names as the closest active methods ([7] DynaMem, [10] Bogenberger et al., [11] Terashima et al.; `docs/PROPOSAL.md` and the survey's "Closest work" section). It also holds for [12] SceneDiff, which that section discusses as passive work. *(Evidence check: the first version said "the four works the proposal names as closest", but the proposal names only [7, 10, 11].)* Each has at most one of the five rubric properties below in full (SceneDiff's passive occlusion handling). The rest are absent or partial.
2. **The survey missed closer work.** Four papers not in `docs/LITERATURE_REVIEW.md` are closer than [7], [10] and [11] on specific parts of the design:
   - **PInVerify** (arXiv 2605.30639, May 2026). On an offline replay of six pre-captured viewpoints, it compares random, angular farthest-point (a geometric angular-diversity rule close to G) and multimodal-LLM view selection under one shared adaptive stopping rule. The three strategies land "within 0.2 pp of each other". The task differs: it verifies whether a *present* candidate matches a description, in simulation.
   - **PBD-AG** (arXiv 2608.10449, Aug 2026). It admits a missed detection as negative evidence only when a depth-based visibility gate passes, and removes an object only after at least three admissible misses. It reports false absence of 0.014, which rises to 0.262 without the gate.
   - **Li et al., IROS 2021** (arXiv 2107.12095). A robot circles a table in 30° steps and learns a move/stop policy to decide whether a queried object exists behind occluders.
   - **INSPECT** (arXiv 2609.20615, Sep 2026). A learned selector picks the next view on a fixed pose lattice, "without candidate images", evaluated by replay on real images against current-view, uniform, minimum-cost and VLM baselines.
3. **Consequence for the gate: no topic change is needed, but the claims must narrow.** The survey and the proposal wording should change (see [Required changes](#required-changes)):
   - "Learned vs geometric view selection under a shared stopping rule" is no longer an open comparison in general. For identity verification, PInVerify compared a model-driven selector (a zero-shot MLLM prompt, not a trained ranker) with geometric and random selectors under shared stopping, and found no difference.
   - "Conservative removal reduces false removals" (H1) is an established idea: PBD-AG ablates it, and Bogenberger et al. guard against premature deletion. H1 becomes a transfer test to a 2D RGB visibility cue, not a novel claim.
   - What remains distinct is the decision itself: present / removed / unknown for a remembered instance, with matched occluded-present and removed pairs from a physical workspace, where the selector must find the view that tells occlusion from removal.

## Rubric

Each work is classified on five properties of the planned study:

| Code | Property | "Yes" means |
| --- | --- | --- |
| (a) | Viewpoint choice to verify a specific object | The method chooses among candidate viewpoints or view actions in order to decide the state of one specific (queried or remembered) object. Frontier exploration, region patrol, or a given capture sequence do not count. |
| (b) | Stopping rule or abstention | An explicit rule decides when to stop acquiring views, and/or the output includes an abstain/insufficient/unknown option. |
| (c) | Occluded vs removed | The decision logic separates "hidden from this view" from "not there / removed". |
| (d) | Learned vs geometric selection, matched | A learned view selector is compared with a geometric (non-learned) selector under a matched budget and a shared stopping rule. |
| (e) | False-removal-type errors | The evaluation reports, separately, errors where a present object is declared absent, removed, or rejected. |

"Partial" means the property is present in a weaker or different form; the table cell says how. "Not described" means I did not find it in the text I read, which is weaker than "No".

**Evidence levels.**
- **Full:** I downloaded the arXiv PDF, converted it with `pdftotext`, and read the method, experiment and conclusion sections.
- **Targeted:** I searched the full text for specific passages and read the relevant sections only.
- **Abstract:** I read the arXiv abstract page only.

Nothing in the abstract-only screen should be cited as verified.

## Comparison table

| Work | Read | (a) View choice for one object | (b) Stopping / abstention | (c) Occluded vs removed | (d) Learned vs geometric, matched | (e) False-removal errors | Setting |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **This study (planned)** | n/a | Yes: 5 calibrated positions; S, F, R, G, A | Yes: shared sequential rule; unknown output | Yes: matched occluded-present / removed pairs; 2D cue | Yes: A vs G (and F, R), shared stopping, matched budget and travel | Yes: false removal rate is primary | Physical tabletop, RGB, replay |
| PInVerify, 2605.30639 v1 (2026) | Full (main text; appendix passages) | Yes: 6 pre-captured sectors around a candidate object (§3, §4.2) | Yes: adaptive stopping over T=6; per-attribute Yes/No/Unsure; ties resolved to No at the horizon (§5.1) | Partial: "trap views" are down-weighted, "unobserved is not contradicted" (§5.1). The object is always present; the question is identity match. | **Partial, closest found:** random vs angular FPS (geometric) vs MLLM prompt, shared stopping, same budget. No trained lightweight ranker; no travel cost (§5.1, §6.2) | Partial: positive-pair accuracy, with positive failures decomposed into rejection bias (§6.5, App. L) | Simulation (Habitat HM3D + Objaverse objects), 3,000 offline episodes; CVPR 2026 workshop poster |
| PBD-AG, 2608.10449 v2 (2026) | Full (main text; supplement not read) | Partial: fixture-centered inspection views ranked by a hand-set cost (Eqs. 5-6, §3.4); fixtures, not individual objects | Partial: removal only when p_exist ≤ 0.12 after ≥3 visibility-admissible misses; a demotion band below 0.30 (§3.3). No per-object acquisition stopping. | **Yes:** depth-based visibility gate on negative evidence (§1, §3.3) | No, in the main text | **Yes:** false absence 0.014 with the gate vs 0.262 without (§4.3, Table 2) | Simulation (OmniGibson), RGB-D; qualitative real-robot demo |
| Li et al. (ROEP), 2107.12095 v2 (IROS 2021) | Full | **Yes:** circle left/right by 30° or stop, to decide whether a queried object exists (§III-A) | Yes: learned stop action, at most 6 moves; binary output, no abstention (§III-A, Eq. 2) | Yes, as absent vs hidden behind a larger object. No memory of a prior state, so "absent", not "removed" (§III-B). | No: learned policy vs passive, random and exhaustive policies with the same architecture; no geometric selector (§V-B) | No: accuracy only (Table IV) | Simulation (CoppeliaSim, Pepper), at most 2 objects on the table |
| INSPECT, 2609.20615 v1 (2026) | Full | Yes, one step: next pose on a fixed lattice chosen "without candidate images" (§III, §III-D) | Yes: supported / contradicted / insufficient; the robot "stays or defers" (§III-A, §III-D); single step only | Partial: occlusion is only "a quality cue" (§IV-A) | Partial: learned selector vs current-view, uniform, minimum-cost, Qwen3-VL and a claim-conditioned view prior on 132 matched trials sharing trigger, observations and costs. No angular-diversity baseline. (§IV-A, Table I) | Partial: verifier false-support rate; counts of correct commitments, incorrect commitments and abstentions (§IV-D, Table IV) | Real images (gearbox assemblies; IMPACT), replay |
| Atanasov et al., 1309.5401 v1 (2013) | Targeted (§III, §V, §VIII-A) | Yes: viewsphere viewpoints to decide class and pose of a detected object, including a null hypothesis "The object is not a Handlebottle" (§III-C, §VIII-A) | Yes: stopping time is part of the objective (Eq. 1) | Partial: occlusion states in the observation model (§V); no removal | No: nonmyopic POMDP vs greedy MI, random and static. Baselines use a heuristic 60% stop, so stopping is not shared. (§VIII-A) | Partial: separate false-positive and false-negative decision costs K+ and K− (§III) | Simulation + PR2 |
| [10] Bogenberger et al., 2509.19851 v2 (RA-L 2026) | Full | No: waypoints sampled from a 2D exploration priority map, closest of M samples (§IV-C, §IV-D) | Partial: objects move to a "missing" library when stationarity ≤ θr; decay stops at θr "to prevent premature deletion" (§IV-B2, §IV-B3). No acquisition stopping; no unknown output. | Partial: expected-view test by field of view and distance (§IV-B1); no depth-order test described | No: compared with random, patrol and DynaMem (Tables I, V, VI) | Partial: instance precision/recall/F1 and voxel FPR (Tables III, IV); false removals not isolated | Real (Stretch 3, three environments) + public datasets |
| [7] DynaMem, 2411.04999 v2 (ICRA 2025) | Full | No: time- and similarity-based exploration value maps (§3.4) | Partial: "we abstain from returning a location when an object is not found" (§3.3); no acquisition stopping | Partial: ray-cast removal only for voxels in front of observed depth (§3.2, Eq. 2). At query level, "not found" means moved *or* an inaccurate response (§3.3). | No: Table 1 ablates memory and query variants | Partial: negative queries for "observed but ... subsequently removed" objects count toward success (§4.3) | Real (Stretch SE3) + DynaBench offline benchmark |
| VAP-TAMP, 2604.26988 v1 (2026) | Targeted (§IV-B, §V-A) | Yes: a VLM suggests a direction when predicate verification is uncertain (§IV-B) | Yes: VLM view-sufficiency check plus viewpoint budget K (§IV-B) | Partial: the sufficiency prompt asks whether the view is "free from significant occlusion" | Partial: VLM-guided vs a greedy baseline that visits every predefined viewpoint (4.35 on average); budgets not matched; no geometric scorer (§V-A) | No | Real mobile manipulator + simulation |
| [11] Terashima et al., 2307.14105 v2 (2023) | Full | Partial: chooses which object-specific journey to advance next, among N=10 (§IV) | No: exploitation runs "until the budget is exhausted" (§IV-A) | **No, explicitly out of scope** (§III, quoted below) | No: bandit planner vs random planner (§VI) | No: top-1 accuracy (§VI) | Recorded real journeys; navigation run "in a human-in-the-loop manner" (§III) |
| [12] SceneDiff, 2512.16908 v2 (2025) | Full (main paper; parts of the supplement) | No: both captures are given; frames are paired by co-visibility (§4.1) | No: max-entropy threshold, no unknown output (§4.3; Fig. 10) | **Yes, passively:** asymmetric depth rule (§4.2) | No | Partial: AP counts false positives; no separate false-removal rate (§3.1) | Real videos, 350 pairs |

For venue details of [7], [10] and [11], see `docs/CITATION_AUDIT.md`, which verifies them. That audit reports DynaMem at ICRA 2025, Bogenberger et al. at RA-L 11(3) 2026, and Terashima et al. at IEEE/SICE SII 2024.

## The four works from the survey's closest-work section ([7], [10] and [11] are named as closest)

### [7] DynaMem (arXiv 2411.04999 v2, 29 May 2025): full text

- **(a) No.** Exploration uses "two value-based exploration maps: one time-based, and one semantic-similarity-based" (§3.4). In the robot experiments DynaMem lets "the robot explore when an object is not found in the memory" (§4.1). Nothing chooses a viewpoint to check a specific remembered object.
- **(b) Partial.** "Unlike previous work, we abstain from returning a location when an object is not found" (§3.3). This abstention is about localization, not about when to stop gathering evidence. Navigation replans every seven waypoints until the robot is within a distance threshold (§3.4).
- **(c) Partial, geometric.** A voxel is removed only if its projected depth is below the observed depth, "denoting the voxel is between the camera and the currently visible object". Pixels beyond 2 m are ignored (§3.2, Eqs. 1-2, Fig. 3). So a voxel hidden behind a nearer surface is not deleted, which is RGB-D occlusion handling. At query level the distinction collapses: "If we don't find the queried object, we assume that the object has either moved, or the response from the voxelmap or mLLM was inaccurate" (§3.3).
- **(d) No.** The ablations (Table 1) vary memory updates, the OWL-v2 cross-check, thresholding and the mLLM. They do not vary view selection.
- **(e) Partial.** DynaBench includes negative queries for objects "that have been observed but were subsequently removed" (§4.3), but they are folded into one success rate.
- **Answer to the survey's reading question** ("what evidence does DynaMem require before an object leaves memory, and what happens when the object is merely occluded?"): Voxels leave memory when a depth ray passes through them within 2 m. A voxel that is merely occluded (behind the observed surface) is kept. There is no object-level evidence count.

### [10] Bogenberger et al. (arXiv 2509.19851 v2, 14 Jan 2026): full text

- **(a) No.** The per-object exploration map is a Gaussian-smoothed "occupancy shadow" whose width grows as stationarity falls (§IV-C1). For map maintenance it is weighted by a Beta-pdf relevancy of stationarity (§IV-C2). Waypoints are sampled from the error between this map and past waypoints, and the closest of M samples is chosen (§IV-D, Fig. 4). The robot is sent *near* stale objects. It does not choose which view would show a given object.
- **(b) Partial.** "Objects with E[vi] ≤ θr are moved from the active object library Ot to the missing object library Omis,t, where they are kept for potential future reidentification" (§IV-B3). For unseen objects, "To prevent premature deletion from the scene belief, we stop the decay process once the expected stationarity score E[vi] reaches the removal threshold θr" (§IV-B2). Navigation continues "until the task q is complete or updated" (§IV-D). There is no per-object stopping rule for acquisition and no unknown output.
- **(c) Partial.** An object is "expected" if enough of its points project into the current image within distance d_max (§IV-B1). I found no depth-order (z-buffer) visibility test in the text. The update is stated for reobserved objects: "We can directly measure the change Δt,i only for objects that have been reobserved" (§IV-B2). How an expected-but-unmatched object lowers stationarity is not spelled out in the main text.
- **(d) No.** Exploration is compared with random and patrol policies (Tables V, VI) and with DynaMem (Table I). Mapping is compared with Khronos (Table II) and with no-change-detection and full-rebuild variants (Tables III, IV).
- **(e) Partial.** Table III's voxel FPR measures *stale retention*, i.e. failing to remove moved or deleted objects. False removals would show up only as lower recall in Table IV.
- **Answer to the survey's reading question** ("after a revisit fails to observe an object, what does the system do next?"): Its stationarity decays, which widens its region in the priority map and raises the region's relevance. When stationarity crosses θr, the object moves to the missing library and can be re-identified if it reappears.

### [11] Terashima et al. (arXiv 2307.14105 v2, 24 Oct 2023): full text

- **(a) Partial.** The decision is "Which journey should the robot select next?" among N = 10 object-specific journeys of L = 100 images each (§IV). This is closer to a bandit over *which object to approach* than to choosing a viewpoint around one object.
- **(b) No.** The robot selects the top-priority journey "until the budget is exhausted" (§IV-A), and the budget is swept (Figs. 6-7).
- **(c) No, explicitly deferred:** "in cases where a book turns out to be changed, we need to consider a post-process of classifying the book's change state into 'appearance' (e.g., the book was occluded by the newly added objects) or 'disappearance' (e.g., the book was removed from the workspace). However, such a change state classification task is out of the scope of the current work" (§III).
- **(d) No.** The comparison is the bandit planner against a random planner (§VI).
- **(e) No.** The metric is top-1 accuracy of picking the target's journey (§VI).
- **Note for the survey:** the mapless navigation module "was run in a human-in-the-loop manner with experimenter assistance" (§III).

### [12] SceneDiff (arXiv 2512.16908 v2, 31 Mar 2026): full text of the main paper

- **(a) No.** SceneDiff receives two videos. It "selects view pairs with high co-visibility from the before and after videos" (§4), which is selection among frames already captured, not acquisition.
- **(b) No.** Changed regions are thresholded with a maximum-entropy threshold (§4.3). An inherently ambiguous swap is resolved to a single "moved" prediction rather than an unknown (Fig. 10, Limitations).
- **(c) Yes, passively.** "Negative differences are ambiguous, as they can arise from occlusion or from objects that exist only in Ipost. Accordingly, we adopt an asymmetric rule" (§4.2). Regions more than 50% masked are excluded from region matching.
- **(d) No. (e) Partial:** AP counts unmatched predictions as false positives (§3.1). There is no removal-specific false-alarm rate.

## Closer work the survey missed (full or targeted reading)

### PInVerify (arXiv 2605.30639 v1, 28 May 2026)

- Defines "Active Instance Verification": "an agent actively selects viewpoints around a candidate object to decide whether it matches a fine-grained natural-language description" (Abstract), as "a finite-horizon decision process" (§3).
- Uses an offline replay on purpose. "Pre-captured observations avoid rendering stochasticity in evaluation and help isolate the verification decision from locomotion" (§4.6, "Why offline?"). This is the same argument as our replay interface.
- Its geometric baseline is essentially G without occlusion flags or a travel penalty: "Angular FPS picks the unvisited navigable direction ... whose azimuth θk maximizes the minimum shortest-arc distance to visited azimuths" (§5.1, Eq. 4). All strategies share "adaptive stopping over T = 6 steps" (§6.1).
- Its visibility handling matches our "ambiguous visibility contributes no absence evidence": "low-visibility observations (trap views push c̃ to 0.02) cannot single-handedly trigger rejection, encoding the principle that unobserved is not contradicted" (§5.1).
- Its result is a null for view selection: "Random, Angular FPS, and LLM-NBV land within 0.2 pp of each other on Qwen3-VL-4B (0.848–0.850), well inside the 95% CI ... any gain from smarter view selection appears to be swamped by single-step verification noise" (§6.2).
- **Why it does not contain our comparison:**
  - The candidate object is always physically present, so there is no removed class and no remembered pre-change reference.
  - Its "trap views" are an annotation, not a controlled occluded-present vs removed pairing.
  - The learned selector is a zero-shot MLLM prompt. The fine-tuned agent chooses views *and* stops end-to-end, so it does not share the stopping rule.
  - It has no travel cost, and it runs in simulation.

### PBD-AG (arXiv 2608.10449 v2, 12 Aug 2026)

- States our problem almost verbatim: "a missed detection alone cannot distinguish physical removal from occlusion, poor viewpoint, or detector failure. Treating repeated misses as evidence of absence can delete an object that remains present" (§1).
- Its visibility gate: "A missed detection contributes negative evidence only when the predicted 3D extent lies within the camera frustum, is not occluded by closer observed geometry, and has sufficient expected pixel support" (§1). In §3.3 this becomes a depth test, d_meas + 0.15 m ≥ d_expected, on projected box samples.
- Its conservative removal rule: a node "emits remove only when p_exist ≤ 0.12 after at least three visibility-admissible misses" (§3.3).
- It measures the effect: "Without visibility gating, false absence rises from 0.014 to 0.262 and event recall falls from 11/12 to 6/12" (§4.3). This is on three simulated Hotel sequences of 300 frames with 12 scheduled events.
- Inspection views are ranked by a fixed cost that combines standoff, travel, clearance and predicted coverage (Eq. 6). The conclusion mentions "paired acquisition experiments" that "support graph-conditioned fixture prioritization under bounded budgets". I did not find these in the main text; they may be in the supplement, which I did not read.
- **Why it does not contain our comparison:**
  - Views are scheduled per fixture (support surface), not per remembered object.
  - The main text has no view-selector comparison and no learned selector.
  - It uses RGB-D depth for visibility, in simulation.

### Li et al., "Robotic Occlusion Reasoning for Efficient Object Existence Prediction" (arXiv 2107.12095 v2; IROS 2021)

- Task: "when being asked about an object, a robot needs to move as few steps as possible around a table with randomly placed objects to predict whether the queried object exists" (Abstract). Actions are circle left or right by 30°, or stop, with at most 6 moves (§III-A). A third of scenes contain a smaller object hidden behind a larger one (§III-B).
- The baselines "have the same architecture as the proposed model, but with different action selection strategies": passive, random, and exhaustive (§V-B).
- On occluded scenes (Table IV) the accuracies are 77.6% passive, 83.3% random, 97.4% exhaustive (6 moves) and 96.9% learned. The learned policy averages 0.71 moves overall.
- A limitation relevant to H3: "The model learns to always choose one direction to move, rather than choose the optimal direction according to the orientation of the visible object or partial occlusion to check the occluded space" (§V-B). In effect, the learned selector collapsed to a fixed order.
- **Why it does not contain our comparison:**
  - The query is a category word with no prior memory, so the negative class is "never there", not "removed".
  - It has no abstention and no geometric selector.
  - It runs in simulation with at most two objects.

### INSPECT (arXiv 2609.20615 v1, 17 Sep 2026)

- "Given RGB o_t at pose V_t, candidate poses V, and claim c, the robot selects V_k ∈ V without candidate images. It can retain V_t or defer a decision" (§III). The view score subtracts a motion cost λd·d(V_t, V_k) (Eq. 10), as A and G do.
- The verifier returns supported, contradicted or insufficient (§III-A). Inspection is "triggered only when the current prediction abstains" (§IV-D).
- Evaluation uses "offline one-step replay of recorded observations", and "Policies share the frozen trigger, observations, and costs" (§IV-A).
- On 132 matched gearbox trials, human-rated utility is 1.205 vs 1.035–1.098 for the non-oracle baselines, and Verifiable@1 is 0.417 vs 0.326–0.356 (Table I). *(Evidence check: corrected from "1.043–1.098". In Table I the Uniform Non-current View baseline scores 1.035. The non-oracle baselines also include a "Claim-conditioned view prior", at 1.098 / 0.356.)* On IMPACT, the correct-decision rate rises from 50.6% to 54.3% (Table III).
- **Why it does not contain our comparison:**
  - Claims are about part identity and installation state in assembly. Occlusion is only a quality cue, and there is no removed-vs-occluded decision.
  - Selection is a single step, not a sequential budget.
  - It has no angular-diversity geometric baseline; the closest is "Minimum-Cost View".

### Atanasov, Sankaran, Le Ny, Pappas, Daniilidis, "Nonmyopic View Planning for Active Object Detection" (arXiv 1309.5401 v1, 2013)

This is the classical framing of our decision: "choose a stopping time τ, a sequence of viewpoints ..., and a hypothesis ... which minimize the total cost" of movement plus a decision cost with separate false-positive and false-negative penalties (§III-C, Eq. 1; §III).

- The hypotheses include a null ("The object is not a Handlebottle") (§VIII-A).
- The observation model has explicit occlusion states (§V).
- Its baselines (random walk, greedy mutual information) use a heuristic stop at 60% posterior, while the proposed planner optimizes stopping (§VIII-A). Stopping is therefore not shared, and the comparison is not learned vs geometric.
- **Use in our study:** it is a citable source for the "sequential hypothesis test with view selection" framing. Our shared, non-probabilistic stopping rule is a deliberate simplification of that framing.

### Detector-confidence audit (arXiv 2607.13361 v1, 15 Jul 2026): targeted reading

This is a perception risk rather than a novelty threat. It pairs every rendered view with a geometry oracle and reports that open-vocabulary detector confidence "stays nearly constant and uncorrelated with visibility". The detector "reports the target present in about nine of ten scenes, firing on same-category distractors", and "No single-view signal we tried, including a realizable localization check, flags the occlusion" (Abstract). The paper does include an active-view condition (24 hemisphere views, LIBERO simulation). It does not study stopping, removal, or learned vs geometric selection.

### VAP-TAMP (arXiv 2604.26988 v1, 28 Apr 2026): targeted reading

- A VLM verifies task predicates such as "Object inview", and "the verification process repeats with the new observation until the VLM confirms sufficient information or a viewpoint budget K is exhausted" (§IV-B).
- Its baseline, greedy perception, "exhaustively visits predefined viewpoints" and averages 4.35 views (§V-A). Budgets are not matched, and there is no geometric or trained selector.

## Abstract-only screen (not verified in full text)

These surfaced in search or citation chasing and are related, but from the abstract none has (a) together with (c) for a remembered object. They should not be cited as verified without reading them.

| arXiv | Title (short) | Closest aspect | Why not the planned comparison (from the abstract) |
| --- | --- | --- | --- |
| 2403.15941 | Explore until Confident (RSS 2024) | Conformal calibration of when to stop exploring for embodied QA | Room-scale QA; no occlusion-vs-removal decision |
| 2507.12846 | Long-term Active EQA ("Mind Palace") | Value-of-information stopping over memory plus exploration | Temporal QA over scene graphs |
| 2410.20263 | EfficientEQA (IROS 2025) | Adaptive stopping via relevance outliers | Exploration for QA |
| 2512.04597 | AbstainEQA | Abstention benchmark for EQA | Ambiguous questions; no active view choice |
| 2604.17969 | E3VS-Bench | 5-DoF viewpoint-dependent evidence gathering | VLM benchmark; no removal |
| 2608.13605 | Active Perception for Embodied Disambiguation | VLM decides continue observing / ask / commit | Target disambiguation |
| 2608.17129 | PROBE (MG-VQA) | "Is my medication still in the cabinet?" with occluders | Resolved by *manipulation*, not viewpoint choice |
| 2409.17641 | AP-VLM | VLM-chosen viewpoints for occluded objects vs passive | No stopping or selector comparison stated |
| 2609.08159 | OmniNav | Failed searches as negative evidence in a Bayesian target belief | Navigation-level |
| 2607.23797 | Memory for Attention (VLMM) | Which object to re-perceive under a budget | Schedules re-perception; not view choice |
| 2606.06721 | SCOUT | Viewpoints trading semantic certainty gain, coverage and travel | Label certainty, not presence |
| 2608.13923 | OpenBelief-Nav | Arrival verification with at most two candidate attempts | Navigation target confirmation |
| 2609.20443 | Spatial-Semantic Uncertainty in VLM Target Search | EIG planner with a confidence-based stop vs random | Search-and-identify |
| 2603.11072 | OA-NBV | Occlusion-aware NBV for a partially hidden person | Single-view quality, humans |
| 2511.18353 | UAV Search under Occlusion with NBV | Geometry vs visibility heuristics for hidden objects | Search coverage; no learned selector |
| 2605.10707 | ObjView-Bench | Classical, learned and hybrid planner rankings change with budget | Reconstruction |
| 2608.16499 | OccamView | Object-conditioned, occlusion-aware view scoring | Reconstruction |
| 2609.23504 | Imagine then Verify (ATAP) | Viewpoints chosen to verify an occluded affordance | Grasping |
| 2607.14899, 2605.00121, 2606.28720, 2606.00576 | OASIS-Map, PredictiveGraphs, CubifyGS, DREAM | Object-level change handling in maps | Passive or map-update methods |

## Ranked closest threats

1. **PInVerify (2605.30639).** Closest on the *protocol*: a finite replayed view set, a shared stopping rule, random vs geometric angular-FPS vs model-based selection, and "unobserved is not contradicted". It also reports a null result for selection. **Threat:** any claim that "comparing learned and geometric view selection under shared stopping" is new. **Not covered:** presence vs removal of a remembered instance, occluded-present controls, a trained ranker, travel cost, and physical data.
2. **PBD-AG (2608.10449).** Closest on the *failure mode and metric*: it gates negative evidence by visibility, removes conservatively after ≥3 admissible misses, and measures false absence with and without the gate. **Threat:** H1 as a novelty, and the claim that separating occlusion from removal is new. **Not covered:** comparing view selectors for per-object verification.
3. **Li et al. IROS 2021 (2107.12095).** Closest on the *task shape*: move among discrete views around a table with a learned stop to decide if a queried object exists behind occluders. **Threat:** any claim that "active view choice plus stopping to decide presence under occlusion" is a new task. **Not covered:** memory of a prior state, removal, abstention, a geometric selector, and real data.
4. **INSPECT (2609.20615).** Closest on the *learned-selector setup*: learned view choice on a fixed pose lattice without candidate images, replay, matched trials, and an abstaining verifier, on real images. **Threat:** claims about "learned view selection without peeking at candidate views". **Not covered:** sequential budget, a geometric angular baseline, and occlusion vs removal.
5. **Atanasov et al. 2013 (1309.5401).** Classical active hypothesis testing with view planning, optimal stopping, occlusion states and asymmetric error costs. **Threat:** framing the decision model as new.
6. **[10] Bogenberger et al.** Active map maintenance with stationarity-driven revisits and a premature-deletion guard. It is region-level, has no selector comparison, and has no per-object stopping.
7. **[7] DynaMem.** Abstention and ray-cast removal; no view selection.
8. **VAP-TAMP (2604.26988).** VLM view choice with a sufficiency stop for predicate verification; unmatched greedy baseline.
9. **[11] Terashima et al.** Budgeted choice of object journeys. Explicitly leaves occluded-vs-removed as future work.
10. **[12] SceneDiff.** Passive occlusion-aware change classification on given captures.

## Recommended distinction statement (for the gate)

> Prior work has compared random, geometric (angular farthest-point) and foundation-model view selection under a shared stopping rule for verifying whether a *present* candidate matches a description (PInVerify, simulation). It has gated negative evidence by geometric visibility to reduce false deletions in persistent robot maps (PBD-AG, RGB-D simulation), and learned move/stop policies for deciding whether a queried object exists behind occluders (Li et al. 2021, simulation). We study a narrower decision that these works do not evaluate together. After an unobserved change, the robot must decide whether a specific remembered instance is present, removed from the monitored area, or undecidable, using five calibrated RGB views of a physical tabletop, with matched occluded-present and removed pairs. In that setting we compare a small learned view ranker with a geometric parallax selector and with fixed and random orders. All four share one conservative stopping rule and matched view and travel budgets, and we report false removals together with removal recall, coverage and effort. The contribution is this controlled empirical comparison and the replayable dataset, not a new selection algorithm or a new stopping principle.

**What would have to be true for this distinction to fail:**
1. **An unfound paper already runs it.** Some paper already compares view-selection policies under a shared stopping rule for present/removed(/unknown) verification of a remembered instance, with occluded-present controls. The most concrete unchecked lead is the PBD-AG supplement ("paired acquisition experiments ... under bounded budgets"). Work only on IEEE Xplore or in proceedings not indexed by arXiv is also under-covered (see Limitations).
2. **The physical data cannot separate occlusion from removal.** If few occluded-present episodes have any available view that shows the target, the study reduces to measuring abstention and coverage, and the selection comparison has nothing to act on. The pilot should report the fraction of occluded-present episodes with at least one diagnostic view.
3. **The selection comparison is degenerate in this design.** With 5 positions and a fixed start there are 4 unvisited positions. At the primary budget of 3 additional views, a policy that does not stop early sees 3 of the 4, so policies differ only in which one position they skip and in the order, which changes when the stopping rule fires. This assumes the cumulative travel cap does not bind. If it excludes far positions, the policies' choices can diverge more (`docs/EXPERIMENT_PLAN.md`: only positions "affordable under the remaining budget" are eligible). At 4 additional views every policy covers all positions. If A and G pick the same views in most episodes, no difference can appear. The finding then shrinks to "selection did not matter here", which PInVerify already reports in its setting.
4. **Perception decides the outcome, not selection.** If the 2D occlusion cue cannot tell occluded views from clear ones, "clear miss" evidence is unreliable, and false removals come from perception rather than from view choice. The detector-confidence audit (2607.13361) makes this a live risk.
5. **The claims outrun the data.** The distinction holds only for the tested workspace, objects, views and models. It does not hold for "robot scene memory" in general.

## Scope flags

- **No forced scope or topic change.** None of the works found contains the planned comparison. The gate condition "the closest work already contains this comparison" is not met on this evidence.
- **Flag, claim wording (must change).** The proposal and survey call [7], [10] and [11] the closest active methods. After this check, PInVerify, PBD-AG, Li et al. 2021 and INSPECT are closer on specific parts of the design. Treat H1 and the general "learned vs geometric under shared stopping" comparison as replications or transfers of known ideas, not as novel.
- **Flag, likely null result (plan for it).** Two direct precedents found no benefit from smarter selection. PInVerify's three NBV strategies landed within 0.2 pp. The learned policy in Li et al. collapsed to one direction. With 20 test arrangements, an inconclusive A-vs-G interval is a likely outcome. The proposal already treats that as acceptable. It should now say *why* it is likely, citing these precedents.
- **Flag, perception risk (pilot must test).** The detector-confidence audit (2607.13361, targeted reading) reports that open-vocabulary detector confidence "stays nearly constant and uncorrelated with visibility" and fires on same-category distractors. This supports the proposal's reliance on reference-crop instance matching rather than detector confidence. It also means the 2D occlusion cue must be validated against visibility annotations in the pilot, which the proposal already plans, before any "clear miss" counts as absence evidence.

## Required changes

These are for the student to decide and apply. Nothing was changed in the proposal files.

1. **Add the four missed papers to `docs/LITERATURE_REVIEW.md`.** Add PInVerify, PBD-AG, Li et al. 2021 and INSPECT to the survey's "Closest work and candidate distinction" section, and add Atanasov et al. as the classical framing. Update "The early literature gate will check ... especially in [7, 10, 11]" to reflect this check. Add matching BibTeX entries.
2. **Reword the contribution in `docs/PROPOSAL.md`** (and the README abstract if it is revised) along the lines of the distinction statement above. Drop any implication that shared-stopping selector comparisons or conservative removal are new.
3. **Reframe H1.** Test whether the known benefit of conservative, visibility-gated removal (PBD-AG) carries over to a monocular RGB, 2D-overlap visibility cue. Do not present it as a finding in itself.
4. **Optional, decide before freezing settings:** add PInVerify-style angular FPS (no occlusion flag, no travel penalty) as an extra *reference* selector in replay. It costs nothing to run and makes G comparable to a published geometric rule. Under the protocol's non-negotiables it must use the same shared stopping rule and be declared before test collection.
5. **Emphasize budgets of 1 and 2 additional views** in the results. At those budgets view choice matters most (see failure condition 3), so they deserve equal weight with the primary point of 3.

## Method

**Date of all searches and downloads:** September 26, 2026.

**Full-text retrieval** (for every paper marked Full or Targeted):

```
curl -sL -o <id>.pdf https://arxiv.org/pdf/<id>
pdftotext <id>.pdf <id>.plain.txt
```

Versions read: 2411.04999v2, 2509.19851v2, 2307.14105v2, 2512.16908v2, 2605.30639v1, 2608.10449v2, 2107.12095v2, 2609.20615v1, 1309.5401v1, 2604.26988v1, 2607.13361v1. Quotes above were checked by string search against these text files. Table values were read from the converted text, where table layout can scramble columns; the numbers quoted come from prose sentences or were cross-checked against the surrounding text.

**Abstract screen:** arXiv abstract pages, fetched with `curl -sL https://arxiv.org/abs/<id>` and parsed for title, versions, comments and abstract.

**Citation chasing:** forward citations from the Semantic Scholar Graph API:

```
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:<id>/citations?fields=title,year,externalIds,venue&limit=1000"
```

| Paper | Citing papers returned |
| --- | --- |
| DynaMem | 59 |
| Bogenberger et al. | 9 |
| Terashima et al. | 2 |
| SceneDiff | 1 |
| Li et al. 2021 | 10 |
| Atanasov et al. | 9 |
| PInVerify | 0 |
| PBD-AG | 0 |

PBD-AG and PlenoCI (2609.28930) were found this way.

**Web searches (22 phrasings, WebSearch tool):**
1. active view selection verify object removed or occluded robot memory stopping rule
2. next-best-view planning object search occlusion 2025 arXiv
3. embodied change detection robot revisit abstention "unknown" object disappeared 2025
4. active perception object permanence robot occluded object memory 2025
5. information gain view planning object existence verification household robot
6. embodied question answering moved objects changed environment exploration confidence stopping 2025
7. semantic next-best-view learned versus geometric heuristic viewpoint selection comparison occluded target
8. active visual verification household robot object still there before acting VLM viewpoint
9. active scene change detection next best view Gaussian splatting robot 2026
10. lifelong object search semi-static environment belief object moved POMDP occlusion viewpoint 2025
11. active hypothesis testing sequential view selection robot object present absent stopping time
12. retail shelf monitoring robot out-of-stock occlusion multiple viewpoints false alarm
13. false deletion occlusion visibility check object map maintenance robot revisit viewpoint selection 2026
14. learned next-best-view vs heuristic baseline active object detection occluded tabletop matched view budget
15. conformal prediction active perception when to stop looking robot viewpoint 2025
16. "missing object" robot verify absence multiple views occlusion reasoning arXiv 2026
17. robot re-observe object before trusting memory stale map verification viewpoint RA-L 2026
18. CoRL 2025 active perception viewpoint selection occluded object presence verification tabletop
19. view planning revisit change detection object disappeared occluded ambiguity "active" arXiv 2025 2026 semi-static
20. "object existence" OR "object presence" active perception learned policy vs heuristic baseline occlusion robot 2024 2025
21. ICRA 2026 OR IROS 2026 active perception verify object disappeared occlusion viewpoint planning semi-static map
22. RSS 2026 active information gathering stopping rule object search abstain uncertain robot

**arXiv abstract search** (arxiv.org/search, `searchtype=abstract`, newest first, first 50 results):

| Query | Results |
| --- | --- |
| active view selection change detection | 7 |
| occlusion removed object robot memory active | 0 |
| next best view object presence verification | 0 |
| active verification object occluded removed | 0 |
| semi-static active exploration | 7 |
| embodied change detection | 43 |
| object permanence active perception | 2 |
| false deletion occlusion | 1 (PBD-AG) |
| object existence prediction | 50, mostly unrelated |
| active object verification | 50; surfaced PInVerify |

**How the key papers were found:**

| Paper | Found by |
| --- | --- |
| PInVerify | Web search 18; arXiv query "active object verification" |
| PBD-AG | DynaMem forward citations; arXiv query "false deletion occlusion" |
| Li et al. 2021 | Web searches 16 and 20 |
| INSPECT | Web searches 1, 4, 8 and 14 |
| Atanasov et al. | Web search 11 surfaced the related ICRA 2013 paper by Sankaran, Atanasov et al. ("Hypothesis Testing Framework for Active Object Detection"); I then fetched the arXiv journal version, 1309.5401, directly |

## Limitations of this check

- **Venue coverage is by search, not by scanning proceedings.** CoRL 2025, ICRA 2026, IROS 2026, RSS 2026 and RA-L 2025-2026 were reached only through web searches with venue terms and through arXiv. Papers that exist only on IEEE Xplore or in proceedings without an arXiv version are under-covered.
- **Older active-recognition work was not read.** Learned next-best-view for object recognition (2014-2019) establishes that learned view selection is not new in general. The distinction statement does not rely on it being absent.
- **Some documents were read only partly.** The PBD-AG supplement and the PInVerify appendix beyond the passages cited were not read. Atanasov et al., VAP-TAMP and the detector-confidence audit were read only in the sections listed ("Targeted").
- **Recency.** Three of the closest papers appeared on arXiv in May-September 2026 (PInVerify, PBD-AG, INSPECT). More may appear before the final report, so this check should be repeated before the final report.

## Evidence check (September 26, 2026, 13:05–13:25)

A second pass re-read the cited passages behind the claims that drive the distinction statement. Each
passage was fetched independently with WebFetch (arXiv abstract and HTML pages). Where WebFetch
returned only a PDF (2107.12095v2, 2307.14105v2), the PDF was run through `pdftotext` and
string-searched. Table I of INSPECT was read from `curl https://arxiv.org/html/2609.20615v1`.

**Confirmed:**
- **PInVerify:**
  - the Angular FPS definition (§5.1, Eq. 4);
  - "within 0.2 pp ... (0.848–0.850)" (§6.2);
  - "unobserved is not contradicted" (§5.1);
  - T = 6 with shared adaptive stopping over 3,000 episodes;
  - "Why offline?" (§4.6);
  - the fine-tuned agent does not share the stopping rule;
  - no travel cost;
  - Habitat HM3D plus Objaverse-XL objects;
  - a CVPR 2026 FMEA workshop poster.
- **PBD-AG v2:**
  - the §1 removal/occlusion quote;
  - the d_meas + 0.15 m ≥ d_expected gate;
  - p_exist ≤ 0.12 after ≥ 3 admissible misses, with a < 0.30 demotion band;
  - 0.014 vs 0.262 false absence and 11/12 vs 6/12 event recall on three Hotel sequences;
  - no view-selector comparison anywhere in the HTML, with "paired acquisition" appearing only in
    the conclusion;
  - RGB-D in OmniGibson.
- **Li et al. 2021:**
  - circle left/right 30° or stop, at most 6 steps;
  - one third of scenes occluded, at most two objects;
  - Table IV L3-2-occ: 77.6 / 83.3 / 97.4 / 96.9 %;
  - 0.71 steps overall;
  - the "always choose one direction" quote;
  - CoppeliaSim with Pepper.
- **Terashima et al. v2:**
  - the "out of the scope of the current work" quote (§III);
  - "human-in-the-loop manner with experimenter assistance" (§III);
  - "until the budget is exhausted" (§IV-A);
  - top-1 accuracy against a random planner (§VI).
- **INSPECT:**
  - the "without candidate images ... retain V_t or defer" quote;
  - the λ_d·d(V_t, V_k) motion cost;
  - supported / contradicted / insufficient;
  - one-step replay over 132 matched trials;
  - no angular-diversity baseline;
  - occlusion as "a quality cue".
- **DynaMem v2, Bogenberger v2, SceneDiff v2 and the detector-confidence audit (2607.13361):** all
  quoted passages found.
- **Semantic Scholar:** the forward-citation counts for DynaMem (59), Li et al. (10) and PBD-AG (0)
  reproduce, and PBD-AG is in DynaMem's citation list.

**Corrected inline:**
- INSPECT's baseline utility range: 1.035, not 1.043, is the lower end.
- The verdict's "four works the proposal names as closest": the proposal names three, [7, 10, 11].

**Added:** a travel-cap caveat to failure condition 3.

**Not re-checked:** Atanasov et al., VAP-TAMP, the abstract-only screen, and the PlenoCI mention
(2609.28930). PlenoCI is named in the Method section but not discussed anywhere.

