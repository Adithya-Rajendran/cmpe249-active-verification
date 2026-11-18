# Occlusion Aware Active Verification for Robot Scene Memory

CMPE 249 Project Proposal | Fall 2026  
Adithya Rajendran | Solo project | September 8, 2026  
Instructor Kaikai Liu  
Selected track 4 VLA / robotics | Research project  
Secondary alignment with track 6 Safety and evaluation

## Abstract

A robot that revisits a workspace may fail to detect a remembered object because it was removed, it is occluded, or the detector missed it. Visibility reasoning helps interpret a miss, but deciding whether to acquire another view also requires a sensing policy. I propose a bounded study of that decision using five calibrated camera positions and three possible outputs: present, removed from the monitored area, or unknown. A small learned view ranker will be compared with fixed, random, and geometric parallax selectors using the same perception front end, stopping rule, and observation and travel limits. A passive single-view rule provides a no-movement reference. The main question is whether learning improves on the geometric selector, and under which occlusion conditions. No experimental results are claimed at proposal time.

## Problem and motivation

Prematurely deleting a present object can trigger an unnecessary search or an incorrect downstream plan. Retaining an outdated record also has a cost. For this study, I prioritize avoiding false removals while requiring useful removal recall and decision coverage. Shelf inspection and household object memory motivate the task; transfer to those applications would need separate evaluation.

Existing systems already reason about changing object state. POCD combines per-object stationarity with geometric change in a Bayesian update [1]; Khronos separates short-term motion from long-term change [2]; Bore et al. model local and global object motion [3]. Persistence filters provide survival estimates [4, 5]. LT-Mem preserves alternative object histories [6], DynaMem updates a spatial memory and guides exploration [7], RoboStream maintains temporal context for manipulation [8], and Lost & Found tracks changes through observed interactions [9]. Bogenberger et al. prioritize exploration using object stationarity and task relevance [10]; Terashima et al. select distant-object inspection journeys [11]. SceneDiff detects changes from an acquired multiview sequence [12].

The candidate contribution is a controlled comparison of view selection and stopping for one ambiguous object verification episode. The survey identifies substantial overlap; it does not establish that this decision has never been studied. The week-2 review will check the distinction against the closest active methods [7, 10, 11].

## Problem formulation

Research question: under a fixed observation and travel budget, does adaptive view selection with conservative stopping reduce false removals, at comparable coverage, relative to passive visibility reasoning and to fixed, random, and geometric inspection orders? Secondary question: does a learned ranker add anything over parallax geometry, and under which conditions?

**Decision model.** The hidden target state is present or removed. Five known camera positions have measured travel costs, and each episode starts at the same designated position. Reference captures from all positions are taken before the change. Each requested query view produces detector, instance-match, occlusion, and image-quality scores. The policy selects an unvisited view or returns present, removed, or unknown within observation and travel caps. The aim is to reduce false removals while maintaining removal recall and coverage. A budget-limited sequential rule confirms presence on a strong instance match and removal only after sufficient clear, unmatched evidence from at least two distinct positions. It returns unknown if evidence remains insufficient when no further view is affordable. This is an empirical evidence rule, not a test with guaranteed statistical error bounds.

| **Element** | **Specification** |
| --- | --- |
| Inputs | Reference captures and target crops from every camera position; a named target; the current and previously requested query views only; position identifiers and travel costs. |
| Outputs | Acquire-view actions and one terminal decision: present, removed from the monitored area, or unknown. A log of views used, evidence scores, and cumulative travel. |
| Scope | One tabletop or shelf, 6 to 10 household objects, five fixed camera positions, a static scene within an episode. Within-area relocation is excluded from the primary task. Removed is a claim about the monitored area, not about the object's existence; unknown is an abstention and counts against coverage. |

Reliable depth and a complete geometric model can simplify visibility checks. This study uses monocular RGB and reference image regions, without a complete 3D model of the changed scene. Known camera geometry still supports a parallax heuristic, which makes G an essential baseline. A result favoring G would support the simpler selector within the tested conditions.

## Proposed technical approach

The implementation will use Python, OpenCV, and PyTorch with locally runnable models. Fixed camera positions and reference captures reduce localization and matching demands. Every policy uses the same frozen perception front end.

**1. Perception and matching.** A small YOLO-World checkpoint [13], prompted with the target and occluder catalog, proposes candidates. A frozen image encoder provides reference-crop similarity, combined with expected-region overlap for instance matching. The checkpoint and matching thresholds will be selected in the pilot and recorded. SAM 2 [14] is optional if masks improve the occlusion cue over boxes.

**2. Evidence and the occlusion cue.** Each reference capture defines an expected target region. The initial cue measures its overlap with detected non-target objects that may obstruct it; optional masks refine this estimate. Raw image difference alone cannot identify occlusion, since removal and lighting changes also alter pixels. Absence evidence requires adequate image quality, low estimated obstruction, and no target match. Ambiguous visibility contributes no absence evidence. This 2D cue cannot prove depth order; its errors will be measured against visibility annotations, with any refinement decided during the pilot.

**3. Passive visibility rule (baseline S).** Use the initial view only: present on a strong match; removed when the expected region is estimated clear, image quality is adequate, and no match is found; unknown otherwise. This is a deliberately simple project baseline inspired by geometric and persistence reasoning [1, 2, 4, 7], not a reproduction of those systems.

**4. Geometric parallax selector (baseline G).** Among affordable unvisited positions, maximize the minimum angular separation from views flagged as occluded, measured about the reference workspace center, minus a normalized travel penalty. If no view is flagged, use all visited positions. Ties favor shorter travel. This geometric diversity heuristic does not assume that an unvisited view has a clear line of sight.

**5. Learned view ranker (proposed A).** A regularized logistic regression predicts whether an unvisited position will provide diagnostic evidence. Inputs are observed evidence and image quality, reference information, and candidate geometry relative to observed views. Development annotations provide training labels. The score subtracts a travel penalty from predicted informativeness. A and G use normalized scores and costs with the same penalty coefficient. The small model limits learning demands; overfitting will still be checked across development arrangements. Future query images are never inputs.

**6. Stopping.** F, R, G, and A share the sequential evidence rule, so their main comparison changes only view selection. Feature definitions, regularization, evidence thresholds, and the travel penalty will be selected using validation folds grouped by development arrangement, then frozen before final testing. Distinct camera views are not assumed to provide independent evidence.

## Experiment and data

A pilot of 24 to 30 episodes comes first and stays out of the test set. The main collection uses 30 physical arrangements, each captured under three target conditions (visible, present but occluded, removed) and two lighting conditions: 180 episodes, five query views each, about 900 query images plus references. Occluded-present and removed captures use matched occluder layouts, reducing label leakage; each scene remains static within an episode. A small, predeclared subset includes paired states that cannot be distinguished from any available view. Unknown is the intended response for these cases, while physical ground truth remains present or removed.

Split by arrangement: 10 development, 20 held-out test, giving 60 development and 120 test episodes with all conditions of an arrangement in one split. This tests new arrangements, not unseen object identities. A replay interface serves only the requested view, so policy comparisons are repeatable and unvisited images cannot leak. SceneDiff [12] is a methodological reference and a source of extra examples, not a navigable environment.

The 120 test episodes come from only 20 independent arrangements, so sensitivity to small effects is uncertain. The pilot will guide the collection plan and identify the coverage limit imposed by unobservable cases. A synthetic replica is an optional extension after the physical study; it cannot replace independent physical evidence.

## Baselines and evaluation

| **Policy** | **View choice** | **Decision rule** |
| --- | --- | --- |
| S  Passive single view | Initial position only | Visibility rule, no movement |
| F  Fixed order | Left to right | Shared sequential stopping |
| R  Random order | Random unvisited position, repeated seeds | Shared sequential stopping |
| G  Geometric parallax | Angular diversity minus travel penalty | Shared sequential stopping |
| A  Learned ranker (proposed) | Logistic informativeness minus travel penalty | Shared sequential stopping |
| Full sweep | All five positions | Higher-budget reference, reported separately |

Budgets range from zero to four additional views from the same initial position, with common observation and cumulative travel caps; the primary operating point is three additional views. S uses no movement and is reported at its actual effort. A versus F, R, and G isolates selection under shared stopping. Ablations remove the occlusion cue from A and replace conservative removal with removal on the first clear miss for each multi-view policy. These tests separate the effects of view choice and stopping. The full sweep is a higher-budget reference, not a guaranteed accuracy upper bound.

| **Metric** | **Definition and planning target** |
| --- | --- |
| False removal rate | Removed decisions over truly present episodes. Target at most 10% at three additional views. |
| Removal recall | Correct removed decisions over truly removed episodes. Target at least 80%. |
| Decision coverage | Present or removed decisions over all episodes. Target at least 80%; unknown stays in the denominator. |
| Resolved accuracy | Correct decisions among non-unknown decisions. |
| Inspection effort | Views requested and cumulative commanded travel; full error, coverage, and effort curves across budgets. |

The three planning targets are evaluated jointly at the primary operating point; no target is a guaranteed outcome. Paired differences use arrangement-level bootstrap intervals, with counts alongside every rate. The primary selection comparison is A versus G. A small observed difference with a wide interval is inconclusive, not evidence of equivalence.

## Hypotheses

H1: conservative stopping reduces false removals relative to first-clear-miss removal for the same selector, potentially reducing coverage. H2: A improves the error, coverage, and effort trade-off over F and R. H3: any advantage of A over G depends on occlusion and image quality. I expect G to be competitive when occluders are compact and reference regions are reliable. These are hypotheses to test. Failure to detect a difference will not establish that learning has no value; sufficiently precise results may support choosing G for this bounded task.

## Devices

ASUS Ascent GB10 for development, inference, image processing, and analysis; Jetson Orin Nano 8 GB for optional onboard perception; Raspberry Pi 5 with AI HAT+ and a Pi camera for capture and optional motor control. Accelerator deployment is optional until compatibility is verified. I, Adithya Rajendran, personally purchased, own, and maintain all listed devices and manage software installations and development access. The motorized camera stage remains an optional build.

## Milestones and feasibility gate

| **Period** | **Deliverable or decision** |
| --- | --- |
| Weeks 1 to 2 | Read [1, 7, 10, 11]; run the front end on the GB10; collect the pilot; implement S and F and a basic G; verify that changing views resolves some occlusions. |
| Weeks 3 to 4 | Reference catalog, capture protocol, replay interface, development split; implement the sequential stopping rule, G, and the ranker. |
| Weeks 5 to 7 | Freeze settings; collect the held-out arrangements; run equal-budget comparisons; inspect failures. |
| Weeks 8 to 10 | Ablations, uncertainty analysis, report, demonstration; motorized camera stage if the core study is stable. |

Gate at the end of week 2: if the front end fails on clearly visible targets, shrink the object set or swap the detector. If no view change resolves occluded cases, change camera spacing or occluder geometry before final collection. If the closest work already contains this comparison, or no credible experiment survives the pilot, I revise the scope or the topic with the instructor.

## Contribution and limitations

Deliverables are a reproducible verification task, a replay interface, the view selectors and stopping rule, and an evaluation of their error and sensing costs. The intended contribution is the controlled comparison, with novelty rechecked against [7, 10, 11] at the early gate. Conclusions apply to the tested workspace, objects, viewpoints, and models. A well-supported negative result is acceptable; inconclusive results will be identified as such. No experiments have been completed. The repository will contain the README, literature survey, experiment protocol, and AI audit.

## AI assistance

**Proposal support:** ChatGPT and Claude assisted with drafting, literature review, and the AI audit.  
**Testing automation:** Codex or Claude Code will help automate test runs and result checks under my review.

## References

[1] Jingxing Qian, Veronica Chatrath, Jun Yang, James Servos, Angela P. Schoellig, Steven L. Waslander (2022). [POCD: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes](https://arxiv.org/abs/2205.01202). RSS 2022.
[2] Lukas Schmid, Marcus Abate, Yun Chang, Luca Carlone (2024). [Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments](https://arxiv.org/abs/2402.13817). RSS 2024.
[3] Nils Bore, Johan Ekekrantz, Patric Jensfelt, John Folkesson (2019). [Detection and Tracking of General Movable Objects in Large Three-Dimensional Maps](https://arxiv.org/abs/1712.08409). IEEE Transactions on Robotics 35(1).
[4] David M. Rosen, Julian Mason, John J. Leonard (2016). [Towards Lifelong Feature-Based Mapping in Semi-Static Environments](https://research.google/pubs/towards-lifelong-feature-based-mapping-in-semi-static-environments/). ICRA 2016.
[5] Amanda Adkins, Tarunvidyut Ravisankar, Joydeep Biswas (2026). [PreSIST: Vision-Language-Informed Object Persistence Prediction in Open-World Scenes](https://arxiv.org/abs/2607.04057).
[6] Yumin Lee, Hyoseok Ju, Giseop Kim (2026). [LT-Mem: Volatility-Aware Spatio-Temporal Memory for Lifelong Scene Understanding](https://arxiv.org/abs/2608.19059).
[7] Peiqi Liu et al. (2024). [DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation](https://arxiv.org/abs/2411.04999).
[8] Yuzhi Huang et al. (2026). [RoboStream: Weaving Spatio-Temporal Reasoning with Memory in Vision-Language Models for Robotics](https://arxiv.org/abs/2603.12939).
[9] Tjark Behrens, Rene Zurbruegg, Marc Pollefeys, Zuria Bauer, Hermann Blum (2025). [Lost & Found: Tracking Changes from Egocentric Observations in 3D Dynamic Scene Graphs](https://arxiv.org/abs/2411.19162). IEEE RA-L 2025.
[10] Benjamin Bogenberger et al. (2026). [Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in Real-World Semi-Static Environments](https://arxiv.org/abs/2509.19851). IEEE RA-L 11(3).
[11] Kouki Terashima, Kanji Tanaka, Ryogo Yamamoto, Jonathan Tay Yu Liang (2023). [Active Robot Vision for Distant Object Change Detection: A Lightweight Training Simulator Inspired by Multi-Armed Bandits](https://arxiv.org/abs/2307.14105).
[12] Yuqun Wu et al. (2025). [SceneDiff: A Benchmark and Method for Multiview Object Change Detection](https://arxiv.org/abs/2512.16908).
[13] Tianheng Cheng et al. (2024). [YOLO-World: Real-Time Open-Vocabulary Object Detection](https://arxiv.org/abs/2401.17270). CVPR 2024.
[14] Nikhila Ravi et al. (2024). [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714).
