# Literature and SOTA Survey

**Project:** Occlusion Aware Active Verification for Robot Scene Memory  
**Prepared for:** CMPE 249 proposal  
**Survey date:** September 9, 2026

## Scope and evidence status

Fourteen entries, numbered to match the proposal. Ten were first released in 2024 to 2026 and satisfy the course's recent-literature requirement; four earlier works ([1], [3], [4], [11]) provide foundational mapping, persistence, and active-selection context. Summaries reflect the authors' claims as stated in the primary sources; nothing has been reproduced yet. This is a focused survey, not proof that the proposed gap is unoccupied.

| Ref | Paper or model | Year | Role in this proposal |
| --- | --- | --- | --- |
| [1] | [POCD](https://arxiv.org/abs/2205.01202) | 2022 | Probabilistic object change and mapping; baseline context |
| [2] | [Khronos](https://arxiv.org/abs/2402.13817) | 2024 | Spatio-temporal mapping context |
| [3] | [Bore et al.](https://arxiv.org/abs/1712.08409) | 2019 | Movable-object tracking with global-motion hypotheses |
| [4] | [Rosen et al.](https://research.google/pubs/towards-lifelong-feature-based-mapping-in-semi-static-environments/) | 2016 | Persistence filter; missed-detection likelihoods |
| [5] | [PreSIST](https://arxiv.org/abs/2607.04057) | 2026 | Scene-conditioned persistence priors |
| [6] | [LT-Mem](https://arxiv.org/abs/2608.19059) | 2026 | Hold and multi-hypothesis memory updates |
| [7] | [DynaMem](https://arxiv.org/abs/2411.04999) | 2024 | Dynamic memory, ray casting, and active exploration |
| [8] | [RoboStream](https://arxiv.org/abs/2603.12939) | 2026 | Object permanence inside VLM-driven manipulation |
| [9] | [Lost & Found](https://arxiv.org/abs/2411.19162) | 2025 | Interaction-driven scene graph updates |
| [10] | [Bogenberger et al.](https://arxiv.org/abs/2509.19851) | 2026 | Closest active map-maintenance work |
| [11] | [Terashima et al.](https://arxiv.org/abs/2307.14105) | 2023 | Closest active-selection formulation |
| [12] | [SceneDiff](https://arxiv.org/abs/2512.16908) | 2025 | Multiview change classification with the sequence given |
| [13] | [YOLO-World](https://arxiv.org/abs/2401.17270) | 2024 | Perception front end |
| [14] | [SAM 2](https://arxiv.org/abs/2408.00714) | 2024 | Optional mask refinement |

## Closest work and candidate distinction

Existing work spans probabilistic geometric change detection [1, 2], tracking under unobserved motion [3], persistence estimates [4, 5], and memory updates for changing scenes [6, 7, 8, 9]. Active methods include DynaMem exploration driven by recency and semantic relevance [7], stationarity-informed exploration [10], and distant-object inspection journeys [11]. SceneDiff reasons about changes using already acquired multiview data [12]. These methods establish substantial overlap with the proposed decision.

The candidate distinction is the controlled comparison of learned and geometric view selection under a shared stopping rule, for one object and a finite set of RGB views. This is a proposed experimental scope, not an established unoccupied gap. S is a simplified no-movement reference; G is the primary selection baseline. The early literature gate will check whether an equivalent comparison already exists, especially in [7, 10, 11].

## [1] POCD: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes

**Authors:** Jingxing Qian, Veronica Chatrath, Jun Yang, James Servos, Angela P. Schoellig, Steven L. Waslander  
**Date and status:** May 2022; RSS 2022  
**Primary source:** [arXiv 2205.01202](https://arxiv.org/abs/2205.01202)

**Summary:** Each object carries a probabilistic state made of a stationarity score and a TSDF change measure, updated by a Bayesian rule that fuses geometric and semantic evidence. The map stays consistent across repeated traversals of semi-static scenes. Evaluated on a released warehouse dataset (TorWIC) and on ToyCar; reconstruction quality beats the compared methods.

**Boundary relevant to this project:** POCD centers on probabilistic change estimation and volumetric map updates from incoming observations. Its TSDF representation and Bayesian update are substantially richer than the proposed 2D heuristic.

**Planned use:** Background for probabilistic object-change reasoning. S is a simplified project baseline, not an implementation of POCD; results against S cannot establish superiority over POCD.

**Question to answer while reading:** How does the update avoid counting an occluded observation as evidence of change, and what does that rule need from geometry that monocular RGB cannot provide?

## [2] Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments

**Authors:** Lukas Schmid, Marcus Abate, Yun Chang, Luca Carlone  
**Date and status:** February 2024; revised May 2024; RSS 2024  
**Primary source:** [arXiv 2402.13817](https://arxiv.org/abs/2402.13817)

**Summary:** Separates fast tracking of short-term dynamics from slower reasoning about long-term change inside one dense spatio-temporal map. Evaluated in simulation and on a physical robot.

**Boundary relevant to this project:** Rebuilding its mapping stack is out of scope; the project fixes camera positions and reference captures instead of running SLAM.

**Planned use:** Motivation for scene-memory maintenance and the need to distinguish perception errors from scene changes. The full mapping method is not reproduced.

**Question to answer while reading:** Which errors come from localization, and which from deciding that the environment changed?

## [3] Detection and Tracking of General Movable Objects in Large Three-Dimensional Maps

**Authors:** Nils Bore, Johan Ekekrantz, Patric Jensfelt, John Folkesson  
**Date and status:** December 2017 preprint; IEEE Transactions on Robotics 35(1), 2019  
**Primary source:** [arXiv 1712.08409](https://arxiv.org/abs/1712.08409)

**Summary:** A mobile robot in a large environment sees only a subset of objects at a time, so objects move while unobserved. The filter models local motion analytically (Kalman) and rare global jumps by sampling over associations, and outperforms prior methods on autonomously collected point-cloud data.

**Boundary relevant to this project:** Models where an object may have gone; does not decide which view to acquire to confirm that it left.

**Planned use:** Justifies "removed from the monitored area" as the honest claim: absence here is not absence everywhere.

**Question to answer while reading:** Which observation moves probability mass from "still here" to "jumped elsewhere"?

## [4] Towards Lifelong Feature-Based Mapping in Semi-Static Environments

**Authors:** David M. Rosen, Julian Mason, John J. Leonard  
**Date and status:** ICRA 2016  
**Primary source:** [Google Research](https://research.google/pubs/towards-lifelong-feature-based-mapping-in-semi-static-environments/)

**Summary:** A generative survival model for semi-static features admits a recursive Bayesian estimator, the persistence filter, that maintains an exact belief that each feature still exists given detections and missed detections.

**Boundary relevant to this project:** The prior and the detection likelihoods are inputs; observation choice is not part of the model.

**Planned use:** Motivation for accumulating evidence across misses. The proposed threshold on distinct clear misses is a heuristic and does not reproduce this Bayesian persistence filter.

**Question to answer while reading:** How is a missed detection weighted when the feature may not have been observable?

## [5] PreSIST: Vision-Language-Informed Object Persistence Prediction in Open-World Scenes

**Authors:** Amanda Adkins, Tarunvidyut Ravisankar, Joydeep Biswas  
**Date and status:** July 2026 preprint  
**Primary source:** [arXiv 2607.04057](https://arxiv.org/abs/2607.04057)

**Summary:** Predicts whether an observed object will remain at its last seen pose at future times by estimating instance-level persistence priors from object properties and scene context, then integrating them with a persistence filter as observations arrive. A VLM variant and a vision-only distilled variant are evaluated on a new in-the-wild persistence dataset.

**Boundary relevant to this project:** Persistence at a previous pose and absence from a monitored region are different targets; a prior is not visual proof of removal.

**Planned use:** Motivates keeping uncertainty after a miss. Development-set thresholds replace VLM-derived priors in the initial study.

**Question to answer while reading:** Which part of the estimate is prior expectation and which part is sensor evidence?

## [6] LT-Mem: Volatility-Aware Spatio-Temporal Memory for Lifelong Scene Understanding

**Authors:** Yumin Lee, Hyoseok Ju, Giseop Kim  
**Date and status:** August 2026; arXiv record reports IROS 2026 acceptance  
**Primary source:** [arXiv 2608.19059](https://arxiv.org/abs/2608.19059)

**Summary:** Multi-session SLAM provides spatially aligned per-object observations; a volatility-aware policy chooses among overwrite, hold, and multi-hypothesis updates; Live, Delta, and Meta memories keep current state, changes, and history. Evaluated with the LT-VQA suite.

**Boundary relevant to this project:** Close prior work on preserving uncertainty and history. An unknown state alone is not a contribution.

**Planned use:** Keep a small object record with its last confirmed observation when evidence is inconclusive. Multi-session SLAM and temporal QA are out of scope.

**Question to answer while reading:** When does holding a record preserve information, and when does it create stale memory?

## [7] DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation

**Authors:** Peiqi Liu, Zhanqiu Guo, Mohit Warke, Soumith Chintala, Chris Paxton, Nur Muhammad Mahi Shafiullah, Lerrel Pinto  
**Date and status:** November 2024 preprint; revised May 2025  
**Primary source:** [arXiv 2411.04999](https://arxiv.org/abs/2411.04999)

**Summary:** A dynamic 3D memory supports open-vocabulary localization, removes outdated voxels using RGB-D ray casting, and can abstain when a queried object is not found. Exploration uses observation recency and semantic relevance, with rescanning and replanning during navigation. Its reported lab pick-and-drop success rate on changing objects is 70%, compared with 30% for OK-Robot.

**Boundary relevant to this project:** DynaMem already couples memory updates to active exploration. The proposed study must be distinguished through its finite RGB view set, common stopping rule, and matched-budget selector comparison; adding exploration or abstention alone is not new.

**Planned use:** Close prior art for visibility, abstention, and exploration. Read Sections 3.2-3.4 during the early literature gate; the full system is not reproduced.

**Question to answer while reading:** What evidence does DynaMem require before an object leaves memory, and what happens when the object is merely occluded?

## [8] RoboStream: Weaving Spatio-Temporal Reasoning with Memory in Vision-Language Models for Robotics

**Authors:** Yuzhi Huang et al.  
**Date and status:** March 2026; revised July 2026; arXiv record reports ECCV 2026 acceptance  
**Primary source:** [arXiv 2603.12939](https://arxiv.org/abs/2603.12939)

**Summary:** Maintains geometric grounding and records action-related scene transitions to support longer manipulation sequences; persistent representations avoid forgetting temporarily occluded objects. Evaluated in simulated manipulation and physical block building.

**Boundary relevant to this project:** Object permanence in robot memory is an established direction and is not claimed as new here.

**Planned use:** Motivates the downstream cost of a premature deletion.

**Question to answer while reading:** How does one wrong perception update propagate into later actions?

## [9] Lost & Found: Tracking Changes from Egocentric Observations in 3D Dynamic Scene Graphs

**Authors:** Tjark Behrens, Rene Zurbruegg, Marc Pollefeys, Zuria Bauer, Hermann Blum  
**Date and status:** November 2024 preprint; IEEE Robotics and Automation Letters 2025  
**Primary source:** [arXiv 2411.19162](https://arxiv.org/abs/2411.19162)

**Summary:** From egocentric recordings with hand and camera pose estimates, tracks the 6DoF pose of an object during a detected interaction interval and applies the change to a transformable scene graph. Reported 34% and 56% lower translational and orientational error than the second-best tracker, plus robot demonstrations including retrieval of an object hidden in a drawer.

**Boundary relevant to this project:** Changes are observed as they happen through interactions. This project deals with changes that happened unobserved between visits.

**Planned use:** Context on interaction-driven updates; not reproduced.

**Question to answer while reading:** What happens to the scene graph when a change occurs with no observed interaction?

## [10] Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in Real-World Semi-Static Environments

**Authors:** Benjamin Bogenberger, Oliver Harrison, Orrin Dahanaggamaarachchi, Lukas Brunke, Jingxing Qian, Siqi Zhou, Angela P. Schoellig  
**Date and status:** September 2025 preprint; IEEE Robotics and Automation Letters 11(3), March 2026  
**Primary source:** [arXiv 2509.19851](https://arxiv.org/abs/2509.19851)

**Summary:** An open-vocabulary semantic exploration system for semi-static environments that keeps a probabilistic stationarity model per object instance, tracks semi-static changes, and actively revisits areas not seen for a long time; LLM reasoning prioritizes contextually relevant areas for object-goal navigation. Reports detecting 95% of map changes on average and more than 29% higher efficiency than random and patrol strategies in real-world runs.

**Boundary relevant to this project:** This is close active prior work: object-level stationarity and task relevance produce a spatial exploration priority map used to select waypoints. Its scope is broader than a fixed-view verification episode, so the intended experimental distinction needs checking against its full planning and update loop.

**Planned use:** Primary prior art to position against; its revisit policy and stationarity update are on the week-1 reading list, and the week-2 gate checks that the selection-and-stopping comparison is not already inside it.

**Question to answer while reading:** After a revisit fails to observe an object, what does the system do next, and what evidence makes it conclude the object is gone?

## [11] Active Robot Vision for Distant Object Change Detection: A Lightweight Training Simulator Inspired by Multi-Armed Bandits

**Authors:** Kouki Terashima, Kanji Tanaka, Ryogo Yamamoto, Jonathan Tay Yu Liang  
**Date and status:** July 2023 preprint  
**Primary source:** [arXiv 2307.14105](https://arxiv.org/abs/2307.14105)

**Summary:** Frames ground-view change detection as an active problem: which journey to a distantly detected object should the robot take next, given that every journey has a sense, plan, and act cost. Contributes a lightweight simulator built from one real journey per object and a bandit-style map-based planner on top of a mapless navigator.

**Boundary relevant to this project:** Its main formulation selects inspection journeys to objects. Budgeted active change detection is therefore established prior art; the finite-view selector and stopping comparison requires a narrower claim.

**Planned use:** Closest active-selection formulation; the budget and the bandit framing are the natural comparison for the ranker.

**Question to answer while reading:** How does the planner decide whether another journey is worth its cost?

## [12] SceneDiff: A Benchmark and Method for Multiview Object Change Detection

**Authors:** Yuqun Wu, Chih-hao Lin, Henry Che, Aditi Tiwari, Chuhang Zou, Shenlong Wang, Derek Hoiem  
**Date and status:** December 2025; revised March 2026  
**Primary source:** [arXiv 2512.16908](https://arxiv.org/abs/2512.16908)

**Summary:** Detects added, removed, and moved objects between captures by combining estimated geometry, segmentation, and appearance features to separate real change from viewpoint and occlusion effects; 350 video pairs in the benchmark.

**Boundary relevant to this project:** Takes the multi-view sequence as given. Substantial overlapping work on the classification step, not on acquisition.

**Planned use:** Methodological reference for the visibility reasoning and a source of extra examples; a reduced run on selected frames is a stretch comparison.

**Question to answer while reading:** What evidence makes absence verifiable, and which observations does the method need before deciding?

## [13] YOLO-World: Real-Time Open-Vocabulary Object Detection

**Authors:** Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan  
**Date and status:** January 2024; CVPR 2024  
**Primary source:** [arXiv 2401.17270](https://arxiv.org/abs/2401.17270)

**Summary:** Text-conditioned open-vocabulary detection in the YOLO family via vision-language pretraining.

**Boundary relevant to this project:** A category detection is not instance identity, and detector confidence is not a calibrated existence probability.

**Planned use:** Frozen front end for every policy; reference-crop matching adds identity.

**Question to answer while reading:** What does a low score mean when the target is hidden, blurred, or unfamiliar?

## [14] SAM 2: Segment Anything in Images and Videos

**Authors:** Nikhila Ravi et al.  
**Date and status:** August 2024; revised October 2024  
**Primary source:** [arXiv 2408.00714](https://arxiv.org/abs/2408.00714)

**Summary:** Prompted masks for images and video with streaming memory across frames.

**Boundary relevant to this project:** A mask does not establish identity, presence, or depth ordering.

**Planned use:** Only if masks improve the occlusion cue over boxes in the pilot. Reference prompts are allowed; manual target prompts on held-out queries are not.

**Question to answer while reading:** Which prompts would a deployed robot actually have, and which would leak the answer?

## Additional context, not counted

[GOAT-Bench](https://arxiv.org/abs/2404.06609) (Khanna et al., 2024) evaluates lifelong multi-modal navigation and the role of scene memory. It is broader context for where a verification module would sit, not a benchmark for this decision.

## Reading order

1. [10], [7], and [1]: active exploration and geometric change updates. Write down what each does after a revisit returns a miss, and whether the proposed comparison is already evaluated.
2. [11] and [12]: the budgeted-selection framing and the classification step.
3. [4] and [5]: what a missed detection is worth, with and without a prior.
4. [13] and [14]: the components and what their outputs do not establish.
5. [2], [3], [6], [8], [9]: mapping and memory context; introductions, system diagrams, limitations.

For each paper: inputs, outputs, central assumption, strongest evidence, closest overlap with this proposal, and one claim to check in code or an experiment.
