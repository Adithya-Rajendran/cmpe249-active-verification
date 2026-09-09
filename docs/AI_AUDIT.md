# AI Novelty and Feasibility Audit

**Project:** Occlusion Aware Active Verification for Robot Scene Memory  
**Prepared by:** ChatGPT / Codex as AI assistance; overlap table and the geometry risk extended by Claude (Anthropic)  
**Date:** September 9, 2026  
**Basis:** The proposed scope, the student's hardware inventory, and the primary references linked in the literature survey. No implementation or hardware measurements have been completed.

## Overall evaluation

This is a plausible first-year MS course project if it stays within a fixed tabletop workspace and a finite set of camera views. Its strongest feature is a measurable robot decision: acquiring more evidence before updating object memory. Its weakest point is novelty. Active perception, change detection, object permanence, and uncertainty-preserving memory are established areas. The proposal currently defines a candidate extension, not a verified novel method or a guaranteed publication.

## Closest overlapping work

| Prior work | Existing overlap | Consequence for the proposal |
| --- | --- | --- |
| [POCD](https://arxiv.org/abs/2205.01202), [Khronos](https://arxiv.org/abs/2402.13817) | Geometric and probabilistic reasoning about object changes | S is a simplified project reference, not a reproduction of these mapping systems. Beating S cannot establish superiority over the published methods. |
| [Bogenberger et al.](https://arxiv.org/abs/2509.19851) | Active revisiting of stale map regions driven by a per-instance stationarity score | Closest active work. The proposal must show that per-object, per-episode view selection with a stopping rule is not already inside its revisit loop. |
| [Terashima et al.](https://arxiv.org/abs/2307.14105) | Choosing which object journey to make next under a cost | Compare the proposed finite-view policy against its bandit framing; do not present budgeted selection as new. |
| [SceneDiff](https://arxiv.org/abs/2512.16908) | Multiview object changes and visibility reasoning | Do not claim that distinguishing occlusion from removal is itself new. |
| [PreSIST](https://arxiv.org/abs/2607.04057), [Rosen et al.](https://research.google/pubs/towards-lifelong-feature-based-mapping-in-semi-static-environments/) | Persistence beliefs updated from observations and missed detections | A persistence threshold alone is a weak contribution. |
| [LT-Mem](https://arxiv.org/abs/2608.19059) | Holding or retaining alternative object histories under changing evidence | An unknown state or a conservative update is insufficient by itself. |
| [DynaMem](https://arxiv.org/abs/2411.04999) | RGB-D ray casting for memory removal, abstention in localization, and active exploration using recency and semantic relevance | Treat this as active prior work. The proposed finite-view comparison needs a narrower distinction than adding exploration or abstention. |

The literature search does not establish that no prior work has evaluated an equivalent selection and stopping rule. The student should inspect these methods and their references before using strong novelty language in the final report.

## Defensible candidate contribution

Test whether a small learned view-ranking model plus conservative stopping improves the error-versus-effort trade-off over the passive visibility rule, a no-learning parallax selector, and fixed and random acquisition, while holding the perception front end constant. The result should identify when additional sensing is worth its cost and where estimated visibility fails. A controlled replay protocol and a documented dataset can support the contribution, but a new dataset alone must have a clear purpose beyond making the demonstration easier.

Broad claims that this is an unsaturated field are not supported. The narrow course-scale study may still be worthwhile. Deployment on a Jetson is an implementation choice, not the scientific contribution; device throughput should be a secondary feasibility measurement only.

## Main technical and experimental risks

| Risk | Why it matters | Required mitigation |
| --- | --- | --- |
| Geometry trivializes the task | Reliable depth and a sufficiently complete scene model can simplify visibility checking; geometric view selection may already be sufficient. | Keep G as the primary selection baseline. It measures angular diversity without assuming visibility in unvisited views. Report differences with intervals; a wide interval is inconclusive. |
| Non-observability | An object hidden from every available view cannot reliably be distinguished from removal using those views alone. | Retain unknown; report coverage and failures on fully hidden cases. |
| Occlusion cue errors | Removal itself creates image differences. Even overlap with detected non-target objects does not establish depth order or true visibility. | Use detected non-target overlap rather than raw image difference alone; validate against visibility labels, ablate the cue, and audit false removals. |
| Weak data association | Same-category objects may be confused across views. | Begin with distinct objects and reference crops; document this limitation before adding duplicates. |
| Test leakage | Reading unvisited images would make the active policy unrealistically informed. | Enforce a view-query interface; keep target-state labels and future-view features outside policy inputs. |
| Misleading uncertainty handling | Always returning unknown or present can produce a low false-removal rate. | Evaluate false removals, removal recall, coverage, and effort jointly. |
| Unfair comparisons | Better detectors, more views, or extra movement could explain improvements. | Freeze perception and initial pose. Compare F, R, G, and A under shared stopping and budget caps; report S at its zero-movement effort. Test stopping separately. |
| Small or dependent samples | The 120 test episodes contain only 20 independent arrangements; frames, lighting variants, and random seeds are dependent observations. | Split and resample by arrangement. Use grouped development validation, report counts and intervals, and avoid treating a nonsignificant result as equivalence. |
| Mechanism construction dominates | Building a rover and navigation system could consume the semester. | Complete recorded-view evaluation first; make a simple translation stage an extension. |
| Software compatibility | Model availability does not establish correct execution on every owned device. | Run the GB10 pilot first; treat Jetson and AI HAT+ deployment as optional until verified. |

## Feasibility judgment

The student owns and maintains an ASUS Ascent GB10, a Jetson Orin Nano 8 GB, a Raspberry Pi 5 with AI HAT+, and a Raspberry Pi camera. This supports local development and controlled data capture without depending on a shared laboratory reservation. The proposal avoids training a foundation model; a small predictor and frozen perception models keep learning requirements modest. Exact memory use, throughput, camera compatibility, and integration effort still require a pilot. Ownership does not substitute for those checks.

The planned 180 episodes and roughly 900 query images are feasible as a bounded collection target, but collection time is not yet measured. The pilot should record preparation and annotation time per arrangement before committing to all 30. The core result must remain possible without a complete motorized robot.

## Two-week decision gate

Continue if the front end works on clearly visible pilot targets, some camera changes resolve ambiguous cases, and a meaningful difference from the closest existing method can be stated. Narrow the task if geometry, object identity, or data collection dominates. Change the method or topic with the instructor if the observation set cannot support the question or the planned extension is already covered. This early gate protects the semester schedule; it is not a claim that a change has already been approved.

## Interpreting eventual results

A positive result requires evidence that the proposed selection policy improves useful decisions at comparable effort, with uncertainty reported. A negative result can still be valuable if it identifies why adaptive inspection fails or when a simpler policy is sufficient. Wide intervals or unreliable labels make a result inconclusive rather than proving that active inspection has no value. The study cannot establish general performance across industrial robots, unseen environments, or all object categories.

## Recommendation

Proceed with the bounded pilot and the literature comparison. Maintain the proposed topic as provisional until the early gate is passed. Keep the final claims tied to the tested decision policy and task; do not sell the project as a new general-purpose robotics system.
