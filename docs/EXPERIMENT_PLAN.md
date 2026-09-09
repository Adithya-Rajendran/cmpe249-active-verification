# Experiment Plan

## Unit of evaluation

One episode is one target-state decision in a static workspace. It contains a target reference, known camera positions, a starting view, and a hidden collection of query images. The environment returns an image only after the policy selects its view. The policy must return present, removed, or unknown within the budget. The target-state ground truth is present or removed; unknown is the policy's abstention, not a ground-truth class.

## Planned collection and split

- Pilot: 24-30 episodes, kept separate from the final test set.
- Main study: 30 arrangements x 3 target conditions x 2 lighting conditions = 180 episodes.
- Conditions: visible target, present but occluded target, and removed target.
- Capture five query views per episode, plus the required reference views.
- Split arrangements before tuning: 10 development arrangements and 20 test arrangements.
- Capture matched occluder layouts in occluded-present and removed pairs. Otherwise the occluder itself could reveal the label.
- Keep all lighting conditions and target states for a physical arrangement in the same split.
- Predeclare a small subset of matched present and removed episodes whose available views cannot distinguish the states. Unknown is the intended response in both cases. Report the resulting maximum achievable coverage alongside the target metrics.
- The initial task does not claim unseen-object generalization or detection of within-area relocation.

## Reference and label policy

Reference crops and expected target regions are collected before changing the scene and are available to every method. Human annotations of query visibility, identity, and physical target state are used for development or scoring only. Do not manually prompt the target location in test query frames. Calibrate camera geometry with the same protocol for all conditions; markers must not encode target presence.

## Initial algorithm specification

The front end produces candidate boxes, an appearance-match score, an image-quality score, and an approximate overlap cue from detected non-target objects. Optional masks refine the overlap. Raw image difference cannot by itself identify occlusion because removal and lighting changes also alter pixels. Keep evidence over distinct visited positions. Detector scores and the occlusion cue are not calibrated existence or visibility probabilities.

For development labels, a view is diagnostic when an annotator can verify the target instance, or can verify that its expected region is unobstructed in a removed-target episode. Ambiguous views receive a non-diagnostic label. Store visibility annotations separately from policy inputs and document disagreements or uncertain labels.

Train a regularized logistic regression on development episodes to predict whether a candidate viewpoint will provide diagnostic evidence. Features use only observed evidence and image quality, known reference visibility, and candidate geometry relative to observed views. Development labels may use the annotated candidate outcome. Test features may not use the candidate query image before it is requested. Select the encoder, features, regularization, and thresholds using validation folds grouped by development arrangement; generated history-view pairs do not count as independent arrangements.

Policies. S uses the initial view only: present on a strong match; removed on an estimated-clear, adequately exposed and focused region with no target match; unknown otherwise. It is a simplified project reference, not a reproduction of POCD or Khronos. F follows a fixed left-to-right order; R samples an unvisited position. G maximizes the minimum angular separation from positions flagged as occluded, measured about the reference workspace center; if none are flagged, use all visited positions. Its score subtracts a normalized travel penalty. A uses predicted informativeness minus the same penalty. Both utility scores and travel costs are normalized to a common scale using development data. The angular-diversity score is a proxy, not a prediction of clear line of sight. Ties favor shorter travel, then a fixed position index.

Stopping. Every multi-view policy uses the same budget-limited evidence rule: present on one strong instance match; removed only when clear, unmatched evidence from at least two distinct positions reaches a threshold; otherwise acquire another affordable position or return unknown. A position counts as a clear miss only if image quality and obstruction checks pass. The initial implementation counts qualifying clear misses, with the required count (at least two) selected on development data. It does not multiply detector scores or assume independent views. Freeze the rule and travel penalty before final testing.

## Comparison matrix

| Policy | View choice | Terminal decision logic |
| --- | --- | --- |
| S  Passive single view | Initial position only | Visibility rule; no movement |
| F  Fixed order | Left to right | Shared sequential stopping |
| R  Random order | Random unvisited position; repeated seeds | Shared sequential stopping |
| G  Geometric parallax | Angular diversity minus travel penalty | Shared sequential stopping |
| A  Learned ranker (proposed) | Logistic informativeness minus travel penalty | Shared sequential stopping |
| A without occlusion cue | A retrained without the cue or cue-derived features | Same stopping rule; isolates the cue in ranking |
| First-clear-miss removal | Same selector as F, R, G, or A | Removed on the first qualifying clear miss; isolates conservative stopping |
| Full sweep reference | All five positions | Higher-observation reference, labeled separately |

Use common initial positions, query access, observation caps, and cumulative travel caps. At each step, filter to unvisited positions affordable under the remaining budget; F and R choose from that set too. The initial observation counts toward total views but not additional views. Report results for 0-4 additional views; the primary point is 3. S always uses just the initial view and is plotted at its actual effort. Full sweep uses all five positions as a separate higher-budget reference, without an accuracy-upper-bound claim. Choose camera spacing and travel caps from the pilot and freeze them before test evaluation. In replay, travel is the path cost between calibrated positions, not measured energy or execution time. Average random seeds within each episode before paired comparisons; seeds add no physical samples.

## Metrics and reporting

False removal rate = number of removed decisions on present episodes / number of present episodes.

Removal recall = number of correct removed decisions / number of removed episodes.

Coverage = number of non-unknown decisions / total episodes.

Resolved accuracy = number of correct non-unknown decisions / number of non-unknown decisions. Report undefined values explicitly if a method resolves nothing.

Report views, cumulative commanded travel, confusion counts, and all metrics separately for visible, occluded, and removed conditions. The initial joint targets at the primary budget are false removal rate <= 10%, removal recall >= 80%, and coverage >= 80%. These are goals, not power calculations or proven guarantees. Compare complete error/coverage/effort curves as well as the primary point.

The primary selection comparison is A versus G. Use paired arrangement-level bootstrap intervals for differences between policies. Keep all related episodes together when resampling. Report the number of arrangements and episodes alongside every result. If intervals are too wide to distinguish policies, report the comparison as inconclusive. A nonsignificant result does not establish equivalence. Any optional synthetic extension is reported separately and does not increase the physical sample count.

## Logs and reproducibility

Each episode log should include arrangement ID, split, target ID, ground truth (scoring only), model/checkpoint identifier, configuration version, requested-view sequence, revealed evidence, decision, and travel. Keep raw captures, annotations, and manifests versioned separately from the small code repository. Record all exclusions and retain failed cases. The final repository should document how to reproduce the comparison; it currently contains proposal materials only.
