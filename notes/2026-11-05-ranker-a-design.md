# Ranker A and A-nocue: design (pinned 2026-11-05, before A is fitted on any data)

SIMULATION, development data. Pinned by the commit that adds this note, before A or A-nocue is trained or run on any
rendered evidence (researcher's decision of 2026-11-05, item 3: "implement one A plus the declared A-nocue, using the
original training-label definition ... Preserve requested-view-only access and arrangement-isolated calibration,
training and model selection"). The 49 remaining tabletop development arrangements stay sealed until this design is
pinned (Amendment 2); after that they are development data.

Declared in SIM_PREREGISTRATION §4: "A regularised logistic regression on observed evidence and candidate geometry, never
on unrequested images"; "same λ (= 0) and candidate set as G"; "Training: grouped (by arrangement) CV on dev only, with
geometric diagnostic labels (§3) as training targets"; "Freeze: features and regularisation grid are fixed in
FREEZE.md"; A-nocue = "A without the occlusion cue". This note fixes everything those lines leave open. It adds one
feature family, the left/right descriptor (`notes/2026-11-05-side-descriptor.md`), and nothing else. The pinned M16
check of that descriptor ran before this note (its result is appended to the descriptor note); as pinned there, it
does not change this design.

## Model

- One L2-regularised logistic regression. Features are standardised with the weighted mean and standard deviation of
  the training rows (a feature with zero deviation is centred only). The intercept is not penalised.
- Objective: sum over training rows of w_i * logloss_i + ||beta||^2 / (2 C), the weights summing to the number of
  training arrangements. Fit by Newton-Raphson from zero until the largest step is below 1e-10 (at most 100 steps;
  not converging is an error). Deterministic.
- Regularisation grid C in {0.01, 0.1, 1, 10}.

## Training label and rows (training arrangements only)

- Label: the original geometric diagnostic label of the candidate stop in that episode (`view_labels.json`
  `diagnostic`, SIM_PREREGISTRATION §3: present, visible fraction >= 0.90; removed, the occluder covers < 10 % of
  the target's region). Simulator truth enters only as this training label.
- Histories: for every training episode (every condition and lighting) and every ordered sequence H of 0, 1 or 2
  distinct non-start stops such that the shared stopping rule, with the fold's thresholds and k = 2, returns CONTINUE
  after the start view P3 followed by H. These are all the states, uncapped, at which a policy with b = 3 makes its
  first, second or third choice. One row per stop not yet visited.
- Weights: each training arrangement carries 1 in total, split equally over its episodes that have rows, and within
  an episode equally over its rows.

## Features (from the requested views only, and the candidate's geometry)

Geometry is measured in units of the arc's half-span S = the largest |azimuth - start azimuth| over the stops (45 deg
on the shelf, 60 deg on the tabletop, from the positions file); D(a, b) = |az(a) - az(b)| / S. The candidate's own
evidence and every unrequested view are never read.

| # | Name | Definition | In A-nocue |
|---|---|---|---|
| 1 | az_from_start | D(c, start) | yes |
| 2 | sep_visited | min over visited v of D(c, v) | yes |
| 3 | sep_occluded | min over visited v with obstruction > tau_occ of D(c, v); sep_visited if none (G's reference set) | no |
| 4 | obstruction_start | the start view's obstruction | no |
| 5 | side_toward | mean over visited v with a descriptor of d_v * sign(az(c) - az(v)); 0 if none | no |
| 6 | side_missing | 1 if no visited view has a descriptor, else 0 | no |
| 7 | match_max | the largest match score over visited views | yes |
| 8 | quality_min | the smallest quality over visited views | yes |

A-nocue drops 3-6: the side descriptor and every other obstruction-derived feature. tau_occ is the fold's threshold.

## Model selection (inside each training set; arrangement-isolated)

Five inner folds of the training arrangements (sorted by id, fold = position mod 5). For each C: fit on four folds,
score the weighted mean log-loss of the label on the fifth (rows built with the outer fold's thresholds), average over
the five. The lowest average wins; ties within 1e-12 go to the smaller C. Then refit on every training row. The
criterion is the training label's log-loss, never a policy outcome.

## Selection at decision time

Candidates: the affordable unvisited stops (G's candidate set; no lookahead). Score: the fitted probability. Argmax;
scores within EPS tie, then shorter travel, then lower stop index (G's tie-break). lambda = 0. The shared stopping rule,
budgets and caps are unchanged: A runs at b = 3 uncapped (primary) and at every declared budget and cap, like G.

## Isolation and access

- Each outer fold of the held-out policy check trains A afresh in its own process (the existing FoldTrainable hook) on
  dev.subset(training arrangements) with that fold's thresholds; the held-out arrangement's evidence, descriptors,
  labels and truth are not reachable from the training subset.
- At decision time the policy receives only the observations of requested views (match, quality, obstruction and the
  descriptor; `verify/side.py`), the candidates, the positions and the current stop.

## Frozen A

Trained once on every development arrangement of both environments with the final frozen thresholds, by the same
procedure. Its feature list, standardisation, chosen C and coefficients go to a JSON model file recorded in FREEZE.md.

## Reported, and not done

Reported: per fold the chosen C, the inner log-loss of every C, the coefficients and the label base rate; the held-out
check rows of A, A-nocue and Dir beside every other policy; A - G (primary, b = 3, uncapped), A - G-lookahead along the
cap grid (secondary), A - A-nocue and A - Dir (explanatory). Not done: no other feature, model family or label; no
interaction terms; no lookahead; no tuning on policy outcomes; no promotion of any cap or subgroup.

## Implementation notes (2026-11-05, after the checkpoint review, before any fit on rendered data)

- Inner folds: the first implementation (5ff3b89) formed them from the training arrangements that have rows, so one
  arrangement without rows would shift the fold assignment of every later one. Fixed to the rule above (all training
  arrangements, sorted, position mod 5) before A was fitted on any rendered data. A fold with no rows is skipped and
  counted (`inner_folds_skipped`).
- An arrangement whose every episode stops at its first view has no row and so no weight: the weights sum to the number
  of training arrangements with rows. Both counts are reported (`n_training_arrangements`, `n_arrangements_with_rows`).
- Rows are built at b = 3 (histories of 0-2 additional views). At larger budgets of the declared budget curves A scores
  histories longer than any it was trained on.
- Per-fold reports are in frozen_candidate.json (`held_out_check.folds[].trained_reports`) and summarised in the
  selection report (C chosen per fold, coefficient means and ranges).
- A frozen model file records its features, thresholds and the candidate's sha256; verify.run refuses a model whose
  name, features, thresholds or candidate do not match.
