# Left/right obstruction descriptor, direction-rule control and the M16 check (pinned 2026-11-05)

SIMULATION, development data only. Pinned by the commit that adds this note and `perception/side_descriptor.py`,
before the descriptor was computed on any rendered view (researcher's decision of 2026-11-05, item 3: "Fix its definition
before examining separation; report failures without starting another feature or registration search").

## Why

On shelf dev, under the two-step travel cap, the any-order ceiling of removals (25 / 60 in L1, 24 / 60 in L2) exceeds the
best one-side route (15 and 13), and 18 of those ceiling removals are the M16 arrangements, each resolvable only on the
side away from its designed board offset (`runs/2026-11-05-headroom-check/side_audit.txt`; a post-hoc development
diagnostic, Amendment 2). v0.3's obstruction is one number per view and says nothing about the side, so no policy that
sees only the scalars can choose the side. This descriptor is the one fixed spatial feature added for A; it is not a
measured occlusion mask and not proof of visibility.

## Definition

`perception/side_descriptor.py` (DESCRIPTOR_VERSION side-0.1-dev; its docstring is the definition):
- C0 registration unchanged (Amendment 2). A refused view has NO descriptor (missing), never 0.
- v0.3's background-obstruction map, its own steps and parameters, over the context window grow(box, 1.0): changed
  against the gain-corrected reference AND not the expected background; invalid pixels are never obstructed.
- Split at the box centre column; u_left and u_right are the obstructed shares of the VALID pixels of each part.
- Missing when either part has fewer than half of its pixels valid.
- d = u_left - u_right. The in-box share of the same map must equal v0.3's background_obstruction (checked per view).

The choice of window is not tuned: it is v0.2.1's context window, and the generator parks every occluder outside it at
every stop (`sim/arrange.py` park_occluder, park_occluder_table), so a parked board's disappearance does not enter it.

## Sign, from geometry only

- Image +x points toward increasing azimuth at every stop of both environments: each other lens projects to +x exactly
  when its azimuth is larger (sim/rig.py cameras of positions_shelf_A.json and positions_tabletop_A.json, checked
  2026-11-05 before any descriptor value existed).
- An occluding surface in front of the target hides the stops toward its own displaced side (the sight line pivots
  about the target and, at the board's depth, moves with the camera); the generator's M16 side rule states the same
  ("R = board centre displaced toward +x ... R hides P3 and P4").
- So d > 0 (more obstruction left of the target) points to larger azimuth (P4, P5 from P3), d < 0 to smaller (P2, P1).
  For M16: side L predicts d > 0 at P3, side R predicts d < 0.

## Direction-rule control Dir (explanatory, post-hoc development origin)

Proposed on 2026-11-05 AFTER the saved-evidence headroom audit above (Codex's plan of 2026-11-05 and the researcher's
decision of 2026-11-05, item 4); not part of the original preregistration. It asks how much of any A gain one side bit
explains. Deterministic, no fitted parameter, the same candidate set, stopping rule, budgets and caps as every policy:
1. d3 = the descriptor of the start view P3 (always the first observation).
2. If d3 is missing or exactly 0: choose as primary G (lambda 0, lookahead off) at every step.
3. Otherwise the preferred direction is increasing azimuth if d3 > 0, decreasing if d3 < 0. At each choice, among the
   affordable unvisited stops on the preferred side of the start stop, take the one nearest the current stop (ties:
   lower stop index); when there is none, choose as primary G.
It is reported beside A, never selected or tuned, and is not a candidate primary policy.

## The M16 check (readout fixed here, before it is run)

- Set: shelf_dev_v3, the 18 M16 arrangements of the 60 (9 side L, 9 side R; `manifest.json` occluder_family and m16_side),
  the start stop P3, the L1-reference jobs (`jobs_refL1.json`, as the evidence of record), lightings L1 and L2, conditions
  occluded and removed (36 views each), and visible (board parked, 36 views) as a null control. 108 views in all.
- Readout, per condition and lighting: the number of the 18 with d > 0, d < 0, d = 0 and missing (by reason), and the
  number whose sign agrees with the geometric prediction (L: d > 0, R: d < 0); every value listed. For removed episodes
  the agreeing side is also the side on which the episode is resolvable under the two-step cap (`side_audit.txt`).
- No pass threshold and no tuning. A wrong sign, a missing descriptor or a weak split is reported as it is.
- Whatever it shows, A's design does not change because of it: A includes this descriptor as defined, A-nocue excludes
  it with every other obstruction-derived feature, and no other feature or registration method is searched.
- Output: `runs/2026-11-05-side-descriptor-m16/` (untracked): the jobs subset, the descriptor rows and the readout.
