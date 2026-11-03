# Occlusion Aware Active Verification for Robot Scene Memory

**Course:** CMPE 249 Intelligent Autonomous Systems, Fall 2026  
**Team member:** Adithya Rajendran (solo)  
**Instructor:** Kaikai Liu  
**Selected track:** Track 4 - VLA / robotics  
**Project format:** Research with a bounded system prototype  
**Secondary alignment:** Track 6 - Safety and evaluation  
**Status (2026-11-02):**
- **Direction change.** With the instructor's agreement, the course study now runs **in simulation** (Isaac Sim 6.2 on the GB10; MuJoCo was used only to cross-check the geometry) instead of by physical capture; see the [amendment](docs/SIMULATION_STUDY.md). A real-robot study with the PIs is planned after the course. There are no study data or experimental results yet.
- **Built:** the capture tooling (also drivable from a phone), the offline verification core, evidence step v0.2.1, and a simulated replica of the shelf rig ([notes](notes/2026-11-02-simulation.md)). The replica reproduces the 2026-10-10 camera and occluder design in two independent renderers.
- **Real pilots:** two single-stop pilots. Frozen v0.2.1 got 51/52 per view on pilot 2, a held-out test of the code, and the one miss exposed a same-footprint occluder gap ([notes](notes/2026-10-20-pilot2.md)). These are now a real-world sanity check for the simulation.
- **Public datasets:** on hand as development data; OCID is turned into 903 labelled episodes ([notes](notes/2026-10-23-public-data-ocid.md)).

## Abstract

A robot that revisits a workspace may fail to detect a remembered object because it was removed, it is occluded, or the detector missed it. Visibility reasoning helps interpret a miss, but deciding whether to acquire another view also requires a sensing policy. I propose a bounded study of that decision using five calibrated camera positions and three possible outputs: present, removed from the monitored area, or unknown. A small learned view ranker will be compared with fixed, random, and geometric parallax selectors using the same perception front end, stopping rule, and observation and travel limits. A passive single-view rule provides a no-movement reference. The main question is whether learning improves on the geometric selector, and under which occlusion conditions. No experimental results are claimed at proposal time.

## Research question

Under a fixed observation and travel budget, does adaptive view selection with conservative stopping reduce false removals, at comparable coverage, relative to passive visibility reasoning and to fixed, random, and geometric inspection orders? Secondary: does a learned ranker add anything over parallax geometry, and under which conditions?

## Proposal submission materials

| Requirement | File |
| --- | --- |
| A - Initial project description | This README |
| B - Literature and SOTA survey | [Fourteen papers: ten recent and four foundational](docs/LITERATURE_REVIEW.md) |
| C - Canvas proposal | [PDF](docs/CMPE249_Project_Proposal.pdf), [editable Word document](docs/CMPE249_Project_Proposal.docx), and [Markdown source](docs/PROPOSAL.md) |
| D - AI novelty and feasibility audit | [AI audit](docs/AI_AUDIT.md) |
| Detailed evaluation design | [Experiment protocol](docs/EXPERIMENT_PLAN.md), amended for the [simulation study](docs/SIMULATION_STUDY.md) (2026-11-02) |
| Citation metadata | [BibTeX references](references.bib) |

## Progress since the proposal

- [Novelty check for the week-2 gate](docs/NOVELTY_CHECK.md)
- [Citation audit](docs/CITATION_AUDIT.md)
- [Lab notebook: dated entries with figures](notes/)
- [First real pilot and evidence v0 (2026-10-17)](notes/2026-10-17-first-real-pilot.md)
- [Pilot 2: frozen evidence v0.2.1 on a held-out arrangement (2026-10-20)](notes/2026-10-20-pilot2.md)
- [Public datasets, the OCID loader and v0.2.1 on OCID (2026-10-23)](notes/2026-10-23-public-data-ocid.md)
- [Simulation: MuJoCo and Isaac Sim replicas of the rig (2026-11-02)](notes/2026-11-02-simulation.md)

| Code | What |
| --- | --- |
| [`capture/`](capture/README.md) | Locked Pi capture, the guided sequencer `run_arrangement.py`, its phone front end `phone_ui.py`, tag registration |
| [`perception/`](perception/README.md) | Evidence step (frozen v0.2.1) and the perception smoke tests |
| [`verify/`](verify/README.md) | Offline verification core: replay, policies, the shared stopping rule, threshold selection |
| [`public_data/`](public_data/README.md) | Public datasets as development data (OCID loader, evidence runner, per-view scorer) |
| [`sim/`](sim/README.md) | Simulator-agnostic rig description and the MuJoCo backend (camera positions, visibility labels) |

## Initial scope

One tabletop or shelf; 6 to 10 objects; five camera positions; recorded-view replay first. Outputs are present, removed from the monitored area, or unknown. The same pretrained perception front end (a small YOLO-World checkpoint plus reference-crop matching; SAM 2 optional) serves every policy. Policies: S, a passive single-view visibility rule; F and R, fixed and random orders; G, a geometric parallax selector; A, the proposed regularized logistic view ranker. All multi-view policies share one sequential stopping rule, so only the view choice differs.

The planned main dataset is 180 episodes split by physical arrangement into 60 development and 120 test episodes. View selection cannot access unvisited query images. Models, thresholds, splits, and movement budgets are frozen before final testing.

## Devices and maintainer

| Available device | Planned use | Owner and maintainer |
| --- | --- | --- |
| ASUS Ascent GX10 (GB10) | Development, local inference, image processing, analysis, simulation (Isaac Sim 6.2) | Adithya Rajendran |
| Jetson Orin Nano 8 GB | Main development copy, tests, MuJoCo cross-check; optional onboard perception | Adithya Rajendran |
| Raspberry Pi 5 with AI HAT+ | Camera acquisition and optional motor control / supported inference | Adithya Rajendran |
| Raspberry Pi camera | Reference and query capture | Adithya Rajendran |

All listed equipment is personally purchased, owned, and managed by Adithya Rajendran. I manage software installations and development access. AI HAT+ execution of the selected models is not assumed. A motorized camera stage remains an optional build.

## Feasibility checkpoint

By the end of week 2, the pilot must show usable perception and meaningful differences between camera views. I will review overlap with the closest papers and narrow or revise the topic with the instructor if the experiment or contribution is not viable.

## AI assistance

**Proposal support:** ChatGPT and Claude assisted with drafting, literature review, and the AI audit.  
**Testing automation:** Codex or Claude Code will help automate test runs and result checks under my review.
