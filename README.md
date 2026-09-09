# Occlusion Aware Active Verification for Robot Scene Memory

**Course:** CMPE 249 Intelligent Autonomous Systems, Fall 2026  
**Team member:** Adithya Rajendran (solo)  
**Instructor:** Kaikai Liu  
**Selected track:** Track 4 - VLA / robotics  
**Project format:** Research with a bounded system prototype  
**Secondary alignment:** Track 6 - Safety and evaluation  
**Status:** Proposal stage; no experimental results yet

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
| Detailed evaluation design | [Experiment protocol](docs/EXPERIMENT_PLAN.md) |
| Citation metadata | [BibTeX references](references.bib) |

## Initial scope

One tabletop or shelf; 6 to 10 objects; five camera positions; recorded-view replay first. Outputs are present, removed from the monitored area, or unknown. The same pretrained perception front end (a small YOLO-World checkpoint plus reference-crop matching; SAM 2 optional) serves every policy. Policies: S, a passive single-view visibility rule; F and R, fixed and random orders; G, a geometric parallax selector; A, the proposed regularized logistic view ranker. All multi-view policies share one sequential stopping rule, so only the view choice differs.

The planned main dataset is 180 episodes split by physical arrangement into 60 development and 120 test episodes. View selection cannot access unvisited query images. Models, thresholds, splits, and movement budgets are frozen before final testing.

## Devices and maintainer

| Available device | Planned use | Owner and maintainer |
| --- | --- | --- |
| ASUS Ascent GB10 | Development, local inference, image processing, analysis | Adithya Rajendran |
| Jetson Orin Nano 8 GB | Optional onboard perception | Adithya Rajendran |
| Raspberry Pi 5 with AI HAT+ | Camera acquisition and optional motor control / supported inference | Adithya Rajendran |
| Raspberry Pi camera | Reference and query capture | Adithya Rajendran |

All listed equipment is personally purchased, owned, and managed by Adithya Rajendran. I manage software installations and development access. AI HAT+ execution of the selected models is not assumed. A motorized camera stage remains an optional build.

## Feasibility checkpoint

By the end of week 2, the pilot must show usable perception and meaningful differences between camera views. I will review overlap with the closest papers and narrow or revise the topic with the instructor if the experiment or contribution is not viable.

## AI assistance

**Proposal support:** ChatGPT and Claude assisted with drafting, literature review, and the AI audit.  
**Testing automation:** Codex or Claude Code will help automate test runs and result checks under my review.
