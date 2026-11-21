# Checking the trained STILL verifier on the scene-memory audit

**20 November update.** The authors released a trained checkpoint and clarified the decision rule for supplied boxes. We completed the archived comparison. The text below records what we knew on 19 November. [Results and reproducible scores](2026-11-20-trained-still.md)

19 November 2026. We found the authors' training, registration and scoring code. We could not verify a public copy of the trained Qwen3.5-9B verifier. We requested the checkpoint and its two-view evaluation settings from the first author. There are currently **zero trained STILL predictions** on our images. The paper's published detection scores use other images and a different endpoint. [STILL paper](https://arxiv.org/html/2611.18734v1) · [Authors' source at the inspected revision](https://github.com/Oran9er/STILL/tree/ef115fdfbcc69ca53f4874720c9e5eab429ec221)

## What is available

The repository had one public branch, one commit, and no releases or tags. The evaluation script loads a model from `runs/<run>/final`. The training script creates that directory and saves the trained weights and processor. The model names in the README identify the base models and other components. Searches of the model hub by author, paper identifier and title, and of Zenodo and Figshare by title, did not locate an author-linked checkpoint. The ordinary Qwen3.5-9B model is present on Ascent. [Training code](https://github.com/Oran9er/STILL/blob/ef115fdfbcc69ca53f4874720c9e5eab429ec221/scripts/still_sft.py) · [Evaluation entry point](https://github.com/Oran9er/STILL/blob/ef115fdfbcc69ca53f4874720c9e5eab429ec221/run/03_evaluate.sh) · [Release listing](https://github.com/Oran9er/STILL/releases)

The main audit contains 1,280 queries from 16 arrangements shared across Isaac Sim and MuJoCo. Each query has a reference image, its target box, and one revisit. We checked both input files for every query against the original checksums. We also prepared 1,280 blinded whole-image queries with the authors' two-frame question and box convention. The records carry no target-state label or previous prediction. This completes the whole-image input preparation. The trained inference and registered crop comparison are pending. [Original input verification](../paper/evidence/packet-restore-check.json) · [Two-frame question](https://github.com/Oran9er/STILL/blob/ef115fdfbcc69ca53f4874720c9e5eab429ec221/scripts/build_verify.py) · [Box conversion](https://github.com/Oran9er/STILL/blob/ef115fdfbcc69ca53f4874720c9e5eab429ec221/scripts/build_rc3d.py)

## Missed removals and false deletions

A missed removal means the object was removed and the verifier answered PRESENT or UNKNOWN. A false deletion means the object stayed and the verifier answered REMOVED. Counts refer to queries, so repeated views of one arrangement stay in the same analysis unit. We checked the published query table against the saved outputs from both systems.

| Queries | Method | Missed removals / actually removed | False deletions / actually retained |
| --- | --- | ---: | ---: |
| Full main audit | Dense geometric rule | 241/640 | 2/640 |
| Full main audit | Registered zero-shot Qwen3.5-9B | 221/640 | 51/640 |
| Passed both systems' registration check | Dense geometric rule | 20/419 | 2/415 |
| Passed both systems' registration check | Registered zero-shot Qwen3.5-9B | 0/419 | 51/415 |

All missed removals for these two methods were UNKNOWN. On the full panel, the shared registration check stopped 221 actually removed queries. Qwen detected all 419 removals it received. It also falsely deleted a retained object 49 more times than the geometric rule. The trained STILL comparison remains open. [Counts by engine and cohort](figures/2026-11-19-still-verifier/baseline-counts.csv) · [Per-query evidence](../paper/evidence/queries.csv) · [Saved model outputs](../paper/evidence/raw-scores/)

## Comparison when the checkpoint becomes available

Use the same 1,280 reference-and-revisit queries and supplied reference boxes for the primary comparison. Set each supplied box's proposal score to one. Run the authors' scene and registered crop questions, preserve their per-answer scores, and record where a registered crop was available. The published fusion takes the geometric mean of the two change scores and uses the scene score when a crop is unavailable. Evaluate all 1,280 queries even where the older Qwen gate abstained. Report its 834 admitted queries separately for continuity. [Published fusion](https://github.com/Oran9er/STILL/blob/ef115fdfbcc69ca53f4874720c9e5eab429ec221/scripts/combine_views.py)

Record the checkpoint identifier and fix the three-way decision rule before examining any STILL prediction. Map *same*, *changed* and *unseen* to PRESENT, REMOVED and UNKNOWN for this task, where removal is the only target change. The authors' two-view benchmark reports ranked average precision. That score alone supplies no threshold for the memory decision. Identify any local decision rule in the report, retain all abstentions and failures in the denominators, and check decisions on the 88 exact MuJoCo pairs. [Two-view scoring](https://github.com/Oran9er/STILL/blob/ef115fdfbcc69ca53f4874720c9e5eab429ec221/scripts/rc3d_ap.py) · [Audit pairs](../paper/evidence/pairs.csv)
