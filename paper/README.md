# Counterfactual audit of object removal

Adithya Rajendran, San José State University.

Read the paper or download the editable LaTeX source. [Paper (PDF)](Rajendran-counterfactual-scene-memory.pdf) · [Source for Overleaf (ZIP)](Rajendran-counterfactual-scene-memory-latex.zip)
The main study tests decisions about a remembered object on a shelf or a tabletop.
The manuscript uses the official CoRL 2026 preprint template and names the author. [Author instructions](https://2026.corl.org/contributions/instruction-for-authors)
Two analysis programs from the same project check the original decisions.
The [versioned release](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/tag/scene-memory-audit-v1) contains the original paper, images, model outputs, native captures, and source.
The LaTeX edition includes the trained STILL comparison, the visibility analysis and the retrospective estimate of one extra camera view.

The shelf follow-up repeats the original camera views at new angles.
The corridor follow-up uses eight fresh layouts, different objects and a sliding door.
The furnished workshop repeats those corridors with a traversable doorway and robot-mounted camera views.
Read the three [shelf](https://github.com/Adithya-Rajendran/cmpe249-active-verification/blob/scene-memory-audit-v1/notes/2026-11-18-level-shelf.md), [corridor](https://github.com/Adithya-Rajendran/cmpe249-active-verification/blob/scene-memory-audit-v1/notes/2026-11-18-corridor-transfer.md) and [workshop](https://github.com/Adithya-Rajendran/cmpe249-active-verification/blob/scene-memory-audit-v1/notes/2026-11-18-workshop-clearance.md) studies.
The manuscript reports each follow-up separately.
The portable supplement includes the protocols, decisions, pair tables, and independent checks under `followups/`.

## Reproduce the main comparison

Download the [supplement](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/Rajendran-scene-memory-supplement.zip), [delivered images](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/delivered-rgb.zip), and [SHA-256 checksums](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/SHA256SUMS.txt) into one directory.
The CPU checks use Python 3, NumPy, and Pillow.
Run the following commands from the download directory.

```sh
sha256sum --check --ignore-missing SHA256SUMS.txt
unzip -q Rajendran-scene-memory-supplement.zip -d scene-memory-paper
cd scene-memory-paper
python3 recount.py --removals > reproduced-removals.json
diff -u evidence/reader-comparison.json reproduced-removals.json
python3 check_rgb.py ../delivered-rgb.zip
python3 restore_inputs.py ../delivered-rgb.zip restored-packet
```

The recount checks each decision against the original scorer output.
It reports missed removals and false deletions by simulator and for both simulators together.
The main panel contains 640 removed and 640 retained queries.
The dense geometric rule missed 241 removals and falsely deleted two retained objects.
Registered Qwen missed 221 removals and falsely deleted 51 retained objects.

After scoring, we examined the shared registration check. It admitted 419 removed and 415 retained queries.
On these inputs, Qwen detected 20 more removals and falsely deleted 49 more retained objects than the geometric rule.
The geometric rule had correctly identified 42 of those 49 objects as present.
The image check recovers 88 exact MuJoCo pairs and zero exact Isaac Sim pairs.
The restored packet contains only the 1,280 input records and their original images.

For new GPU inference, use the scorer commands in `software/README.md` inside the supplement.
The Qwen verifier used the original weights with zero-shot prompting. [Pinned Qwen3.5-9B model on Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)
The primary comparison tests homography and dense geometry. Registered Qwen passed a clear-view screen and serves as a diagnostic reader.

### Reproduce the trained STILL comparison

The retrospective comparison uses the same 1,280 main queries and ten separate development layouts.
STILL missed 178 of 640 removals and falsely deleted 36 of 640 retained objects.
Zero-shot Qwen missed 221 and falsely deleted 51. [Results and example crops](../notes/2026-11-20-trained-still.md)

From a repository checkout, run these commands at the repository root with Python 3.11 or later.
The CPU recount needs only the Python standard library and the [comparison package](../notes/figures/2026-11-20-still-verifier/trained-still-comparison.zip).

```sh
evidence="$PWD/paper/evidence"
package="$PWD/notes/figures/2026-11-20-still-verifier/trained-still-comparison.zip"
work_dir=$(mktemp -d)
unzip -q "$package" -d "$work_dir"
cd "$work_dir"
python3 evaluate.py calibrate . "$evidence/exploratory-decisions.csv"
python3 evaluate.py evaluate . "$evidence/queries.csv" "$evidence/pairs.csv"
python3 raw_recount.py . "$evidence"
sha256sum --check SHA256SUMS.txt
```

In a standalone supplement built from this revision, set `evidence="$PWD/evidence"` and `package="$evidence/trained-still-comparison.zip"` at its root.
Then run the same commands from `work_dir=$(mktemp -d)` onward.

`counts.csv` gives both error counts by simulator and by the original Qwen alignment check.
`exact-pairs.csv` counts deletions on the 88 identical pairs.
`paired-differences.csv` gives the paired layout intervals.
The manuscript reads the exported [counts](evidence/still-comparison-counts.csv) and [exact-pair table](evidence/still-exact-pairs.csv).

To rebuild the paper's STILL figure, run this command inside the extracted comparison package.
The reference figure uses Matplotlib 3.11.2 and the two JPEG crops that the verifier received.

```sh
python3 draw_example.py . "$evidence/pairs.csv" still-registered-pair.png
```

For fresh STILL inference, follow the extracted package's README with the [main RGB archive](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/delivered-rgb.zip) and [development RGB archive](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/exploratory-inputs.zip).
The package includes both inference scripts, saved Pi3 projections, batch settings and model revision identifiers.
For fresh zero-shot Qwen inference, follow `software/README.md` in the [original scoring supplement](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/Rajendran-scene-memory-supplement.zip).

Xie, Li and Meng describe the trained verifier and supply the released seed-2 weights. [STILL paper](https://doi.org/10.48550/arXiv.2611.18734) · [Exact checkpoint](https://huggingface.co/oran9er/STILL-Qwen3.5-9B/tree/81d9373e1dd9c07c537d5fb2436b36a26a2206b4)
The original reader uses the Qwen Team's base release. [Qwen3.5 publication](https://qwen.ai/blog?id=qwen3.5) · [Exact base checkpoint](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)

### Retrospective extra-view estimate

The replay uses archived homography and dense answers from the original panel. It runs on a CPU and needs no new images.
From the repository root, compare its output with the saved table.

```sh
python3 paper/archived_stop_replay.py | diff - paper/evidence/archived-stop-replay.txt
```

After UNKNOWN, the replay permits one other stop in the same layout, simulator, camera level and physical state.
The random choice averages all four alternatives. The visibility choice uses the retained twin's evaluator mask.
The ceiling uses the true state and all alternative answers. The replay ignores travel costs.

Its historical `next` rule moves one stop toward P5 and returns from P5 to P4.
Section A of the [saved table](evidence/archived-stop-replay.txt) covers the 640 blocker-placed starts. The proposed fresh study ended before collection.

### Other image sets

| Download | Contents |
| --- | --- |
| [Exploratory inputs](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/exploratory-inputs.zip) | Earlier reference images, revisits, boxes, and decisions for 1,200 queries |
| [Qualification inputs](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/qualification-inputs.zip) | Images, expected answers, and model replies for the 116 archived controls |
| [Shelf follow-up](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/level-shelf-inputs.zip) | Delivered images, native images and masks, original answers, and camera comparisons for 320 queries |
| [Plain corridor](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/corridor-transfer-inputs.zip) | Inputs, native captures, raw answers, and an independent checker for 192 queries |
| [Corrected workshop](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/workshop-inputs.zip) | Inputs, native captures, raw answers, and an independent checker for 320 queries |
| [Native calibration](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/native-calibration.zip) | Original RGB, masks, and scene records for four calibration layouts in both engines |
| [Native Isaac evaluation](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/native-isaac-evaluation.zip) · [Native MuJoCo evaluation](https://github.com/Adithya-Rajendran/cmpe249-active-verification/releases/download/scene-memory-audit-v1/native-mujoco-evaluation.zip) | Original captures and scene records for each engine's 16 main layouts |

Use `restore_inputs.py` with the exploratory, qualification, or shelf archive to restore its blinded input directory.
The exploratory archive also preserves the original scorer job inventory.
The tagged file `evidence/image-archives.sha256.txt` records all eleven image archive checksums.
Each native archive supplies a file list with individual checksums under `checksums/`.
From the tagged repository root, verify the other two downloaded archives.

```sh
python3 development/corridor_transfer/pack_inputs.py verify /path/to/corridor-transfer-inputs.zip
python3 development/workshop_realism/supplement.py /path/to/workshop-inputs.zip
```

## Check the visibility analysis

Use the revised source and the original query and pair tables.
Run `python3 draw_visibility.py` from the manuscript directory to rebuild the figure and the [answer counts](evidence/answers-by-visibility.csv).
Each removed query inherits its retained twin's exposure. The table lists every answer by simulator, method, physical state, and exposure band.

The largest band starts at 99.95%. Of its 406 retained views, 402 had complete exposure.
All 47 exposed targets that Qwen falsely deleted had complete exposure. [Visibility figure](figures/answers-by-visibility.png)

Run `python3 reader_controls.py` for the unchanged images, target sizes, reference boxes, and exploratory score threshold.
We selected this threshold check after scoring.

## Read the evidence

- Open `evidence/queries.csv` for one row per fresh query.
- Open `evidence/pairs.csv` for matched retained and removed queries.
- Open `evidence/arrangements.csv` for target identities, seeds, and layout hashes.
- Read `evidence/summary.json` for coverage, physical errors, and layout intervals.
- Read `evidence/independent-evaluator/criteria.json` for each predeclared comparison condition.
- Read `evidence/independent-evaluator/breakdown.json` for results by target, layout, intervention, and camera motion.
- Open `evidence/independent-analysis-check.json` for the comparison of both analyses.
- Inspect `evidence/raw-scores/` for original model outputs and software fingerprints.
- Read `evidence/pre-score-input-audit.json` for the image comparison before scoring.
- Read `evidence/design/PROTOCOL.md` for the prospective development design.
- Use `evidence/calibration.json` to inspect the patch and the controls for each calibration pair.
- Use `evidence/archived-capability.csv` to check the language model screen on clear controls.
- Open `evidence/exploratory-decisions.csv` for the earlier comparison on ten layouts.

One row in the fresh query table contains a physical target state and one decision from each verifier.
The simulator keeps or removes the object with the blocker parked or placed.
The homography and dense geometry columns use the same appearance rule.
The registered column uses the zero-shot vision-language model.
Unknown counts against coverage and correct decisions per query.

The pair table records exact equality at inference resolution and at native resolution.
The recorded render-state interventions define target presence or removal in these static scenes.
Visual support remains unresolved on some unequal image pairs.

| Query field | Meaning |
| --- | --- |
| `id` | A random identifier that links each method decision to the delivered input |
| `level` | The nominal camera pose (`D0`) or a 6 cm shift and a 3° yaw (`D6`) |
| `state` | The target state with the blocker parked (`clear`) or at its intervention position (`placed`) |
| `visible_pixels` | The target mask area in this native revisit. A removed target gives zero. |
| `isolated_pixels` | The native target mask area in the isolated rendering at the same pose |
| `visible_isolated_overlap_pixels` | The native target pixels that also belong to the isolated target mask |
| `box` | The half-open reference box at the delivered resolution of 1200 by 675 pixels |
| `registered_gate` | Whether the registration allowed a language-model response, or the reason for an abstention or failure |
| `registered_raw` | The decoded language-model response when the scorer called the model |

Each pair links the retained query and the removed query at the same camera pose and blocker position.
The `exact` field includes both images, the reference box, and the fixed task.
The `native_exact` field compares the native reference and revisit images before resizing.

## Reproduce the paper

The source archive contains the main text, appendix, references, six figures, and both official CoRL style files. [Main LaTeX file](manuscript.tex)
It also contains the three evidence files and the program that generates the four result tables. [Editable source archive](Rajendran-counterfactual-scene-memory-latex.zip)

In Overleaf, upload the source archive as a new project.
Select `manuscript.tex` as the main document and compile it with XeLaTeX.
The archive includes the generated table rows for direct LaTeX builds.

For a local build, install Tectonic and put it on your command path.
The reference build used Tectonic 0.17.0 and its default TeX bundle. [Tectonic](https://tectonic-typesetting.github.io/en-US/)
Run the following command from the manuscript directory or the extracted source archive.

```sh
python3 build.py
```

This command regenerates the table rows from the evidence, runs LaTeX and BibTeX, and writes the PDF and editable ZIP.

### Check the data and figures

The CPU recount used Python 3.10.12, NumPy 1.21.5, and Pillow 9.0.1.
The original five figures used Matplotlib 3.5.1. The calibration recount used SciPy 1.8.0.

Run these commands from the full paper directory in the repository.

```sh
python3 recount.py > reproduced-summary.json
python3 draw_example.py
python3 draw_calibration.py
python3 draw_results.py
python3 draw_visibility.py
```

The recount checks each decision against the raw score and each pair against the earlier image audit.
It recomputes the paired layout intervals.
It also verifies the decoded pixels in the two exploratory examples.
The image archive uses one file per unique PNG.
The SHA-256 columns in the query table identify all reference and revisit images.

From the repository root, compare the two analysis programs.

```sh
python3 development/paper_audit/compare_exports.py paper/evidence paper/evidence/independent-evaluator \
  --homography baseline_200 --dense geometry --registered vggt_qwen
```

From the root of the standalone supplement, use the copy under `software/project`.

```sh
python3 software/project/development/paper_audit/compare_exports.py evidence evidence/independent-evaluator \
  --homography baseline_200 --dense geometry --registered vggt_qwen
```

This check compares all original decisions, both analytic controls, input pairs, primary intervals, and the improvement criteria.

Check every decoded image and matched pair directly from the archive.
Restore the original scorer packet in a new directory.

```sh
python3 check_rgb.py /path/to/delivered-rgb.zip
python3 restore_inputs.py /path/to/delivered-rgb.zip restored-packet
python3 draw_fresh_case.py /path/to/delivered-rgb.zip
```

The restored packet contains the original case order, numerical reference boxes, and image files.
Keep the evaluator tables separate when running a new verifier.

For a native calibration recount, extract `native-calibration.zip` into a directory named `native`.
Run each command once per simulator. Replace `mujoco` with `isaac` for the second run.

```sh
python3 recount_calibration.py mujoco /path/to/native/mujoco/calibration
python3 recount_calibration.py mujoco /path/to/native/mujoco/calibration --convention pixel-centers
python3 render_differences.py mujoco /path/to/native/mujoco/calibration
```

See `software/README.md` in the supplement for simulator and scorer details.
