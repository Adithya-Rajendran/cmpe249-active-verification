# Counterfactual audit of object removal

Adithya Rajendran, San José State University.

Read [the paper](Rajendran-counterfactual-scene-memory.pdf) for the research question, results and interpretation.
The manuscript evaluates decisions about a remembered object on a shelf or a tabletop.
The source uses Typst. The supplementary tables support an independent recount.
This repository includes the query tables, model outputs, example images and source code.
The complete RGB and native simulation archives are held separately.

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
- Use `evidence/archived-capability.csv` to check the language model qualification.
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

The CPU recount used Python 3.10.12, NumPy 1.21.5, and Pillow 9.0.1.
The figures used Matplotlib 3.5.1. The calibration recount used SciPy 1.8.0.
The PDF build used the Python package `typst==0.14.2` with Python 3.13.5.
The PDF identifies its underlying Typst compiler as version 0.14.0.
The PDF build uses Bitstream Charter and Liberation Sans.

Run these commands from the manuscript directory.

```sh
python3 recount.py > reproduced-summary.json
python3 draw_example.py
python3 draw_calibration.py
python3 draw_results.py
python3 build.py
```

The recount checks each decision against the raw score and each pair against the earlier image audit.
It recomputes the paired layout intervals.
It also verifies the decoded pixels in the two exploratory examples.
The image archive uses one file per unique PNG.
The SHA-256 columns in the query table identify all reference and revisit images.

From the repository root, compare both independent analyses.

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

For a native calibration recount, extract the raw data archive in a new directory.
Run each command once per simulator. Replace `mujoco` with `isaac` for the second run.

```sh
python3 recount_calibration.py mujoco /path/to/native/mujoco/calibration
python3 recount_calibration.py mujoco /path/to/native/mujoco/calibration --convention pixel-centers
python3 render_differences.py mujoco /path/to/native/mujoco/calibration
```

See `software/README.md` in the supplement for simulator and scorer details.
