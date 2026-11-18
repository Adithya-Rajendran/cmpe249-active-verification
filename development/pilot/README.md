# Development pilot procedure

This procedure uses old development images. It does not authorize a confirmatory test.
Keep the review file and all existing run outputs.

## Prepare and audit

1. Run `python3 development/label_audit/build_v1.py` to create the audit ledger and sheets.
2. Run `python3 development/label_audit/prepare_inputs.py` for marked model inputs.
3. Review the inputs and save `support_review.json` before model predictions.

4. Run `python3 development/pilot/build_manifest.py` to initialize a pending manifest.
5. Run `python3 development/pilot/merge_review.py` to bind the review.
6. Run `python3 development/pilot/prepare_unmarked.py` for the image gate references.

7. Run `python3 development/pilot/prepare_evidence.py` for image-only jobs.
8. Stage the sanitized files and `perception/evidence_v03.py` on Ascent.
9. Run the image evidence jobs and save scores and diagnostics in the audit folder.
10. Run `python3 development/pilot/bind_evidence.py` to bind unmarked-reference scores.

The image gate uses unmarked resized RGB and the scaled reference box.
The model sees a red outline on the same resized reference and the same query.
The first marked-reference gate run is retained as superseded evidence.
The support review is a single non-blind AI review, with no human agreement measure.
See `SCORING_DRAFT.md` for prompts and complete answer scoring.
See `CALIBRATION.md` for the fixed development grid and analysis weights.

## Run and analyze

Run the unit tests and check the full manifest and image hashes before model use.
A reviewer must approve the concrete readiness package before task predictions.
Stage the manifest, model inputs and pilot source files on Ascent.

Run `run_models.py` once per pinned model. Supply the manifest, image folder and output path.
Each completed query is saved before the next call. Resume accepts only the same bound files.
A failed call is recorded and is not retried.
Run `analyze.py` once per model with the manifest, prediction file and result path.

Save the raw outputs, configuration, software versions and timing with the results.
A missing eligible calibration setting is a calibration failure, not a reason to change the grid.
The reported uncertainty is exploratory and uses arrangements as the sampling unit.

## Project environment

The Ascent project environment uses Python 3.13.5.
This pilot adds `num2words==0.5.14` and `docopt==0.6.2` to that environment.
The additions do not change the historical project lock files.
The package lists before and after the additions are in this folder.
A later environment sync can remove these extra packages.
Restore them with the project environment's uv command before a repeat run.

```sh
uv pip install --python .venv/bin/python 'num2words==0.5.14' 'docopt==0.6.2'
```

Run the pilot with `.venv/bin/python`, from the project environment.
Do not use `uv run` with automatic sync for this recorded setup.
