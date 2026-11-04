# Reproducing the study

This is the single entry point for rerunning the code, the simulated dev sets and the checks.
Status as of 2026-11-03: the dev loop is running. There are **no study results yet**. Section 10 lists what cannot be reproduced yet.

Every command below was checked against the code (`--help` or the argparse source). Section 7 says which ones were also run, and where.

## 1. What the study is

- A robot revisits a workspace and cannot see a remembered object. Was it removed, hidden, or missed?
- The study compares view-selection policies (S, F, R, G, and a learned ranker A) under one stopping rule. Outputs are present, removed or unknown.
- Overview: [README.md](../README.md). Design: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), amended by [SIMULATION_STUDY.md](SIMULATION_STUDY.md). Analysis rules: [SIM_PREREGISTRATION.md](SIM_PREREGISTRATION.md) (draft, not frozen).

**Simulation or physical**
- **The course study is simulation only.** Images are Isaac Sim 6.2 renders of a replica of the measured shelf rig. Every result is labelled "simulation".
- **Physical** means the capture tools in `capture/` and two single-stop real pilots (notes of 2026-10-17 and 2026-10-20). They are a sanity check of score ranges. They are never pooled with simulation results.
- **Synthetic** means random evidence with no images (`--synthetic`). It tests the pipeline and never appears in results.

## 2. Hardware and hosts

| Host | Role | Needed for |
|---|---|---|
| **Jetson** (Orin Nano 8 GB, Linux aarch64) | Working copy of the repo. Unit tests, arrangement generation, episodes, set checks, `verify.*`, the run driver | Everything except rendering and evidence on GPU |
| **GB10** (ASUS Ascent GX10, DGX Spark class, aarch64) | Isaac Sim 6.2 renders. DINOv2 + YOLO-World evidence on its GPU | `sim.isaac_batch`, `perception.evidence` |
| **Raspberry Pi 5** + Camera Module 3 | Physical capture (`capture/`) | The physical pilots only. Not needed for the simulation study |

- Isaac Sim 6.2 is supported by NVIDIA on the GB10's aarch64 build, not on the Jetson (no RT cores) ([notes/2026-11-02-simulation.md](../notes/2026-11-02-simulation.md)). Other hosts are untested.
- Hosts are written as placeholders here: `<GB10_HOST>`, `<GB10_USER>`, `<PI_HOST>`. Use ssh key login (`BatchMode`), because the run driver needs it.
- **One shared data folder.** The Jetson and the GB10 must see the same data root at the same absolute path. Jobs and annotations store absolute paths.

```bash
export DATA_ROOT=/path/to/cmpe249-active-verification   # the author used /mnt/bench-data/datasets/cmpe249-active-verification
export SETS=$DATA_ROOT/sim
```

## 3. Software

### 3.1 The Jetson (tests, generator, episodes, checks, verify)

- Tool: [uv](https://docs.astral.sh/uv/). Python is pinned to **3.13.5** (`.python-version`).
- These steps need only numpy and OpenCV. Use a throwaway env, not the project env:

```bash
cd <REPO>
uv run --no-project --python 3.13.5 --with numpy==2.5.3 --with opencv-python-headless==5.0.0.93 \
    python -m unittest discover -s tests -t .
```

- Expected: `Ran 645 tests`, `OK (skipped=10)`, about 110 s on the Jetson. Skips: 9 need `mujoco`, 1 needs CUDA and the model weights.
- Below, `UV_RUN` is that same environment as a shell variable:

```bash
UV_RUN="uv run --no-project --python 3.13.5 --with numpy==2.5.3 --with opencv-python-headless==5.0.0.93 python"
```

- MuJoCo is optional (geometry cross-check only). It is not in `uv.lock`; add `--with mujoco==3.17.0` ([sim/README.md](../sim/README.md), "How to run").

### 3.2 The GB10 project env (evidence)

- `pyproject.toml` plus `uv.lock` pin torch 2.12 (CUDA 13.0 wheels from the PyTorch index), ultralytics 8.4.163, transformers, open-clip-torch and the rest.
- Install: `cd <REPO> && uv sync`. Run: `uv run --no-sync python -m ...`.
- `uv.lock` is restricted to `linux` + `aarch64` (`[tool.uv] environments`). `uv sync` will not resolve on x86.
- Copy the whole repo to the GB10, for example with `rsync -a --exclude data --exclude runs --exclude .git <REPO>/ <GB10_USER>@<GB10_HOST>:<REPO_ON_GB10>/`. Evidence imports `verify/` and `capture/`.

### 3.3 The GB10 Isaac Sim env (renders)

- A separate Python 3.12 env, outside the repo and the project env. About 18 GB. The author approved NVIDIA's licence on 2026-11-02 (you must accept it yourself: `OMNI_KIT_ACCEPT_EULA=YES`).
- Install once ([sim/README.md](../sim/README.md), "Isaac Sim backend"; [notes/2026-11-02-simulation.md](../notes/2026-11-02-simulation.md)):

```bash
cd ~/cmpe249-sim/isaacsim-6.2 && uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "isaacsim[all,extscache]==6.2.0.0" "tinyobjloader==2.0.0rc13" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
```

- uv needs `--index-strategy unsafe-best-match` (cross-index matching) and the explicit pin for the pre-release `tinyobjloader`.
- Launch environment, every time:
  - `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` (aarch64 only)
  - `OMNI_KIT_ACCEPT_EULA=YES`
- Driver: 595.99.02 worked. NVIDIA documents 580.159.03 for Spark. A forum report of R595 crashing Isaac Sim 6.0 did not reproduce on 6.2.
- Folder layout used on the GB10 (the layout `tools/pipeline.py` and the render script assume):

```
~/cmpe249-sim/
  isaacsim-6.2/.venv/        the Isaac env
  assets/ycbv/models/        YCB-Video models (section 3.4)
  run/                       isaac_batch.py and isaac_rig.py (exact copies from sim/), arrangement folders, logs
```

- **Copy `sim/isaac_batch.py` and `sim/isaac_rig.py` unchanged into `run/`.** Their sha256 go into every render, and `--skip-existing` compares them.

### 3.4 YCB-Video model assets

- Source: `ycbv_models.zip` from the BOP benchmark on Hugging Face, `bop-benchmark/ycbv` (MIT per the card; the YCB models keep their own terms, see [SIMULATION_STUDY.md](SIMULATION_STUDY.md) rule 5).
- Unzip so that the folder `models/` (108 files: `models_info.json`, `obj_0000NN.ply`, `obj_0000NN.png`) lands at `~/cmpe249-sim/assets/ycbv/models` on the GB10. Pass it as `--assets-dir`.
- Check: `sha256sum models/models_info.json` must give `13978298d8c37f089a7e0c421ed8be867fa836818936f5499786175da2c4c4cb`. That is the hash recorded in `configs/sim/arrange_shelf_v3.json`.
- The author's zip has sha256 `1e517a9b513cadb8eaf406ac3ce3c5f1d6fc4a80161c866b4e2a73d13de5e554` (`.verified` marker on the share).
- The generator also pins `configs/sim/ycbv_hulls.json` (sha256 `9d7004d0…7e3c`) and refuses another file.
- Corner tag images (`--tags-dir`) are read only by scenes with tags (shelf_v1 and v2). shelf_v3 is tag-free, so the folder is not needed.

### 3.5 Evidence model weights

- Evidence v0.2.1 (`perception/evidence.py`) uses:
  - detector `yolov8s-worldv2.pt` (YOLO-World v2 small, via ultralytics), and
  - DINOv2-small, `facebook/dinov2-small` at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- They download on first use. Detector weights and the CLIP text encoder go to `checkpoints/` (gitignored). DINOv2 goes to the Hugging Face cache. Needs network access once.
- No weights are stored in the repo. `perception/README.md` ("Setup and run") has the details.

## 4. Data layout

`$SETS/<set>/` is one rendered set. Folders are never edited by hand.

| Path | What |
|---|---|
| `<ARR>/` | One arrangement (`SD1000`, `DES0`, ...): `arrangement.json`, `rgb/` (40 PNGs: 4 phases x 5 stops x L1/L2), `masks/`, `annotations/`, `labels.json`, `render.json`, contact sheets |
| `manifest.json` | Verify manifest, `"simulation": true`, split dev, with `occluder_family`, `m16_side`, `design_case` per arrangement |
| `view_labels.json` | Exact per-view visible fractions and diagnostic labels, from instance masks. Scoring and calibration only, never a policy input |
| `true_alignment.json` | Exact query-to-reference homographies from the camera poses. Evaluation only, never an evidence input |
| `truth.json` | present / removed per episode. Scoring only |
| `jobs_refL1.json` | **Primary** evidence jobs: every query (L1 and L2) against the L1 reference |
| `jobs_same.json` | Secondary jobs: each query against the reference of its own lighting. Reported separately |
| `evidence_refL1.jsonl`, `.diag.jsonl` | Evidence output, written next to the jobs (section 5.5) |
| `*_design*` | The same files for the design cases DES0-DES3, which form a separate group |
| `generation_log.json` | The generator's failed tries and replacements |
| `positions_shelf_A.json` | Copy of `configs/sim/positions_shelf_A.json` |
| `set.json`, `summary.md`, `README.md` | File hashes and generator/config hashes, a per-arrangement table, a description |
| `check/check.json`, `check_stdout.txt` | Output of `sim.check_batch` |

**Which sets are current** (all under `$SETS/`; the layout is as the author's share, none of it is in git)

| Set | Status |
|---|---|
| `shelf_dev_v3` | **Current dev set.** 60 slot arrangements `SD1000`-`SD1059` plus `DES0`-`DES3` = 64. Tag-free. Rendering was in progress on 2026-11-03 |
| `shelf_dev_v3_check` | **Current check set.** 8 random arrangements plus DES0-DES3, tag-free, rendered 2026-11-03. Has no evidence yet |
| `shelf_v3_geometry/dev60/` | The generated v3 arrangement files and `index/` for the 60 dev slots. This is the render input of `shelf_dev_v3` |
| `shelf_dev_v0`, `shelf_dev_v1_check`, `shelf_dev_v2_check` (+ `_superseded_adjacent_only`) | **Superseded records.** v0: queries at the nominal pose, with tags. v1: target drawn per try, with tags. v2: 15 cm M16 board, with tags. Kept for the notes that cite them |

- `shelf_dev_v1_check` is the only set with evidence output today (v0.2.1, 6 arrangements; the smoke test of [the note](../notes/2026-11-03-check-batch-evidence.md)).
- Generator configs: `configs/sim/arrange_shelf_v3.json` is the default and the current one. `arrange_shelf_v0/v1/v2.json` are records. Pass `--config` to use them.

## 5. The pipeline, step by step

Run from `<REPO>`. Replace `$SET` with `$SETS/shelf_dev_v3`.

### 5.1 Generate arrangements (Jetson)

```bash
$UV_RUN -m sim.arrange --split dev --slots all --design --out $SETS/shelf_v3_geometry/dev60
```

- `--config` defaults to `configs/sim/arrange_shelf_v3.json`. `--slots` takes `all`, `first-per-class` or indices such as `0-5,9`. A non-empty `--out` is refused without `--overwrite`.
- Output: 60 `SD*.json` + 4 `DES*.json` + `index/{index,slots,generation_log}.json`. The log of the author's run: 76 failed tries, 0 replacements.
- Runtime: 186 s for all 60 slots plus the design cases, on the Jetson (`dev60.seconds` on the share).
- `--split test` is refused (section 9).
- The output is deterministic. A rerun of slots 0-5 plus the design cases gave files byte-identical to the author's `dev60/`.

### 5.2 Render (GB10)

Copy the arrangement folder to `~/cmpe249-sim/run/arr_shelf_dev_v3/` on the GB10, and `sim/isaac_batch.py` and `sim/isaac_rig.py` into `run/`. Then:

```bash
cd ~/cmpe249-sim/run
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 OMNI_KIT_ACCEPT_EULA=YES \
  ~/.local/bin/uv run --no-project --python ~/cmpe249-sim/isaacsim-6.2/.venv/bin/python isaac_batch.py \
  --arrangements arr_shelf_dev_v3/*.json --assets-dir ../assets/ycbv/models \
  --out $SETS/shelf_dev_v3 --skip-existing
```

- Other options (defaults in brackets): `--lighting` [L1 L2], `--renderer` [RaytracedLighting], `--rt-subframes` [16], `--settle` [200 updates], `--capture-updates` [100], `--tags-dir` [tags].
- Run it in the background with a log and an exit-code file (section 6).
- Each arrangement is written to `<ARR>.partial` and renamed when complete. A crash leaves a `.partial` folder.
- **Runtime:**
  - About 130 s per arrangement. The 10-arrangement v1 check took 124.5-130.6 s each, median 128.6 s (`render_report_20261103T000139.json`, [note](../notes/2026-11-03-dev-batch-check.md)).
  - The v3 check set took 123-174 s each (`sim.check_batch` output, a busier machine).
  - Boot is about 9 s. The first settle includes one-time shader compilation, about 100 s ([sim/README.md](../sim/README.md)).
  - 64 arrangements x 130 s is about 2.3 h. That is arithmetic, not a measurement of the full set.
- Output per arrangement: section 4. Progress lines start with `ISAAC_BATCH` and give seconds, patterns and `label_sanity_ok`.

### 5.3 Build episodes (Jetson)

```bash
$UV_RUN -m sim.episodes --root $SET --contact-sheets \
    --generation-log $SETS/shelf_v3_geometry/dev60/index/generation_log.json
```

- Run it on the set's **final** location (paths in the jobs are absolute).
- `--ref-lighting` chooses `L1` (writes `jobs_refL1.json`), `same` (`jobs_same.json`) or both (default). `--positions` must be the file the arrangements recorded.
- It refuses incomplete, mixed, wrong-positions and `.partial` folders.
- Output: the manifest, jobs, truth, `view_labels.json`, `true_alignment.json`, `set.json`, `summary.md` (section 4).

### 5.4 Check the set (Jetson)

```bash
$UV_RUN -m sim.check_batch --root $SET            # writes $SET/check/check.json
```

- Exit 0 if no arrangement is flagged. Exit 1 if one is flagged.
- Runtime: 16 s for the 12 arrangements of `shelf_dev_v3_check`.
- Checks (tag checks are skipped, and say so, for tag-free sets):
  - frame margins of every capture, revisit pose deltas (+-10 mm, +-0.5 deg), visible fractions and patterns;
  - the true-alignment corner displacement per view;
  - how far visible-phase queries differ from their references, and render seconds;
  - the label-sanity guard: in the visible phase each stop must show at least 0.9 x its target-alone pixels at the same pose;
  - crash leftovers (`.partial`, `.old*`) are skipped and reported. A folder whose id is not its name is refused.
- On `shelf_dev_v3_check`: 12 arrangements, 480 captures, no flagged arrangement.

### 5.5 Evidence (GB10)

**v0.2.1 (frozen baseline):**

```bash
cd <REPO_ON_GB10>
uv run --no-sync python -m perception.evidence --jobs $SET/jobs_refL1.json \
    --out $SET/evidence_refL1.jsonl --diag $SET/evidence_refL1.diag.jsonl --device cuda
```

- Speed: 180 views in 34.5 s on the GB10 ([note](../notes/2026-11-03-check-batch-evidence.md)).
- `perception/evidence.py` is frozen at sha256 `d0e0927ff9d36ee2a8027cc8050f1801e372faea1e12cd98c513345d258d46cf`. Check it with `sha256sum`. Never edit it.
- **Tag-free scenes:** v0.2.1 finds no tags, falls back to no alignment, and sets quality to 0 for every view. An unregistered view never counts as a clear miss. So on shelf_v3 it is only a baseline.

**v0.3 (dev, not frozen):** image-based registration (ECC homography) plus a background-consistency occlusion cue. The same pass also writes the *v0.2.1 + image alignment* ablation:

```bash
cd <CODE_DIR>      # a checkout or `git archive` of the commit; checkpoints/ linked as in section 3.5
PYTHONPATH=. uv run --no-sync --project <GB10_PROJECT> python -m perception.evidence_v03 --align image \
    --jobs $SET/jobs_refL1.json --out $OUT/v03_image_refL1.jsonl \
    --out-v021-aligned $OUT/v021_image_refL1.jsonl --diag $OUT/v03_image_refL1.diag.jsonl --device cuda
```

- Speed on `shelf_dev_v3`: 1800 views in 2013 s (CPU-bound ECC). Frozen v0.2.1 took 225 s for the same views.
- The driver (section 6, `configs/pipeline/shelf_dev_v3_evidence.json`) runs both versions for the random and the DES jobs, and checks and records every output.

**True alignment (evaluation only):** `python -m sim.true_align_evidence --set $SET --plane target|front --jobs ... --out ... --diag ...` runs v0.3 on the simulator's true homography. It is a diagnostic and never an input to calibration or policies (section 6).

### 5.6 Calibrate thresholds: `verify.select` (Jetson)

```bash
$UV_RUN -m verify.select --manifest $SET/manifest.json --evidence $SET/evidence_refL1.jsonl \
    --view-labels $SET/view_labels.json --positions configs/sim/positions_shelf_A.json \
    --tag shelf_dev_v3 --out runs/select-v4-shelf_dev_v3
```

- Procedure select-v4 (one joint grid search over tau_q x tau_occ x tau_match; [SIM_PREREGISTRATION.md](SIM_PREREGISTRATION.md) section 7). Only dev arrangements are accepted.
- Exit codes:

| Code | Meaning | Files |
|---|---|---|
| 0 | Candidate written and eligible for freeze | `frozen_candidate.json`, `.sha256`, `selection_details.json`, `selection_report.md` |
| 2 | Usage or input error | none |
| 3 | Calibration infeasible on all dev data. No candidate | `calibration_infeasible.json` |
| 4 | Candidate written but **not** eligible (a CV fold is infeasible, or fewer than two dev arrangements) | as 0, marked `eligible_for_freeze: false` |

- Nothing is relaxed automatically. Exit 3 is a valid outcome that goes in the notes.
- Input check: `--manifest` needs `--evidence`, `--view-labels` and `--positions`, or it exits 2.
- Other options: `--alpha-clear` [0.10], `--frr-limit` [0.10], `--budget` [3], `--cap-steps` [1,2,3,4,6], `--r-seeds` [20], `--bootstrap-B`, `--k-sensitivity`, `--lam-sensitivity`, `--lam-breakpoints`.
- Known result so far: the 6-arrangement smoke test with v0.2.1 gave exit 3 (best false-clear rate 20/36 = 0.556 against 0.10; [note](../notes/2026-11-03-check-batch-evidence.md)). There is no frozen candidate.

### 5.7 Replay and score: `verify.run` (Jetson)

```bash
$UV_RUN -m verify.run --manifest $SET/manifest.json --evidence $SET/evidence_refL1.jsonl \
    --view-labels $SET/view_labels.json --positions configs/sim/positions_shelf_A.json \
    --tag shelf_dev_v3 --out runs/run-shelf_dev_v3
```

- Writes `config.json`, `logs.jsonl`, `ground_truth.json`, `results.json`, `table.md`. A manifest with `"simulation": true` is labelled SIMULATION. With no test arrangement it scores on dev and runs no bootstrap.
- Thresholds are still **placeholders**. `verify.run` does not read a frozen candidate yet. Its numbers are not results.
- Options: `--budgets` ('0..4'), `--cap-steps`, `--r-seeds` [20], `--bootstrap-B` [10000], `--primary-budget` [3].

### 5.8 Analysis

- `analysis/` (`metrics.py`, `bootstrap.py`, `intervals.py`) is a library with no command line. `verify.run` and `verify.select` call it.
- The paired arrangement-level cluster bootstrap is stratified by occluder family when every stratum has at least 2 arrangements. See [SIM_PREREGISTRATION.md](SIM_PREREGISTRATION.md) sections 8-9.

## 6. The run driver and watcher

`tools/pipeline.py` runs render -> episodes -> check -> evidence -> select -> summary unattended for one set; a stage without a spec section is skipped. `tools/watch.py` prints only what needs a person.

```bash
UV_PY="uv run --no-project --python 3.13.5 python"       # both scripts are stdlib only
$UV_PY tools/pipeline.py configs/pipeline/shelf_dev_v3.json --launch      # --from-stage / --to-stage, --run-dir
$UV_PY tools/watch.py runs/shelf_dev_v3 --stale-minutes 20 --resume        # --from-start
```

- **GB10 connection.** Copy `configs/pipeline/remote.example.json` to `configs/pipeline/remote.local.json` (gitignored) and fill in `host`, `host_key_alias`, `user`, `key` and `run_dir`. Run specs refer to it by path.
- **Run spec** (`configs/pipeline/shelf_dev_v3.json`): `render.launch_script` (the launch command and its arguments), `arrangements_dir`, `log`, `rc_file`, `expected` (64), and `set_root` / `generation_log` (absolute paths on the shared data folder).
- **Launch script.** `sim/run_render.sh` is the committed launcher. Copy it to `remote.run_dir` next to `isaac_batch.py` and `isaac_rig.py`. It takes `ARR_DIR OUT_SET LOG RC_FILE [isaac_batch args]`, writes stdout and stderr to the log, and writes the exit code to the rc file when the render ends. (The first shelf_dev_v3 render was started with an equivalent GB10-only `run_v3.sh`.)

- The driver attaches to a running render by `pgrep` on `isaac_batch.py --arrangements <arrangements_dir>/`, follows the log, and waits for `v3.rc`. Without `--launch` it fails if nothing is running.
- It then runs `sim.episodes` and `sim.check_batch` locally with `local_python` from the spec.
- Output in `runs/<run_id>/`: `events.jsonl` (one JSON object per event), `status.json` (stage, progress, ETA, driver pid), `episodes.out`, `check.out`.
- Event levels: `info`, `alert` (a person must look), `done`. A terminal failure is an alert with `terminal: true`, and the driver exits non-zero.
- Render alerts: error line in the log, failed label sanity, an arrangement slower than 2.5 x the running median, no arrangement for 10 min, process gone without an rc file, non-zero rc, or fewer arrangements than expected, and 5 failed ssh polls in a row.
- **Evidence, select and summary stages** (`configs/pipeline/shelf_dev_v3_evidence.json`, run with `--from-stage evidence`):
  - `evidence` ships `git archive` of the last commit that changed a `.py` file under `perception/`, `verify/` or `capture/`, or `pyproject.toml` or `uv.lock`, to `<code_dir>/<job>` on the GB10, links the checkpoints, and runs one step per (jobs file, variant) in the GB10 project env. It refuses a dirty tree and a `perception/evidence.py` that is not the frozen v0.2.1. The job is keyed on the run id, that commit, and a hash of the steps and the jobs files. Rerunning after a driver-only or docs change re-attaches instead of recomputing. Any change to the code, steps or jobs launches a new job. Each job (`evidence_<run_id>_<commit>_<steps hash>`) has its own code dir and its own output dir, so two jobs never share or overwrite files. Provenance also records the GB10 env: its `uv.lock` sha256 and the installed package versions.
  - Variants: `v021_frozen` (perception/evidence.py, tag registration; on tag-free scenes it refuses every view), `v03_image` (evidence v0.3 on ECC image registration) and, from the same pass (`--out-v021-aligned`), `v021_image`: v0.2.1's match, quality and obstruction on v0.3's registration. That is an ablation, not unchanged v0.2.1.
  - Outputs go to `<evidence.out_dir>/<job>/` on the shared volume (`{out}` in the spec); only that job's launch empties and writes it. The driver checks one record per job in every output and writes `evidence_provenance.json` in the run dir and `provenance.json` in the job's output dir (commit, commands, output dir, sha256 and record count of each output). Reattaching to a job that had already finished checks every output against the sha256 saved when it finished and stops if any differs.
  - `select` and the summaries read evidence only through the run's `evidence_provenance.json` (`tools/provenance.py`): a file must be a recorded output and still have its recorded sha256. The true-alignment summary reads the image diagnostics through the provenance of the run named by `summary.image_run`. Runs made before per-job output dirs (all outputs in `out_dir`) still read correctly.
  - `select` runs `verify.select` on each variant's random-arrangement evidence, with the same alpha, grid and folds for every variant. Exit 3 (infeasible) and 4 (not eligible) are recorded in `select/outcomes.json`, not treated as failures.
  - `summary` (`tools/evidence_summary.py`) writes `summary.md`/`summary.json`: registration refusals, gate pass by view class per lighting (at each variant's thresholds and at one illustrative setting), held-out FRR / recall / coverage per lighting, and the DES cases separately.
- **Host paths.** The evidence stage takes `uv`, `project`, `checkpoints` and `code_dir` from the `evidence` block of `remote.local.json` (see `remote.example.json`), not from the tracked spec.
- **Order and working directory.** Run the specs from the repo root, in this order:
  1. `shelf_dev_v3_evidence.json --from-stage evidence`;
  2. `shelf_dev_v3_truealign.json --from-stage evidence`. Its summary reads `runs/shelf_dev_v3_evidence/select/v03_image/frozen_candidate.json`, a run output of step 1, by a path relative to the repo root.
- **True-alignment diagnostic** (`shelf_dev_v3_truealign.json`): `sim/true_align_evidence.py` on the true target and front planes. Its `summary.script` is `tools/quality_resampling.py`, which compares every view's quality, quality parts and gate / clear-miss / strong-match decisions with the image-registered v0.3 run, and writes `quality_resampling.md`. Its `evidence.code_paths` adds that tool to the shipped code.
- **Retrying a failed evidence job.** A job that exited non-zero leaves its rc file, and a rerun re-attaches and fails again. After fixing the cause, rerun with `--retry`, which clears that job's rc file and relaunches.
- **Offline env.** The evidence specs run the local stages with `uv run --offline`. Run the section 3.1 test command once first, so uv's cache holds the same pins.
- Watcher output: `ALERT ...`, `DONE ...` (exit 0), `FAILED ...` (exit 1). It also alerts if the driver pid died or no event came for `--stale-minutes`. `--resume` continues from `RUN_DIR/watch.offset`.

## 7. What you can verify

**Unit tests** (section 3.1): 645 tests, OK, 10 skipped. The tests for the physical capture tools are separate: `uv run ... python -m unittest discover -s capture/tests -t .` gave 159 tests, OK, 6 skipped, in 292 s on the Jetson. On the Pi: `python3 -m unittest discover -s capture/tests` ([capture/README.md](../capture/README.md)).

**Synthetic reruns are byte-identical.** Run each command twice with different `--out` and compare with `diff -r`:

```bash
$UV_RUN -m verify.run --synthetic --n-arr 30 --seed 0 --budgets 0..4 --bootstrap-B 200 --tag t --out /tmp/run_a
$UV_RUN -m verify.select --synthetic --n-arr 30 --seed 0 --bootstrap-B 200 --tag t --out /tmp/sel_a
```

- Both pairs of output folders were identical on 2026-11-03 (`diff -r` printed nothing). `verify.run` took about 92 s per run on the Jetson.
- The default is `--bootstrap-B 10000`; a smaller value just saves time.
- `verify.select --synthetic` exits 0 and writes a candidate that is marked synthetic. `require_freeze_eligible` refuses such a candidate.

**Checks on a rendered set:** `sim.check_batch` (section 5.4).

**Cross-checks of the evidence file:** on the share, `sha256sum $SETS/shelf_dev_v1_check/evidence_refL1.jsonl` starts with `a2a75da1` (as in the note).

**A sim-data replay:** `verify.run --manifest $SETS/shelf_dev_v1_check/manifest.json --evidence $SETS/shelf_dev_v1_check/evidence_refL1.jsonl --positions configs/sim/positions_shelf_A.json` ran with exit 0 and wrote `table.md`, labelled SIMULATION.

## 8. Determinism and provenance

- **Seeds.** Dev seeds are 0-9999. Slot arrangements use `primary_seed_start` 1000 (`SD1000`...), reserve seeds 2000-2411 (one block per family), and the quota slot list comes from a seeded permutation (`permutation_seed` 20261103). Test seeds start at 100000 and are in the committed config. Seeds 10000-99999 are unassigned and refused.
- **Arrangements.** An arrangement depends only on seed, attempt, family, side, target, the config content, the code and the input-file hashes (hulls, positions, geometry), and on numpy's version. Use numpy 2.5.3.
  - `arrangement_sha256` hashes the record without its `unhashed` block (config path, file byte hash, Python version).
  - `set.json` `provenance` holds `generator_sha256`, `config_sha256`, `positions_sha256`, `episodes_py_sha256` and every arrangement's hashes.
  - `sim.episodes` refuses a set whose arrangements come from more than one generator or config hash.
- **Revisit noise.** One seeded pose draw per (arrangement, stop). It is shared by the visible, occluded and removed phases, both lightings and every policy. References use the nominal poses.
- **Render settings hash.** `labels.hashes.render_settings_sha256`, with the full record in `render.json`: resolution, renderer, rt subframes, settle and capture updates, lighting presets, materials, Isaac version, `models_info.json` hash and tag image hashes.
- **`--skip-existing`** skips an arrangement only if all of these hold. Otherwise it re-renders and prints the reason.
  - `labels.json` records the sha256 of the arrangement file given now.
  - The sha256 of `isaac_batch.py` and `isaac_rig.py` match.
  - The lighting set and render settings hash match.
  - Every phase x stop x lighting PNG is present.
- **Renders are not bit-reproducible.** Two renders of an unchanged scene are never pixel-identical (mean difference about 0.1 grey level, [note](../notes/2026-11-03-dev-batch-check.md)). Labels come from integer instance masks, not colours. Rerunning a render gives the same arrangement and nearly the same pixels, not the same bytes.
- **Sidecars.** `frozen_candidate.json.sha256` is checked by `require_freeze_eligible`, which accepts a candidate file only. `set.json` and `render.json` carry file and code hashes.
- **Frozen evidence.** `perception/evidence.py` v0.2.1, sha256 `d0e0927ff9d36ee2a8027cc8050f1801e372faea1e12cd98c513345d258d46cf`. It was run on the pilots and the check batch without edits.
- **Run record.** The run driver stores the git commit and a dirty flag in `status.json`. `runs/` is gitignored.

## 9. The test-split rule

**No test seed is generated, rendered or scored before `docs/FREEZE.md` exists.** [SIM_PREREGISTRATION.md](SIM_PREREGISTRATION.md) section 12 lists what the freeze records.

- `docs/FREEZE.md` does **not exist** yet. So no test scene exists.
- In code today the refusal does not look for the file. It is unconditional:
  - `sim.arrange --split test` exits 2 (`seed 100000 is in the reserved TEST range`).
  - `sim.isaac_batch` refuses arrangements with seeds outside 0-9999 before Isaac starts.
  - `verify.select` accepts only dev arrangements.
- How the guard is lifted at the freeze is not written yet.
- Do not work around the guard. A test scene made before the freeze invalidates the test split.

## 10. Known gaps: not yet reproducible

- **Ranker A is not built.** Policies S, F, R, G (and ablations) run. The held-out check has a per-fold hook for A. No A exists, so the primary contrast A - G cannot be run.
- **Evidence v0.3 is not frozen.** It is the development evidence step (section 5.5). Freezing it is part of the final pre-registration review.
- **No feasible frozen thresholds.** The only select-v4 run on rendered data (6 arrangements, v0.2.1) was infeasible (exit 3). `verify.run` uses placeholder thresholds.
- **Dev results only, shelf only.** Evidence and select-v4 on `shelf_dev_v3` (`notes/2026-11-04-dev60-evidence-variants.md`) are development numbers.
- **The tabletop environment does not exist in the repo.** No config, positions file or scene. Results are not yet split by environment.
- **`docs/FREEZE.md` and the test split do not exist.** The pre-registration is a draft. A final review comes after A.
- **No results.** Nothing in this repo is a study result.
- **The generators of the geometry files are not in git.** `configs/rig_geometry.json` and `configs/sim/positions_shelf_A.json` are committed outputs. Their scripts (`make_rig_geometry.py`, `make_positions_shelf_A.py`) and the measurements are in the author's gitignored `data/raw/`. So the geometry can be used, but not regenerated from the measurements.
- **`tools/pipeline.py` needs the GB10 launch script**, which is not in the repo (section 6). The committed run spec holds the author's host details.
- **`uv sync` works only on linux aarch64**, and rendering is tested only on the GB10.
- **Real data are not included.** The two pilots' photos, the 2026-10-10 camera measurements and the public datasets (OCID etc.) are on the author's share, not in git. [public_data/README.md](../public_data/README.md) describes the OCID loader.
- **Physical study.** The capture tools are tested (section 7) but no physical study data beyond the two single-stop pilots exist. The physical replay is out of scope for the course.
