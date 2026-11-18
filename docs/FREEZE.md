# FREEZE: the simulation study's test (freeze id `simulation-freeze-1`)

**Frozen on 2026-11-06, after the researcher's final review of the preregistration.** The researcher confirmed every
open slot as recommended, with the review's corrections (decision D1(b)). Before any test arrangement is generated,
the researcher sends the freeze commit id and this file's sha256 to the instructor as an outside timestamp (decision
D2(a)). The branch stays local.

The freeze is the commit that adds this file. The machine-readable record is the one ` ```json ` fenced block at the
end, written by `tools/make_freeze.py --freeze-id simulation-freeze-1`.
- The record lists every file the test path runs or reads, with its sha256.
- It names the commit those files come from: the parent of the freeze commit.

No seed of a test block (>= 100000) is generated, rendered or scored before the freeze commit. There was one exception
while the gate was being built. On 2026-11-05 a unit test, under a mocked freeze, generated the shelf test arrangement
of slot 0 in a temporary directory and deleted it. It was not rendered, scored or viewed. The gate was repaired the
same evening, and the seed schedule is kept (`notes/2026-11-06-test-seed-incident.md`).

The gate:
- Generation (`sim/testsplit.py`) and episodes (`sim/episodes.py --freeze`) check this committed file with
  `sim/freeze.py`.
- Rendering (`sim/render_test.py`) runs on the GB10, which has no repository. It accepts an arrangement only when the
  FREEZE.md sha256 the arrangement records equals that of the run dir's copy of this file.

Any change after the freeze is a dated addendum (SIM_PREREGISTRATION §13), written in a new file,
`docs/FREEZE_ADDENDA.md`. It is never written into this file or into any file the record lists: those are checked
byte for byte whenever a test seed opens.

## 1. What is frozen

**Analysis rules.**
- `docs/SIM_PREREGISTRATION.md` as confirmed in the review of 2026-11-06.
- Amendment 2 of `notes/2026-11-04-registration-repair-plan.md`.
- The pinned designs `notes/2026-11-05-side-descriptor.md` and `notes/2026-11-05-ranker-a-design.md`.
- The development record:
  - `notes/2026-11-06-tabletop-dev49-transfer.md`;
  - `notes/2026-11-06-pooled-calibration.md`;
  - `notes/2026-11-06-test-seed-incident.md`.

**Environment.**
- Jetson and GB10 evidence: Python 3.13.5, with `uv.lock` and `pyproject.toml` at their recorded sha256.
- Each evidence job records the GB10's uv.lock sha256 and installed package versions in its provenance.

**Renderer.**
- Isaac Sim 6.2.0.0 in its own Python 3.12 environment on the GB10, installed as in `notes/2026-11-02-simulation.md`:
  `isaacsim[all,extscache]==6.2.0.0` and `tinyobjloader==2.0.0rc13`, NVIDIA driver 595.99.02.
- `sim/isaac_batch.py` f617ffb1... and `sim/isaac_rig.py` 385bf6fa..., the files that rendered the tabletop
  development sets. Defaults: RaytracedLighting, 16 rt subframes, 200 settle updates, L1 and L2.
- The test launcher is `sim/run_render_test.sh`. It is `sim/run_render.sh` with `render_test.py` and the FREEZE.md
  copy in place of `isaac_batch.py`.
- Render-settings hash, recorded in every arrangement's `labels.json`:
  - tabletop random arrangements 09244c0c... (all 60 development arrangements);
  - shelf a1faaceb... (all 64 shelf development arrangements, rendered with the earlier isaac_batch 95390851...).
- The current renderer reproduces the shelf development renders within run-to-run noise (section 4).
- What actually ran is checked twice:
  - `render_test.py` prints the sha256 of itself, of the isaac_batch.py and isaac_rig.py it loaded, and of the
    FREEZE.md copy, on the render log's first line;
  - every `labels.json` records the renderer's two hashes, and `tools/test_validity.py` compares them with this
    record.

**Generator.** `sim/arrange.py` e08da916..., `sim/rig.py` and `sim/tabletop.py`, unchanged since the development sets.
Each development arrangement's sha256 covers the generator's own hash, and the pin tests reproduce them.
- Configs: `configs/sim/arrange_shelf_v3.json` 352b36ef... and `configs/sim/arrange_tabletop_v1.json` 17c39bf1...
- Test blocks:
  - shelf: primary seeds from 100000, permutation seed 20261104, target seed 20261106, reserves 101000-101019 (S8),
    101100-101129 (M16), 101200-101209 (IND28), 101300-101319 (white_box) and 101400-101419 (ycb_box);
  - tabletop: primary seeds from 110000, permutation seed 20261108, target seed 20261110, reserves 111000-111419 by
    family in the same pattern.
- Quotas per environment: S8 20, M16 15 left and 15 right, IND28 10, white_box 20, ycb_box 20, so 100 test
  arrangements.
- Retry maximum: 20.
- Revisit noise: +-10 mm and +-0.5 deg per axis, at most 400 redraws.
- The slot lists (family, M16 side and target per slot) are frozen as a deterministic function of the frozen config
  and generator code (SIM_PREREGISTRATION §12).
  - For the study they are written to `index/slots.json` when `sim/testsplit.py` generates after the freeze, and they
    are reported with the results.
  - The unit tests of the incident record wrote the shelf test slot list to temporary directories, unread.
  - The test suite computes the slot lists in memory. Its synthetic-seed fixtures reuse the shelf list's family, side
    and target sequence, with synthetic seeds and so synthetic placements (the incident record).

**Objects, targets and scene geometry** are recorded in the frozen configs:
- `configs/sim/arrange_*.json`:
  - the YCB-Video model list with its licence note (YCB-Video / YCB terms; renders in a paper carry attribution);
  - the target ids 4, 9, 14, 21 and 6;
  - the per-slot target assignment, with the excluded (target, family) pairs and their tested-search counts
    (SIM_PREREGISTRATION §2);
  - the scene block, with `tags` false, so the scenes are tag-free;
- `configs/rig_geometry.json`, `configs/sim/tabletop_geometry.json` and `configs/sim/ycbv_hulls.json`;
- the positions files.

The renderer settings enter the render-settings hash above:
- the renderer, subframes, settle and capture updates and lighting presets;
- the tabletop's spec lighting, materials and texture sha256.

**Strings inside frozen files that this file supersedes.** These were written during development and cannot change
without moving their pinned hashes:
- the generator configs' `slots.note` ("DEV and not frozen", "no test arrangement is generated");
- `arrange_shelf_v3.json`'s `excluded_note`;
- `frozen/frozen_candidate.json`'s `status` ("not frozen until ... FREEZE.md");
- `sim/arrange.py`'s comments and help texts that say test seeds are never generated before the freeze, and Amendment
  2's "no seed >= 100000 is generated, rendered or scored before FREEZE.md". Each was true when written; the
  incident record qualifies them.

**Episodes and labels.** `sim/episodes.py` and `sim/check_batch.py`. The geometric labels, visible fractions and
no-diagnostic flags are outputs of the frozen code after rendering, not inputs (SIM_PREREGISTRATION §12).

**Evidence.**
- C0 as pinned by Amendment 2: `perception/evidence_v03.py --align image` eeae5273..., with the frozen
  `perception/evidence.py` d0e0927f..., `verify/schema.py` fc348327... and `verify/stopping.py` 35575cae...
- Models:
  - DINOv2-small `facebook/dinov2-small` at revision ed25f3a31f01632728cabb09d1542f84ab7b0056;
  - YOLO-World v2-s `yolov8s-worldv2.pt` 9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792;
  - the CLIP text encoder `clip/ViT-B-32.pt` 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af;
  - the prompt list in `perception/evidence.py`.
- The left/right descriptor: `perception/side_descriptor.py` 1c987512...
- The evaluation-only harness: `perception/registration_v032.py --candidate C0` 8a1e2245..., which records the warp
  for plane geometry.
- Pipeline specs: `configs/pipeline/shelf_test.json` and `tabletop_test.json`. Their evidence steps are those of the
  development sets.

**Calibration and models.** These are byte copies in `frozen/`, described in `frozen/README.md`.
- `frozen/frozen_candidate.json` 3a45b23c... (with its sidecar): tau_q 0.3, tau_occ 0.4, tau_match 0.7, k 2, lam 0.
  It is eligible with complete grouped CV over 120 development arrangements.
- `frozen/model_A.json` 297e11af... (C 10) and `frozen/model_A-nocue.json` eab90ded... (C 1).

**Policies.** S, F, R (20 seeds), G, G-lookahead, G-myopic, Dir, A, A-nocue and FULL, plus the -fcm variants.
- Budgets 0-4, with the primary at 3. Caps: uncapped and 1, 2, 3, 4 and 6 arc steps.
- The start stop is P3. The positions files `configs/sim/positions_shelf_A.json` and
  `configs/sim/positions_tabletop_A.json` give the F order and the stop geometry.

**Analysis code.**
- `verify/run.py` (real-v7) produces:
  - the primary contrast, and the declared secondary and sensitivity runs;
  - the family breakdowns at the occluder-family and slot-class levels;
  - the gate-off ablation;
  - the tau_match sweep: the grid below 0.7 and the fixed extension 0.75-0.9;
  - the section-10 diagnostics;
  - the metric-level zero-width flags and the interval note (section 5), added at the final review. Every estimate
    and interval of real-v6 is unchanged.

  The definitions are in SIM_PREREGISTRATION §8-§10.
- `analysis/*`.
- `tools/registration_compare.py --report-only` for the view-level reporting, and `tools/test_validity.py` for the
  validity checks.

## 2. Runbook (only after this file is committed)

**Precondition for step 1.** The researcher has confirmed that the timestamp (the freeze commit id and this file's
sha256) was sent to the instructor. No test arrangement is generated before that confirmation.

Run everything from a clean checkout at the freeze commit.
- First run the test suite with the slow goldens: `CMPE249_GOLDEN_SLOW=1 UV -m unittest discover -s tests -t .`
  (docs/REPRODUCE.md section 3.1). Then check that `git status` shows no tracked change.
- Tag the freeze commit with an annotated tag: `git tag -a simulation-freeze-1 -m "simulation freeze 1"`.
- SET below stands for `<share>/sim/<env>_test`, the rendered set. EV stands for its evidence job directory, named in
  `runs/<env>_test/evidence_provenance.json`.
- Create `runs/<date>-test/<env>/` before steps 4-6 (`mkdir -p`).
- Long steps run under `tools/pipeline.py` and `tools/watch.py` (Monitor).
- UV below stands for `uv run --offline --no-project --python 3.13.5 --with numpy==2.5.3 --with
  opencv-python-headless==5.0.0.93 python`.

1. **Generate** (Jetson; writes to the shared volume):

   ```
   UV -m sim.testsplit --config configs/sim/arrange_shelf_v3.json --freeze docs/FREEZE.md --slots all \
     --out <share>/sim/shelf_test_geometry/arr
   (the same with configs/sim/arrange_tabletop_v1.json -> <share>/sim/tabletop_test_geometry/arr)
   ```

   Read only the counts, the generation log's retries and replacements, and the index.
2. **Prepare the GB10 test run dir.** Make a new directory beside the development run dirs, because the renderer
   reads `../assets/ycbv/models`. Never use a development run dir itself: those hold older renderer copies. Copy into
   it:
   - from the freeze commit: `sim/isaac_batch.py`, `sim/isaac_rig.py`, `sim/render_test.py` and
     `sim/run_render_test.sh`;
   - `docs/FREEZE.md` as `FREEZE.md`;
   - the `tags/` directory (read only for scenes with tags; these scenes have none);
   - `textures/tabletop_light_oak_v1.png`, whose sha256 isaac_batch checks against the spec before Isaac starts;
   - the generated arrangements into `arr_shelf_test/` and `arr_tabletop_test/`.

   Then:
   - Check the copied code's sha256 against the record.
   - FREEZE.md cannot list itself. Check its copy's sha256 against `sha256sum docs/FREEZE.md` at the freeze commit,
     and against the `freeze.sha256` that the generated arrangements record.
   - Check each copied arrangement's sha256 against `index/index.json`.
   - Write `configs/pipeline/remote.test.local.json` (untracked) as `remote.local.json`, with `run_dir` set to the
     new directory.
3. **Render, episodes, check, evidence:**

   ```
   python3 tools/pipeline.py configs/pipeline/shelf_test.json --launch
   python3 tools/pipeline.py configs/pipeline/tabletop_test.json --launch
   ```

   Run them one after the other. Each watcher prints only alerts and the terminal line.
4. **Validity checks** (section 3), on operational outputs only, once per environment:

   ```
   ssh <GB10> 'cd <checkpoints> && sha256sum yolov8s-worldv2.pt clip/ViT-B-32.pt' > runs/<date>-test/gb10_checkpoints.sha256
   ssh <GB10> 'head -1 <run dir>/<env>_test.log' > runs/<date>-test/<env>/render_log_head.txt
   UV tools/test_validity.py --freeze docs/FREEZE.md --env <env> --geometry <share>/sim/<env>_test_geometry/arr \
     --set SET --evidence-provenance runs/<env>_test/evidence_provenance.json \
     --gb10-checkpoints runs/<date>-test/gb10_checkpoints.sha256 \
     --render-log runs/<date>-test/<env>/render_log_head.txt --out runs/<date>-test/<env>/validity.json
   ```

   Also check the pipeline's terminal lines:
   - render rc 0, the expected count, and no label-sanity flag;
   - the check stage rc 0, including table clipping;
   - the evidence records complete.
5. **Decisions** (Jetson), once per environment.

   ```
   UV -m verify.run --manifest SET/manifest.json --evidence EV/v03_image_refL1.jsonl \
     --view-labels SET/view_labels.json --positions configs/sim/positions_<env>_A.json \
     --candidate frozen/frozen_candidate.json --side EV/side_refL1.jsonl \
     --model A=frozen/model_A.json --model A-nocue=frozen/model_A-nocue.json \
     --k-sensitivity 3 --lam-sensitivity 0.7,0.9,1.2 --gate-off-ablation --tau-match-sweep \
     --bootstrap-B 10000 --model-id v03_image --tag <env>-test --out runs/<date>-test/<env>
   ```

   Then rerun step 4's validity check with `--verify-run runs/<date>-test/<env>` added. It confirms that the run used
   the frozen candidate, models, positions, thresholds, analysis code, and this set's evidence and descriptors. Only
   then are the results opened.
6. **Views** (Amendment 2 reporting), once per environment:

   ```
   UV tools/registration_compare.py --set SET --jobs jobs_refL1.json --c0 EV/c0_refL1.diag.jsonl \
     --cand EV/c0_refL1.diag.jsonl --name C0 --c0-timing EV/c0_refL1.timing.json \
     --original-c0 EV/v03_image_refL1.diag.jsonl --thresholds frozen/frozen_candidate.json --report-only \
     --data-label "TEST data" --out-dir runs/<date>-test/<env>/views
   ```

7. **Report.** Write a tracked results note with every declared comparison and the counts that §11 requires:
   - retries and reserve replacements per family, from `index/generation_log.json`;
   - revisit redraws, from the arrangement records;
   - re-renders and exclusions, from the render events. Exclusions should be none.

   Then the venue recommendation.

## 3. Validity requirements (checked on operational outputs before any outcome is read)

`tools/test_validity.py` checks requirements 1, 2 (its hashes), 4 and 5:
- requirement 4 covers the shipped code, the GB10 environment (uv.lock and every installed package's version), the
  record counts and the harness C0's reproduction of the evidence of record. The reproduction is an equality test of
  two evidence files, and no score is read as an outcome;
- the pipeline's own terminal lines give the rest (rc, counts, label sanity, table clipping).

The requirements:
1. Every test arrangement was written by `sim/testsplit.py` under this freeze and records this file's sha256:
   - its seed is in the config's test block, one arrangement per slot;
   - its arrangement_sha256 recomputes;
   - each file matches the generation index.
2. Render:
   - the expected 100 arrangements per environment, rc 0, no label-sanity flag, and tabletop table clipping passed;
   - frame-check redraws are logged;
   - every render is of its generated file, made by the frozen isaac_batch.py and isaac_rig.py, with the development
     render settings of its environment.
3. Episodes were built with `--freeze`, and the check stage exited 0.
4. Evidence:
   - every output has its expected record count, from the frozen commit's code;
   - the GB10 environment has the frozen uv.lock, and the detector and CLIP weights have their listed sha256;
   - the harness C0 reproduces the evidence of record (largest difference 0, equal refusals);
   - the render-settings hash equals the development value for that environment.
5. verify.run:
   - it accepts the frozen candidate and both models (sha256, features, thresholds);
   - the positions equal the frozen configs, and its analysis code is the frozen code (real-v7);
   - every declared setting was on, and the run completes.

If a requirement fails, stop, record what failed, and bring it to the researcher before any outcome is read.
- A mechanical render failure is re-rendered from the same seed and logged (§11).
- Nothing is excluded for its outcome.

**Never rebase, amend or reset across the freeze commit.** The record names its commit by full id, and the gate
requires that commit to be an ancestor of HEAD.

**A change to a frozen file after the freeze closes the gate by design.**
- `require_freeze`, the evidence stage's freeze check and `tools/test_validity.py` all refuse once any file in the
  record differs.
- If a bug in frozen code is found during the test, stop. Record it in `docs/FREEZE_ADDENDA.md` (what, why, what had
  been seen), and bring it to the researcher. The researcher decides the fix and how the test continues, for example
  a superseding freeze record with a new id and how already generated arrangements carry over.
- Nothing is changed silently, and frozen files are never edited in place.

## 4. Renderer check before the freeze (development arrangements)

The shelf development set was rendered with an earlier `isaac_batch.py` (95390851...).
- The tabletop work changed the renderer to f617ffb1... (d24311a). That change states that the shelf path makes the
  same pxr calls; the shelf goldens and a pinned call-log hash cover that.
- The shelf test renders use the current renderer.
- So, before the freeze, three shelf development arrangements were re-rendered with it on the GB10: SD1000 (M16-L),
  SD1024 (white_box) and SD1047 (M16-R). They ran through `tools/pipeline.py`, 127.9 s median each.
- They were compared with their development renders (`runs/2026-11-06-rendercheck/compare.py` and `compare.json`,
  untracked).

The results:
- **Images.** 120 captures; none is pixel-identical, as expected, since renders are not bit-reproducible.
  - Mean absolute grey difference: 0.082 / 0.127 / 0.759 (min / median / max).
  - Share of pixels more than 8 grey apart: median 0.0005 %, max 0.017 %.
  - The recorded run-to-run noise of an unchanged scene at the same pose is 0.09 / 0.11 / 0.57 grey, with a median of
    0.004 % of pixels beyond 8 grey (`notes/2026-11-03-dev-batch-check.md`).
- **Render settings.** The render-settings hash is equal on all three (a1faaceb...).
- **Labels.** Every visible fraction, diagnostic flag and occlusion pattern is identical. The only label difference is
  the L1/L2 instance-mask agreement diagnostic, 0-2 pixels per capture, which varies with render noise.

The current renderer reproduces the shelf development renders within run-to-run render noise.

## 5. Reporting

- **Primary first, whatever it shows.** A - G at b = 3, uncapped, per environment and lighting. It is inconclusive when
  the 95 % paired cluster-bootstrap interval contains 0, and inconclusive is never equivalence.
- **Intervals are pointwise and unadjusted.**
  - Every interval covers one metric of one policy or pair, in one environment, lighting and setting.
  - None is adjusted for the number of comparisons, and no simultaneous coverage is claimed.
  - Each conclusion is stated for its environment, lighting and metric. There is no omnibus claim that A wins, or
    loses, in some cell or in every cell.
- **Zero-width intervals** are reported as verify.run flags them:
  - `zero_width`, a `degenerate_note`, and the discordant episode and arrangement counts of a contrast;
  - `same_decisions` and an `effort_note` for policies that decide alike with different effort, such as A and Dir on
    development data.

  The raw estimates and the declared bootstrap are kept. A zero-width interval is never presented as evidence of zero
  risk, of a perfect rate or of equivalence. Zero observed errors are reported as a count.
- **Then the declared secondary contrasts:**
  - A - G-lookahead across the caps;
  - A - Dir (explanatory, post-hoc development origin);
  - A - A-nocue, A - F and A - R;
  - G - F and G - R;
  - each multi-view policy against S;
  - G - G-lookahead and G - G-myopic;
  - X vs X-fcm (H1);
  - the gate-off ablation for F and R;
  - the tau_match risk-coverage sweep, with the fixed extension 0.75-0.9 as secondary reporting;
  - A - G on common targets;
  - the family breakdowns with target composition (H3). The occluder-family level, with M16's sides pooled, is
    secondary reporting. The slot-class rows (M16-left and M16-right) and the side-stratified resampling are kept
    beside it;
  - the k and lam sensitivity runs.
- **Errors, unknowns and effort:**
  - both error directions (false removals and false presents), unknowns, mean views and travel;
  - the view-level refusals, per-view error directions and plane geometry, by environment and lighting.
- **Mechanism wording.**
  - A's agreement with Dir, and A-nocue's with G, are reported as observed decision agreement at the stated
    settings, with their travel and views.
  - A-nocue drops four features together, so no effect is attributed to the side descriptor alone.
- **Development estimates stay conditional.** The precision estimate of SIM_PREREGISTRATION §14 rests on the sparse
  development disagreements and is not a power guarantee.
- **Limitations reported with the results, not retuned:**
  - the shelf L1 subgroup's gate false-clear rate (21 / 176) against the pooled constraint;
  - the tabletop L2 design-case gate failures (TDES0-TDES3);
  - C0's failed tabletop refusal criterion;
  - the test-seed incident.
- **Kept apart:** the first 11 tabletop arrangements, the remaining 49 and the test.
- **No promotion.** No cap, subgroup or sensitivity setting is promoted to the primary. The conditional development
  diagnostics stay labelled as development.

```json
{
 "commit": "dccf8d2ce20ae9054eb4385d84e2ce6b56f97e3b",
 "files": {
  "analysis/__init__.py": "bb749568a772826d5c693a1d16df2857a839e4761f800d607ffdf85cbcb58445",
  "analysis/bootstrap.py": "5268ff40e17af4ecdedc2e4d0c137510c8700bf2096b856bf80397e43a411021",
  "analysis/common_target.py": "fbe9566423e2d861a950dc70bdf94fa8afa1c23b3e1648bee4e5d2bea363ca09",
  "analysis/intervals.py": "a88c725e83848522e92c9d0973d930430a0cf246c74c6e2e1e8ea688dd532fe0",
  "analysis/metrics.py": "d01c87ac8e188442e4f12fe03cb6e0cb8a32f8ace9796b13d6e0b068d441c638",
  "configs/pipeline/shelf_test.json": "71a771f40c891ab1b1b817d866f679eb0bd5bfa505f941466ba8f1d9fc4285ea",
  "configs/pipeline/tabletop_dev49_evidence.json": "0f070e421ffd774bd57475ec95ceed64763178b9602463b24bc605b4d7a446b9",
  "configs/pipeline/tabletop_test.json": "9e1a89f061e72dc712956789dba8f10f613dec4076b37b71833cffd7c5fb79ee",
  "configs/rig_geometry.json": "5ead0a14500f3e4f510a7a8f929f764ebe53276e2d40a84e498d312b52728295",
  "configs/sim/arrange_shelf_v3.json": "352b36efeda5cd4534c6c6d9fdb284fb7f2784d41c3af5e981253a1c9f3b746f",
  "configs/sim/arrange_tabletop_v1.json": "17c39bf15cdb5f43b985f3126a27c3072317aa718d2349d3691337c4208c03b1",
  "configs/sim/positions_shelf_A.json": "695dbd3339039679df03755c2613ddb84b9c09906b22763c363b6219e0190dcf",
  "configs/sim/positions_tabletop_A.json": "1a74b258b28b0961b65c15840f53ccecce9aa0b658299020c18817230a383a19",
  "configs/sim/tabletop_dev49.json": "a82d841f6c2e5b7e7c8730dbed0f2c77f5273a87e907989d41f110e04f5a21e4",
  "configs/sim/tabletop_geometry.json": "76317422198c9e25ae3d0af8502551a729a19651a349993ae99b3428ff55e249",
  "configs/sim/ycbv_hulls.json": "9d7004d0be6226a13ca219324eae9c21f35ef20ba6e25ccfa89e1c3725bb7e3c",
  "docs/EXPERIMENT_PLAN.md": "60cfbfe8d3f1aa04abe0afbc5a22d10d2ba65a7c6c9b16b51401a5878d1b6226",
  "docs/SIMULATION_STUDY.md": "fb229f89bde47d9757e3147100d5f28aa3e2ecc693ba690763ede42a1de3d9fa",
  "docs/SIM_PREREGISTRATION.md": "9a04a76351219ff481540ea59a7d4a4418790b4ba4010d48d71e61e47396e7f7",
  "frozen/README.md": "c2ab10f1f51e6d49e16d171d0b959ccb10f9d368e47dd6403cc2870d7483487d",
  "frozen/frozen_candidate.json": "3a45b23cbe82856660a9c1e34de3074238b8d8204512663d972a215939e9c26d",
  "frozen/frozen_candidate.json.sha256": "d0ffbf42e39800011c702934390fce9f64b3d7a854b12bb8e45db99ca591f262",
  "frozen/model_A-nocue.json": "eab90ded6364d689d59104070b52fd47d851818ca9aebd1d718ee3c8c93370c8",
  "frozen/model_A.json": "297e11af452d169525dd60c974291e5969e7c124194a809398f7519663381a19",
  "notes/2026-11-02-simulation.md": "0a4548cc013c975b6d487ecd45a17a812d02715fb9c8080bc09bb58712769c6c",
  "notes/2026-11-04-registration-repair-plan.md": "1fec7363604637c536d32e32c35250172cb8489394226446b54d20f0ea133325",
  "notes/2026-11-05-ranker-a-design.md": "efd3be5c9683d0f1febd42c50b8d2febd076f746a19916ee8f221dc647f4e69f",
  "notes/2026-11-05-side-descriptor.md": "eb4a5a2b0a8e0ce4701addfe6beeb767d338c12d4ce07c5fd620ac4e5e6d7e48",
  "notes/2026-11-06-pooled-calibration.md": "008e58d905d4edd59faf348d94a040d07b6bb4c8934f1d26397a6d8afd0235fd",
  "notes/2026-11-06-tabletop-dev49-transfer.md": "2d566d124816e83e3deb247fcf6ab1bf1e62b2debf24a209fef7016952afd4b7",
  "notes/2026-11-06-test-seed-incident.md": "352d0c6e38a4632252a981201bfa1b452fa7fe413af31ea89580a3f143aa9bfb",
  "perception/common.py": "1b2ad0bc55489b75e856b6a80ea1c607f84ba50f1543d4db259faa876c4e0891",
  "perception/evidence.py": "d0e0927ff9d36ee2a8027cc8050f1801e372faea1e12cd98c513345d258d46cf",
  "perception/evidence_v03.py": "eeae5273db57647735dc8cc35f25f7dcc2c6a740b69c7d4c0a1ea4025901fd97",
  "perception/registration_v032.py": "8a1e224509211e65e38d5af13775796d4c8b007ea1666024f788906a6bb44371",
  "perception/side_descriptor.py": "1c987512681940758932ccad1adb043f67a46269360d62fc3cf4eaeaa36e6900",
  "pyproject.toml": "245a585041f832a3c34f335b0bf9cd5427568c4950497daf29b6529bc6c7ee5d",
  "sim/__init__.py": "c02465fca7e74052b05f49b30e8c89b7c97670622ee4b7d36f784b613c130003",
  "sim/alignment.py": "289f74bdda014310542b7c04e82bca5dce91df176786636e9da7609492bf1262",
  "sim/arrange.py": "e08da91658e2864a96a6735c3a78e49e5056d1b9e91b4798aa9a571fa910fd85",
  "sim/check_batch.py": "6803aae12ad8bea0a43131a3455aea8b4b42d36a29cddc19a80514e0fb262fe4",
  "sim/episodes.py": "92bdbed2241503d5cfef2284d35a1f6e93453f4a084db13b9f2a70a2c90167b0",
  "sim/freeze.py": "008735a016f737512aaf39062abe4e48a21519157ec59aca0efb229736ce1b1e",
  "sim/isaac_batch.py": "f617ffb1f5579132146dd85525c4975b0e5b7eb490001cfb6f9fbde68429f4ab",
  "sim/isaac_rig.py": "385bf6fab96c87cd9573bcf51151be96c89bc770af502b6b358ba47559545b30",
  "sim/render_test.py": "c8544ac08e32b9ea7c49b6819501c0a523b5aff97caa097ce658f46733117483",
  "sim/rig.py": "a68dc6c04baf2eeb12c34688e67fe3c953b421caf51132bf394750c0d524660d",
  "sim/run_render.sh": "fe5f688bd276872a3e5bafbc2c3d14537afc88ba1c3f369ef1e4a79d202f0cdb",
  "sim/run_render_test.sh": "31713bce0e62324d6a093e1f5f20de085abfef06c04d668eeb28043bca3435e4",
  "sim/tabletop.py": "7af7a3936315d663056f0a92f6e4032d3a54d90c5e6740909f9bd7284d6cda85",
  "sim/testsplit.py": "758581bf0d1e384ea26e9c0b07869abf3f2d29b1a3169fd73dec6ca8786e9370",
  "sim/true_align_evidence.py": "153d24f667bbab82d166899ca16197a7f37eb7e60bc17dac354df0d168149a45",
  "sim/wood_texture.py": "fcea1a7748498fff20009b172b680d135606ba2481d4d73fee5110f8b5248664",
  "tools/evidence_summary.py": "ad83f3510e1752682bfe7c4aef76712fa5bb280362e9d4cc3c7cf07cf970f75a",
  "tools/merge_dev_sets.py": "7d87165f5f52479868364402033b2717773de6e6dcdc82a5265ef74cf2d01013",
  "tools/pipeline.py": "01d8fd382df6bfac633e3c6e3b6071929247f70ef6aa052c889bd3687865a497",
  "tools/provenance.py": "d3bf4e9085065099341efb8958e9d1ad7c54885a943c9db57399c5bd81872051",
  "tools/registration_compare.py": "c3d9d2fd2073a0dc3e8bee22f434e253ba30f4ede793e8cd773d7041cb3a7827",
  "tools/test_validity.py": "ba774e572f4a5eb82471978de4afdb53bb14cab3acd66bafa6a70836990d1ff5",
  "tools/watch.py": "caebccb618a94577743ed636940b8ef0f553be5ed6ab364817caaf9291f36d05",
  "uv.lock": "6f6743a67be3f88867000cda6cdbc351463f6ecbb6e4e5a599ef6b7e5e14d053",
  "verify/__init__.py": "5bce172a7af9dd4ba5a084ea6ada5c6ce6437be42825d6791671886dd5498c45",
  "verify/lamgrid.py": "cb17bbb652f7d04ccd1972f9935b3d716dcee80902d95a8ac7f646d09feaeeba",
  "verify/manifest.py": "35d3a8f9f616a200fb2449a12e7ea01ab23730d368f8cc3db0ce32e0bcf997b5",
  "verify/policies.py": "32d1413890cc0247b67c00f71f10b773ffd75bff2910404dc7bc818720686fec",
  "verify/ranker.py": "ef1a84382a7abf132ff5b8e4fd05d1b8cb16e9a2db29e4887d73cb5b3bafd0d3",
  "verify/replay.py": "1afb168b5f06ba927f0a30f0838f25bf1cb95c4323e34aa941803dfacb706ace",
  "verify/run.py": "514150597dbb5195cb9f966e8f1a8b02d4a69a3391affe5491c41a998b7a96b2",
  "verify/schema.py": "fc34832788f1e1b947f32c663b789b20e5449b9095cf2065f241ac2cbb739b8f",
  "verify/select.py": "035a4d01cda59ced5bce71f46c42df94b8a2ec09e56e76c53e8fc8d431ff1617",
  "verify/side.py": "32076e9e9a31469bc5e43b76c53f1a6ec70433fe3445a2a99b9c0115b4282423",
  "verify/stopping.py": "35575caea0c62a56dee08881d9b3add72c2182733f9a6e50487b344ca91a4765",
  "verify/synthetic.py": "816f5e04f7964853081584d651585482483425969bab77793ac58f81ca51b268"
 },
 "freeze_id": "simulation-freeze-1",
 "schema": "cmpe249.freeze.v1"
}
```
