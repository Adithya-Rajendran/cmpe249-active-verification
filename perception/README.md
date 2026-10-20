# perception/

GB10-side perception front end for the study: an open-vocabulary detector that proposes
candidates, and a frozen image encoder that scores candidate crops against the reference crop.
Smoke-test code plus the v0 evidence step. No study thresholds have been set: the evidence thresholds are placeholders, and `novel_cos` was picked on pilot-1 dev data.

| File | What it does |
| --- | --- |
| `common.py` | Manifest loading, image ops (rot180, crop, blur, roll, gain, fill) applied to the image and its box together, IoU. Keeps ultralytics settings, weights and runs inside the repo, with telemetry and auto-install off. |
| `manifests/make_2026-09-26_smoke.py` | Writes `manifests/2026-09-26_smoke.json`: the image list, the synthetic variants and the hand-annotated mug boxes. |
| `smoke_test.py` | Runs YOLO-World v2 S, YOLOE-26 S (both prompted with the catalog), YOLO11 S and YOLO26 S (COCO controls) on every manifest entry. Writes detections (JSONL/CSV), a per-image summary, warm GPU latency and overlays. |
| `match_check.py` | Scores mug crops, same-size background crops, augmented reference crops and detector proposals against the reference crop with DINOv2-S and CLIP B/32. |
| `report_smoke.py` | Builds the tables and figures for `notes/2026-09-26-perception-smoke.md`. |
| `evidence.py` | **Evidence step v0.2.1** (real captures -> `verify.schema.EvidenceRecord`). For each (target, view) it registers the query onto the per-stop reference (`capture/tags.py`, identity fallback), then computes three scores. `match_score` = DINOv2-small patch nearest-neighbour similarity of the reference box against the query near the expected box. `quality` = min(sharpness vs reference, unclipped share, registered share); sharpness is measured on a ring around the box with changed pixels masked out, and is gain-normalised. `obstruction` = max(detector cue, novelty). The detector cue is the share of the box that both changed and lies under a YOLO-World v2-s non-target detection. Novelty is the share of query-box patches whose best DINOv2 cosine to the whole reference window is < `novel_cos`. Thresholds are placeholders. `novel_cos` was chosen on pilot 1 (dev), and v0.2.1 is frozen for the held-out pilot 2; see `notes/2026-10-17-first-real-pilot.md`. Input: `--jobs` JSON. Output: `evidence.jsonl` for `verify.run --manifest --evidence`. Tests: `tests/test_evidence.py`. |

Evidence step (annotation and smoke example in `data/raw/2026-10-12_pilot-prep/evidence/`):

```bash
uv run --no-sync python -m perception.evidence --jobs JOBS.json --out evidence.jsonl --diag diag.jsonl
# MANIFEST.json: a cmpe249.arrangements.v1 manifest, built with verify.manifest (save_manifest)
uv run --no-sync python -m verify.run --manifest MANIFEST.json --evidence evidence.jsonl
```

Until the far-spot stops are taped, `configs/positions.json` is the old close-spot arc, so travel/cap numbers from a real run are not meaningful (pilot 1 is a single stop).

## Setup and run (from the repo root)

```bash
uv sync                      # uv-managed CPython 3.13.5 (.python-version, same version as the Pi);
                             # torch 2.12.1+cu130 from the official PyTorch cu130 index (aarch64 wheel)
uv run python perception/manifests/make_2026-09-26_smoke.py
uv run python perception/smoke_test.py --manifest perception/manifests/2026-09-26_smoke.json \
    --out runs/2026-09-26-perception-smoke/detect
uv run python perception/match_check.py --manifest perception/manifests/2026-09-26_smoke.json \
    --detections runs/2026-09-26-perception-smoke/detect/detections.jsonl \
    --out runs/2026-09-26-perception-smoke/match
uv run python perception/report_smoke.py --run runs/2026-09-26-perception-smoke \
    --manifest perception/manifests/2026-09-26_smoke.json --figs notes/figures/2026-09-26
```

Weights download on first use into `checkpoints/` (gitignored): the four detector checkpoints,
`checkpoints/clip/ViT-B-32.pt` (YOLO-World's text encoder) and `checkpoints/mobileclip2_b.ts`
(YOLOE's text encoder). DINOv2 and the open_clip weights go to the Hugging Face cache, at the
revisions pinned in `match_check.py`. Each run's `env.json` records the code, `uv.lock` and
manifest sha256, library versions and weight hashes (`provenance`). Every `uv run` reinstalls
`nvidia-cusparselt-cu13` from cache, because that wheel's internal platform tag (`manylinux2014_sbsa`)
does not match its filename; this is harmless. The
ultralytics settings file is `checkpoints/ultralytics_config/Ultralytics/settings.json`;
`smoke_test.py` pins `sync: false` and `weights_dir: checkpoints/` in it before any prediction,
because ultralytics recreates the file with `sync: true` when it is missing.
