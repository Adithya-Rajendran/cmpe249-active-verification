# public_data/: public datasets as evidence-step development data

Development data for the evidence step (`perception/evidence.py`: match / quality / obstruction) on
PRESENT / OCCLUDED / REMOVED views, and in particular for the known failure where an occluder sits
right over the target's footprint. **This is never the held-out test.** The study's test set is our
own capture; nothing here may be used to report results.

| File | What |
| --- | --- |
| `ocid.py` | OCID -> episodes with label-derived truth (`python -m public_data.ocid`) |
| `run_evidence.py` | runs the frozen `perception.evidence.run_jobs` on ONE split of those jobs with `identity_align` (`--split dev`; val also needs `--allow-val`) |
| `score_views.py` | scores a run per view against truth, as pilot 2 was scored; writes `analysis.json` + `analysis.md` |
| `../tests/test_ocid.py` | synthetic-sequence tests (no real data, no GPU: mock encoder and detector) |

## OCID: what the data actually looks like (checked 2026-10-23)

Source: `/mnt/bench-data/datasets/cmpe249-active-verification/ocid/extracted/OCID-dataset`
(unpacked from `ocid/raw/OCID-dataset.tar.gz`). Every statement below was checked by a script over
all 178 sequences / 2390 label frames, not taken from the paper.

- **Layout.** `<set>/<surface>/<camera>/[<category>/]seqNN/{rgb,depth,label,pcd}`. Sets ARID10,
  ARID20, YCB10; surface table|floor; camera top|bottom. ARID10 and YCB10 have a category level
  (box, curved, fruits, mixed, non-fruits; cuboid, curved, mixed for YCB10), ARID20 has none.
  `seqNN` is unique within (set, surface, camera). Sequences per set and surface: ARID10 table 40,
  ARID10 floor 38, ARID20 floor 26, ARID20 table 26, YCB10 floor 24, YCB10 table 24 (89 top + 89 bottom).
- **Frames.** `result_YYYY-MM-DD-HH-MM-SS.png`; rgb/ and label/ hold the same names. Sorted names
  are strictly chronological in every sequence. 10, 11, 20 or 21 frames per sequence (62, 64, 26, 26
  sequences). All frames are 640 x 480 (W x H); rgb is 8-bit BGR, label and depth are uint16.
- **Label encoding.** `0` = no valid depth (it is *exactly* the depth==0 pixels in every first
  frame of all 178 sequences: 100 % of label-0 pixels have depth 0 and 0 % of the others have).
  The support surface has no fixed id:
  - floor sequences (87 of 88): `1` = the floor;
  - table sequences (all 90): `1` = the background (wall and floor around the table) and `2` = the
    **table**, alone in frame 0 (an empty-table frame, which is why table sequences have 11 or 21
    frames and floor sequences 10 or 20); objects are then 3, 4, ...;
  - `ARID10/floor/top/curved/seq05`: no `1` at all; `256` = the floor, objects 257, 258, ...
  The loader (0.2.0) treats as surface `1` plus every id present in frame 0 that covers >= 40 % of
  the image (`SURFACE_MIN_SHARE`). On the real data the surfaces cover >= 0.51 of frame 0 and any
  frame-0 object <= 0.12, so the rule finds exactly the cases above (MANIFEST
  `sequences_by_n_surface_ids`: 87 with one surface id, 91 with two). Surface ids are never targets
  and never count as occluders. Loader 0.1.0 got this wrong (the table and the seq05 floor were
  `obj01`), so 0.1.0 output must not be used.
- **One object added per frame, never removed.** After the surface is set aside, at most one object
  first appears per frame (the loader raises otherwise): floor sequences add one in every frame from
  frame 0, table sequences from frame 1. Objects are named by first appearance, never by id value.
- **Ids are stable within a sequence.** Over all loader candidates the share of the query mask
  outside a 5 px dilation of the reference mask is 0 at the median and 90th percentile, 0.0013 at
  the 99th, 0.0095 at the 99.9th and 0.040 at most (`NUDGE_MAX_OUTSIDE` is 0.05; these numbers are
  from the 0.1.0 build). The nudge rule excludes nothing on the real data; it is exercised only by
  the synthetic test. The one odd track found: id 13 in `ARID20/table/bottom/seq11` has areas 683,
  509, 679, 704 over frames 11 to 14, i.e. a temporary 25 % loss at frame 12 that recovers at
  frame 13 (its frame-13 mask lies inside a 5 px dilation of its frame-11 mask). It is not a move;
  the (11, 12) pair has v = 0.745 and is excluded as ambiguous.
- **A fully hidden object can vanish from the label.** One case: id 17 in `ARID20/floor/bottom/seq04`
  is missing from frames 17 to 19. The loader treats a missing label as area 0 (fully occluded if the
  lost pixels are now other objects), consistent with "never removed".
- **top/ and bottom/ are the same scene at the same moments.** For all 89 pairs the frame counts
  match, and every bottom frame i was taken after top frame i-1 and before top frame i+1 (offsets
  -28 s to +45 s, median +6 s; objects are added every 16 s or more, median 32 s). So top frame i and
  bottom frame i show the same arrangement state: they are two stops (`top` = 0, `bottom` = 1) of
  one arrangement, and the object placed k-th is the same physical object on both cameras (the loader
  names it `objKK` on both). `build()` asserts this: per arrangement, the first-appearance frames of
  the (non-surface) objects must be identical on both cameras, or the build fails. It passes on all
  89 pairs with 0.2.0 (in 0.1.0 the seq05 top floor shifted every top name by one). Pixel
  correspondence between the cameras was not checked and is not used.
- **Each camera is static within a sequence**, so reference and query frames are registered as-is.

## Episodes and truth (`ocid.py`, loader 0.2.0)

Episode = (sequence = one camera, reference frame r, target t, query frame q); the box is the bbox
of t's label mask in r. The rules and every constant are in the module docstring and frozen at
module level; they were fixed before any evidence was run on these episodes. In short:

- targets: every non-surface object (see label encoding);
- reference: t's area in r >= 0.9 x its max area in the sequence and >= 400 px; targets within
  2 px of the border in any frame are skipped;
- q after r: v = area_q / area_r; nudged (query mask leaves a 5 px dilation of the reference mask by
  more than 5 % of area_r) -> excluded; v >= 0.9 -> **visible**; v <= 0.5 and >= 80 % of the lost
  pixels now other objects -> **occluded** (occluders and their bbox IoU recorded); else excluded;
- **removed** by time reversal: q is a frame before t was placed, r a clear reference of t;
  `box_other_new_share` / `box_other_change` record whether other objects placed after q sit in the
  box in r;
- `same_footprint`: an occluder's bbox IoU with t's box >= 0.5 (the known failure);
- visible and occluded records also hold `target_lost_share` and `box_new_object_share` (share of
  the box covered in q by objects placed after r). A **visible** target can still have a new object
  resting partly on it: in the 0.2.0 build `box_new_object_share` on visible episodes has median 0,
  90th percentile 0.112, max 0.323 (occluded: median 0.543). Stratify obstruction results on visible
  by it; the truth rule itself is unchanged.

Sampling is deterministic (sha256-based rank, same_footprint occlusions first), capped per
sequence (visible 2, occluded 4, removed 2) and per target (1, 2, 1). Occlusions that meet the rule
are rare: 72 of 178 sequences have any. The split is by **arrangement** = `<set>-<surface>-seqNN`,
shared by top and bottom: val if sha256(`ocid-split-v0|<arrangement>`) / 2^64 < 0.25. Each
arrangement's split depends only on its own id.

Output (`--out`, default `/mnt/bench-data/datasets/cmpe249-active-verification/ocid/episodes/<loader_version>/`):
`annotations/<seq_key>/<frame>.json` (cmpe249.evidence-annotation.v0, one per reference frame,
stop = camera index), `jobs.json` (cmpe249.evidence-jobs.v0), `truth.json` (episode_id ->
visible|occluded|removed, as in pilot 2), `truth_detail.jsonl` (every recorded number) and
`MANIFEST.json` (version, constants, counts per label / split / set / camera, exclusions, source
path, sha256 of `ocid.py`, unmatched frame count). All paths inside are absolute.
**Episode ids are not blind.** `<seq_key>:objKK:rNN:qNN` does not spell the label, but q < r if and
only if the episode is removed (and the query file's timestamp shows the same). The frozen evidence
step never reads ids; any later consumer that must not see truth (a policy, a threshold fit) must
not parse ids or query paths.

Rebuilding into an existing output needs `--force` (`build(force=True)`): it refuses unless the
existing `MANIFEST.json` has this loader's schema, then deletes the old `annotations/` and
`evidence/` (evidence scored on the old episodes is stale).

Full-dataset truth build, loader 0.2.0 (2026-10-23, written to a scratch directory, no evidence
run; numbers from its MANIFEST.json): 178 sequences, 89 arrangements (68 dev, 21 val), 903 episodes
on 749 reference annotations: visible 356, occluded 191 (43 same_footprint), removed 356 (156 with
box_other_change). Dev: 272 / 157 / 272 (visible / occluded / removed); val: 84 / 34 / 84.
Candidates before sampling: visible 66317, occluded 1800, removed 86430. Exclusions: ambiguous
partial 12017, lost pixels not other objects 16, nudged 0, no clear reference 2 targets, border 12
targets (0.1.0 had 70, mostly the surfaces themselves). No surface is a target (max reference area
36526 px). Rebuild it to get the canonical copy under `ocid/episodes/0.2.0/`.

## Running

Truth build (no project env needed: numpy + opencv only):

```
uv run --no-project --python 3.13.5 --with numpy==2.5.3 --with opencv-python-headless==5.0.0.93 \
    python -m public_data.ocid [--root OCID-dataset] [--out DIR] [--limit-sequences N] [--force]
```

Evidence (GB10, project env; models as in `perception/evidence.py`):

```
uv run --no-sync python -m public_data.run_evidence --jobs <out>/jobs.json --split dev [--out-dir DIR] [--device cuda]
```

One split per run: `--split` is required, each job's split is read from `truth_detail.jsonl` (checked
against the arrangement lists in `MANIFEST.json`; a job without a detail record is an error), and
`--split val` is refused unless `--allow-val` is given too. By default it writes into
`<episodes>/evidence/<EVIDENCE_VERSION>/<split>/` and refuses to overwrite an existing
`evidence.jsonl` without `--force`. It writes `evidence.jsonl`, `diagnostics.jsonl` and
`evidence_run.json` (split, its job count and the job count per split in `jobs.json`, evidence
version, sha256 of `perception/evidence.py`, Params, and the sha256 of the `jobs.json`, `truth.json`,
`truth_detail.jsonl` and `MANIFEST.json` it scored plus the loader version and sha256), so a run can
be checked against the episode build it belongs to. `perception/evidence.py` is not modified.

Per-view scoring (stdlib and `verify/` only, no GPU; refuses val views without `--allow-val`, and truth
files whose sha256 differs from the run's record):

```
python -m public_data.score_views --evidence-dir <out>/evidence/<EVIDENCE_VERSION>/dev [--force]
```

Verdict per view, as pilot 2: strong_match -> present, else clear_miss -> removed, else
flagged_occluded -> occluded, else unknown (the placeholder-threshold flags in `diagnostics.jsonl`).
It writes `analysis.json` and `analysis.md` next to the evidence.

## Limits

- Truth comes from OCID's labels; label errors pass straight into truth. The masks are visible
  (modal) masks, so "occluded" means the visible area shrank under other objects, not a 3D test.
- OCID occluders are mostly objects stacked on or placed in front of the target in a static, well
  lit scene. That is narrower than our shelf (no lighting change, no blur, no camera move).
- Removals are simulated by time reversal: the "removed" query also lacks every object placed after
  it, which `box_other_change` flags when it touches the box.
