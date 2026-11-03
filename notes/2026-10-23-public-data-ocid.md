# Public datasets, the OCID loader, and evidence v0.2.1 on OCID (2026-10-21 to 10-23)

**Role of public data.** Development data and secondary checks for the evidence step only. The study's held-out test is still our own rig captures. Nothing here can be reported as a study result, and val splits stay untouched until a decision needs them.

## Datasets on the share

Location: `/mnt/bench-data/datasets/cmpe249-active-verification/`, an NFS share mounted at the same path on the Jetson and the GB10. The `README.md` there gives each dataset's source, licence and citation; `manifests.json` gives every URL, size and checksum.

| Dataset | Size | Why it is here | Licence |
|---|---|---|---|
| OCID | 6.4 GB | tabletop clutter built up one object at a time, masks every frame | CC BY 4.0 |
| SceneDiff | 12 GB | real before/after videos, added / removed / moved labels | MIT (kitchen subset: HD-EPIC terms) |
| ObChange | 109 GB | robot revisits of 5 rooms, YCB objects moved / removed | CC BY 4.0 |
| 3RScan | 66 GB | 478 rooms re-scanned 2–12 times, posed frames | CC BY-NC-SA 4.0 + terms of use (form signed by the student) |
| YCB-Video, T-LESS, HOPE | 16 / 11 / 6 GB | object models and real occluded views (BOP format) | MIT / CC BY 4.0 / CC BY-NC-SA 4.0 |
| GraspNet-1Billion | ~151 GB | 256 posed views per tabletop scene | CC BY-NC-SA 4.0 |
| OSD-Amodal | 0.3 GB | amodal (hidden-but-present) masks | no licence stated; academic use |

All files were verified: a published checksum where there is one, otherwise exact size and a readable zip directory. 3RScan publishes no checksums, so its 10,075 files are checked only by download success. Nothing is unpacked except OCID.

## OCID loader (`public_data/ocid.py` 0.2.0)

- **Data checked by script**, not from the paper (details in `public_data/README.md`). The table is a labelled surface, not an object. Object ids are stable within a sequence. The top and bottom cameras record the same scene at the same moments, so each arrangement has **two stops**.
- **Truth from the masks:**
  - *visible*: the target is ≥ 90 % as visible as in the reference;
  - *occluded*: ≤ 50 % visible, with the lost pixels covered by later objects;
  - *removed*: the sequence played backwards, so a frame from before the object was placed is the query.

  Episodes in between are excluded. The thresholds were set from label statistics before any evidence ran.
- **Episodes:** 903 from 89 arrangements, split by arrangement into dev (701; 68 arrangements) and val (202; 21). 43 occluded episodes have an occluder over the target's outline (box IoU ≥ 0.5), the pilot 2 clock case.
- **Caveats:**
  - Time reversal gives "removed" cases with no occluder, while the occluded cases have one. That breaks the study's matched-occluder rule, so OCID is for the per-view evidence step only, never for the policy comparison.
  - The frame order also reveals the label; the evidence code never sees timestamps.
  - Images are 640×480, much coarser than the rig's 2304×1296.
- **Review:** an adversarial review found two real bugs in 0.1.0. The table or floor was being used as a target, and one sequence's top/bottom object names were off by one. Both are fixed in 0.2.0.

## Frozen evidence v0.2.1 on OCID dev (2026-10-23)

Run on the GB10 with identity alignment (each OCID camera is static). Scored per view as in pilot 2 (k = 1), at the placeholder thresholds. Evidence and analysis: `.../ocid/episodes/0.2.0/evidence/0.2.1/dev/` (`analysis.md`).

| | result |
|---|---|
| False removals | 7 / 429 present targets (0 / 272 visible; 7 / 157 occluded) |
| Removal recall | 150 / 272 |
| Coverage (present or removed decided) | 634 / 701 |

- **All 7 false removals are occluded targets** with match 0.53–0.60 (just under 0.6) where neither obstruction cue fired (detector ≤ 0.044, novelty ≤ 0.25): an occluder that is neither detected nor novel, the limit the evidence code already documents.
- **Missed removals are the bigger weakness:** 63 / 272 removed targets were called present. This rises as the target fills less of its box: 16 % at fill ≥ 0.75, 23 % at 0.5–0.75, 38 % below 0.5. That fits the match comparing the whole box, so unchanged background inside it keeps matching. **This is a correlation; the cause is untested.**
- **The same-footprint mechanism from pilot 2 recurs:** in 10 of the 36 such views, a detection was dropped as "the target itself". Here those views kept a high match, so none became a false removal.
- **All 35 unknowns are removed views with low sharpness.** That is an artifact of time reversal: neighbours placed later are missing from the query, and their lost texture reads as blur.
- **OCID scores differ from the rig:** visible match p05 0.81 against a rig minimum of 0.89, so OCID cannot set the rig's thresholds.

**What it suggests for v0.3** (not yet built or tested): restrict the match to the target region rather than the whole box, and decide "is this detection the target?" by appearance, not overlap alone.
