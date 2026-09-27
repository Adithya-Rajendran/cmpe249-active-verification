# Citation Audit

**Project:** Occlusion Aware Active Verification for Robot Scene Memory
**Audit date:** September 26, 2026
**Scope:** all 15 entries in `references.bib`, the 14-item numbered list in `docs/PROPOSAL.md` (and the identical lists in the submitted PDF and DOCX), the reference table and per-paper sections of `docs/LITERATURE_REVIEW.md`, and the links in `docs/AI_AUDIT.md`.
**Prepared with:** Claude Code (AI assistance). This audit changes nothing in the audited files. Corrections are listed at the end for the student to apply.

## Result in one paragraph

All 15 references exist. Every arXiv ID resolves to the paper named, and every title, full author list (order and count) and year matches its source. No reference is fabricated or attributed to the wrong paper. This includes the three recent preprints that most needed checking: PreSIST (2607.04057), LT-Mem (2608.19059) and RoboStream (2603.12939). Every venue that the markdown states (RSS 2022, RSS 2024, T-RO 35(1), ICRA 2016, RA-L 2025, RA-L 11(3), CVPR 2024) is confirmed by a publisher or proceedings record. The one-line characterizations match the abstracts, and the full text where I checked it. The problems are all **MINOR**:

- Four papers have a published version that the repo does not mention anywhere: DynaMem (ICRA 2025), Terashima et al. (IEEE/SICE SII 2024), SAM 2 (ICLR 2025) and GOAT-Bench (CVPR 2024). Khronos is cited as RSS 2024 in the markdown, but the bib has no venue. *(Evidence check: this said "Five papers ... that the repo does not mention", but the markdown does mention Khronos's venue.)*
- The bib has no venue for YOLO-World, even though the markdown cites CVPR 2024.
- The Lost & Found bib entry lacks volume, issue and pages.
- The literature review says Khronos was tested "on a physical robot". The paper used two robots.

**Status counts:** 8 OK, 7 MINOR, 0 MAJOR, 0 NOT FOUND.

Status definitions:
- **OK:** ID resolves; title, authors, year and stated venue match; the characterization matches. At most, optional extra metadata such as a DOI could be added.
- **MINOR:** the reference is correct, but a published venue is left out, a volume or issue is missing, the bib disagrees with the markdown, or a characterization is slightly off.
- **MAJOR:** wrong paper, wrong authors or year, a false venue claim, or a characterization that contradicts the source.
- **NOT FOUND:** the ID or DOI does not resolve.

## Audit table

Numbers are the proposal and literature-review numbers. Every numbered reference maps one-to-one to a bib key (matched by URL).

| Ref | Bib key | Status | What differs / what was checked | Evidence |
| --- | --- | --- | --- | --- |
| [1] POCD | `qian2022pocd` | OK | Title and all 6 authors match. RSS 2022 is confirmed: RSS XVIII paper 13, DOI 10.15607/RSS.2022.XVIII.013 (the bib has no DOI; optional). The LR summary checks out, including the "TorWIC" dataset name, which is in the paper body (not the abstract). | https://arxiv.org/abs/2205.01202 ; https://www.roboticsproceedings.org/rss18/p013.html |
| [2] Khronos | `schmid2024khronos` | MINOR | Title and 4 authors match. RSS 2024 in the proposal and LR is correct (RSS XX paper 81, DOI 10.15607/RSS.2024.XX.081). **The bib is `@misc` with no venue**, so it disagrees with the markdown. The LR says "Evaluated in simulation and on a physical robot", but the paper validates on **two** robots, a Clearpath Jackal and a Boston Dynamics Spot (arXiv v2 PDF, Sec. on real-world experiments). | https://arxiv.org/abs/2402.13817 ; https://www.roboticsproceedings.org/rss20/p081.html |
| [3] Bore et al. | `bore2019movable` | OK | T-RO 35(1):231-247, Feb 2019 and DOI 10.1109/TRO.2018.2876111 are confirmed by Crossref. The bib title ("...Large Three-Dimensional Maps") is the T-RO title. The arXiv preprint it links to is titled "...Large 3D Maps". That difference is expected, not an error. The 4 authors match. | https://api.crossref.org/works/10.1109/TRO.2018.2876111 ; https://arxiv.org/abs/1712.08409 |
| [4] Rosen et al. | `rosen2016lifelong` | OK | ICRA 2016, pp. 1063-1070 and the 3 authors are confirmed by Crossref. DOI 10.1109/ICRA.2016.7487237 is not in the bib (optional). The Google Research URL resolves (HTTP 200) but shows metadata only. The LR summary (generative survival model; recursive Bayesian "persistence filter" with an exact belief) matches the abstract. | https://api.crossref.org/works/10.1109/ICRA.2016.7487237 ; https://research.google/pubs/towards-lifelong-feature-based-mapping-in-semi-static-environments/ |
| [5] PreSIST | `adkins2026presist` | OK | Exists: v1 posted 4 Jul 2026, one version. The title and all 3 authors match exactly. It is a preprint (no journal-ref), which matches "July 2026 preprint". The summary (instance persistence priors plus a persistence filter; a VLM variant and a vision-only variant trained on VLM pseudo-labels; a new in-the-wild dataset) matches the abstract. | https://arxiv.org/abs/2607.04057 |
| [6] LT-Mem | `lee2026ltmem` | OK | Exists: v1 posted 19 Aug 2026. The title and all 3 authors match. The arXiv comment reads "Accepted to ... IROS 2026", which matches the LR wording "arXiv record reports IROS 2026 acceptance". No proceedings record exists yet. The summary (multi-session SLAM; overwrite/hold/multi-hypothesis policy; Live/Delta/Meta memory; LT-VQA) matches the abstract. | https://arxiv.org/abs/2608.19059 |
| [7] DynaMem | `liu2024dynamem` | MINOR | Title and all 7 authors match, as do the dates (v1 Nov 2024, v2 May 2025). **It was published at ICRA 2025** (pp. 13346-13355, DOI 10.1109/ICRA55743.2025.11127619), but the bib is `@misc` and the LR and proposal cite a preprint. The characterization is confirmed in the v2 full text: RGB-D ray casting removes outdated voxels (Sec. 3.2), abstention when the object is not found (Sec. 3.3), time-based and semantic-similarity exploration value maps (Sec. 3.4), and 70% vs. 30% for OK-Robot over 30 lab pick-and-drop queries (Sec. 4.1). "Sections 3.2-3.4" exist as cited. | https://arxiv.org/abs/2411.04999 ; https://api.crossref.org/works/10.1109/ICRA55743.2025.11127619 |
| [8] RoboStream | `huang2026robostream` | OK | Exists: v1 13 Mar 2026, v2 10 Jul 2026. The title and all 12 authors match exactly. The arXiv comment reads "Accepted by ECCV 2026", which matches the LR wording. That acceptance is author-reported only; Semantic Scholar lists arXiv only. The summary (geometric anchoring, action-triggered state graph, object permanence under occlusion, RLBench simulation plus real block building) matches the abstract. | https://arxiv.org/abs/2603.12939 |
| [9] Lost & Found | `behrens2025lostfound` | MINOR | Title and all 5 authors match; the diacritics in the bib are correct. RA-L 2025 is confirmed. **The bib lacks volume/number/pages/DOI:** RA-L 10(4):3739-3746, April 2025, DOI 10.1109/LRA.2025.3544518. The markdown transliterates "René Zurbrügg" as "Rene Zurbruegg", which is acceptable. The summary (34% / 56% lower translational/orientational error; the drawer-retrieval demo) matches the abstract. | https://arxiv.org/abs/2411.19162 ; https://api.crossref.org/works/10.1109/LRA.2025.3544518 |
| [10] Bogenberger et al. | `bogenberger2026glasses` | OK | Title and all 7 authors match. RA-L 11(3), March 2026 is confirmed by Crossref (pp. 3342-3349, DOI 10.1109/LRA.2026.3656790; optional additions to the bib). The summary and boundary are confirmed in the full text: a per-instance stationarity score, a "semantic exploration priority map" used to sample target waypoints, 95% of map changes detected, and more than 29% efficiency gain over random and patrol. | https://arxiv.org/abs/2509.19851 ; https://api.crossref.org/works/10.1109/LRA.2026.3656790 |
| [11] Terashima et al. | `terashima2023active` | MINOR | Title and all 4 authors match. **It was published at 2024 IEEE/SICE SII** (pp. 1079-1084, DOI 10.1109/SII58957.2024.10417695, same title), but the repo cites only a "July 2023 preprint". Crossref's record swaps given and family names for these authors, so do not copy Crossref's autogenerated BibTeX. The summary matches the abstract. | https://arxiv.org/abs/2307.14105 ; https://api.crossref.org/works/10.1109/SII58957.2024.10417695 |
| [12] SceneDiff | `wu2025scenediff` | OK | Title and all 7 authors match. Dates are v1 Dec 2025 and v2 Mar 2026, as the LR states. No published venue was found. The summary matches: 350 video pairs, and pose, segmentation, and semantic/appearance (DINOv3) plus geometric comparison. | https://arxiv.org/abs/2512.16908 |
| [13] YOLO-World | `cheng2024yoloworld` | MINOR | Title and all 6 authors match. The markdown's CVPR 2024 claim is confirmed (pp. 16901-16911, DOI 10.1109/CVPR52733.2024.01599). **The bib is `@misc` with no venue**, which is inconsistent with the markdown. | https://arxiv.org/abs/2401.17270 ; https://api.crossref.org/works/10.1109/CVPR52733.2024.01599 |
| [14] SAM 2 | `ravi2024sam2` | MINOR | Title and all 18 authors match in order. **It was published at ICLR 2025** (listed in the ICLR 2025 proceedings), but the bib and markdown do not mention it. The bib also drops diacritics: `Roman Radle` should be `Roman R{\"a}dle` and `Piotr Dollar` should be `Piotr Doll{\'a}r` (the arXiv spelling). The dates (Aug 2024, revised Oct 2024) match. | https://arxiv.org/abs/2408.00714 ; https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c1f6a8cbf2da59ebf2c802b4f742cd-Abstract-Conference.html |
| (unnumbered) GOAT-Bench | `khanna2024goat` | MINOR | Title and all 10 authors match. **It was published at CVPR 2024** (pp. 16373-16383, DOI 10.1109/CVPR52733.2024.01549), but the bib is `@misc`. The LR cites it only as uncounted context, which is fine. | https://arxiv.org/abs/2404.06609 ; https://api.crossref.org/works/10.1109/CVPR52733.2024.01549 |

## Cross-document consistency

These were checked by script against the repo as it stood on the audit date.

- **Proposal in-text citations vs. numbered list:** the text cites [1]-[14], and the list contains [1]-[14]. No number is cited without being listed, and no listed number goes uncited. Each in-text name matches its number: POCD [1], Khronos [2], Bore [3], persistence filters [4, 5], LT-Mem [6], DynaMem [7], RoboStream [8], Lost & Found [9], Bogenberger [10], Terashima [11], SceneDiff [12], YOLO-World [13], SAM 2 [14].
- **Literature review vs. proposal numbering:** the LR table and the 14 per-paper sections use the same number-to-URL mapping as the proposal list. The LR uses only bracket numbers 1-14.
- **Bib keys:** the markdown never cites bib keys directly. Each numbered reference maps to exactly one bib entry by URL. `khanna2024goat` is the only bib entry outside the numbered list; it matches the LR's "Additional context, not counted". The bib parses cleanly with pybtex 0.24.0 (15 entries).
- **Submitted copies:** the reference list in `CMPE249_Project_Proposal.docx` is text-identical to `PROPOSAL.md` once links are stripped. The PDF list (from `pdftotext`) reads the same. Any correction below therefore applies to all three.
- **AI_AUDIT.md:** its 9 links are all bib URLs and all resolve.
- **Recency count:** the claim of "ten first released 2024-2026, four earlier ([1], [3], [4], [11])" holds by first arXiv or publication date. Adding the published venues below does not change it, because Terashima et al. was first released in July 2023.

## Recommended corrections (exact replacement text)

None of these changes a claim or conclusion in the proposal. They make the bibliography complete and consistent. The proposal was submitted on Sep 8, so these are for `references.bib`, the literature review, and the final report. Existing bib keys can stay as they are: keys are labels, and no markdown cites them.

### C1. `references.bib`: Khronos (RSS 2024 venue is missing from the bib)

```bibtex
@inproceedings{schmid2024khronos,
  title = {Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments},
  author = {Lukas Schmid and Marcus Abate and Yun Chang and Luca Carlone},
  booktitle = {Robotics: Science and Systems (RSS)},
  year = {2024},
  doi = {10.15607/RSS.2024.XX.081},
  eprint = {2402.13817},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2402.13817}
}
```

### C2. `references.bib`: YOLO-World (CVPR 2024, which the markdown already claims)

```bibtex
@inproceedings{cheng2024yoloworld,
  title = {YOLO-World: Real-Time Open-Vocabulary Object Detection},
  author = {Tianheng Cheng and Lin Song and Yixiao Ge and Wenyu Liu and Xinggang Wang and Ying Shan},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {16901--16911},
  year = {2024},
  doi = {10.1109/CVPR52733.2024.01599},
  eprint = {2401.17270},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2401.17270}
}
```

### C3. DynaMem: published at ICRA 2025

`references.bib`:

```bibtex
@inproceedings{liu2024dynamem,
  title = {{DynaMem}: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation},
  author = {Peiqi Liu and Zhanqiu Guo and Mohit Warke and Soumith Chintala and Chris Paxton and Nur Muhammad Mahi Shafiullah and Lerrel Pinto},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages = {13346--13355},
  year = {2025},
  doi = {10.1109/ICRA55743.2025.11127619},
  eprint = {2411.04999},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2411.04999}
}
```

`docs/PROPOSAL.md`, replace the reference line:

- Old: `[7] Peiqi Liu et al. (2024). [DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation](https://arxiv.org/abs/2411.04999).`
- New: `[7] Peiqi Liu et al. (2025). [DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation](https://arxiv.org/abs/2411.04999). ICRA 2025.`

`docs/LITERATURE_REVIEW.md`, section [7]:

- Old: `**Date and status:** November 2024 preprint; revised May 2025  `
- New: `**Date and status:** November 2024 preprint; revised May 2025; ICRA 2025  `

### C4. Lost & Found: add the RA-L volume, issue and pages

`references.bib`:

```bibtex
@article{behrens2025lostfound,
  title = {Lost \& Found: Tracking Changes from Egocentric Observations in 3D Dynamic Scene Graphs},
  author = {Tjark Behrens and Ren{\'e} Zurbr{\"u}gg and Marc Pollefeys and Zuria Bauer and Hermann Blum},
  journal = {IEEE Robotics and Automation Letters},
  volume = {10},
  number = {4},
  pages = {3739--3746},
  year = {2025},
  doi = {10.1109/LRA.2025.3544518},
  eprint = {2411.19162},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2411.19162}
}
```

`docs/PROPOSAL.md`: change the end of reference [9] from `IEEE RA-L 2025.` to `IEEE RA-L 10(4), 2025.`

`docs/LITERATURE_REVIEW.md`, section [9]:

- Old: `**Date and status:** November 2024 preprint; IEEE Robotics and Automation Letters 2025  `
- New: `**Date and status:** November 2024 preprint; IEEE Robotics and Automation Letters 10(4), April 2025  `

### C5. Terashima et al.: published at IEEE/SICE SII 2024

`references.bib`:

```bibtex
@inproceedings{terashima2023active,
  title = {Active Robot Vision for Distant Object Change Detection: A Lightweight Training Simulator Inspired by Multi-Armed Bandits},
  author = {Kouki Terashima and Kanji Tanaka and Ryogo Yamamoto and Jonathan Tay Yu Liang},
  booktitle = {IEEE/SICE International Symposium on System Integration (SII)},
  pages = {1079--1084},
  year = {2024},
  doi = {10.1109/SII58957.2024.10417695},
  eprint = {2307.14105},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2307.14105}
}
```

`docs/PROPOSAL.md`, replace the reference line:

- Old: `[11] Kouki Terashima, Kanji Tanaka, Ryogo Yamamoto, Jonathan Tay Yu Liang (2023). [Active Robot Vision for Distant Object Change Detection: A Lightweight Training Simulator Inspired by Multi-Armed Bandits](https://arxiv.org/abs/2307.14105).`
- New: `[11] Kouki Terashima, Kanji Tanaka, Ryogo Yamamoto, Jonathan Tay Yu Liang (2024). [Active Robot Vision for Distant Object Change Detection: A Lightweight Training Simulator Inspired by Multi-Armed Bandits](https://arxiv.org/abs/2307.14105). IEEE/SICE SII 2024.`

`docs/LITERATURE_REVIEW.md`, section [11]:

- Old: `**Date and status:** July 2023 preprint  `
- New: `**Date and status:** July 2023 preprint; revised October 2023; IEEE/SICE SII 2024  `

(Keep the LR table's Year column at 2023, or change it to 2024 and add "(arXiv 2023)". Either way, the "four earlier works" scope sentence stays correct because it counts first-release dates.)

### C6. SAM 2: published at ICLR 2025; restore diacritics

`references.bib`:

```bibtex
@inproceedings{ravi2024sam2,
  title = {{SAM} 2: Segment Anything in Images and Videos},
  author = {Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman R{\"a}dle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Doll{\'a}r and Christoph Feichtenhofer},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2025},
  eprint = {2408.00714},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2408.00714}
}
```

`docs/PROPOSAL.md`, replace the reference line:

- Old: `[14] Nikhila Ravi et al. (2024). [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714).`
- New: `[14] Nikhila Ravi et al. (2025). [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714). ICLR 2025.`

`docs/LITERATURE_REVIEW.md`, section [14]:

- Old: `**Date and status:** August 2024; revised October 2024  `
- New: `**Date and status:** August 2024; revised October 2024; ICLR 2025  `

### C7. GOAT-Bench: published at CVPR 2024

`references.bib`:

```bibtex
@inproceedings{khanna2024goat,
  title = {GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation},
  author = {Mukul Khanna and Ram Ramrakhya and Gunjan Chhablani and Sriram Yenamandra and Theophile Gervet and Matthew Chang and Zsolt Kira and Devendra Singh Chaplot and Dhruv Batra and Roozbeh Mottaghi},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {16373--16383},
  year = {2024},
  doi = {10.1109/CVPR52733.2024.01549},
  eprint = {2404.06609},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2404.06609}
}
```

`docs/LITERATURE_REVIEW.md`, "Additional context": change `(Khanna et al., 2024)` to `(Khanna et al., CVPR 2024)`.

### C8. `docs/LITERATURE_REVIEW.md`, section [2] Khronos: robot count

- Old: `**Summary:** Separates fast tracking of short-term dynamics from slower reasoning about long-term change inside one dense spatio-temporal map. Evaluated in simulation and on a physical robot.`
- New: `**Summary:** Separates fast tracking of short-term dynamics from slower reasoning about long-term change inside one dense spatio-temporal map. Evaluated in simulation and in real-world runs on two robots (a Clearpath Jackal and a Boston Dynamics Spot).`

### Optional enrichments (entries already rated OK)

Add these lines inside the existing entries:

- `qian2022pocd`: `doi = {10.15607/RSS.2022.XVIII.013},`
- `rosen2016lifelong`: `doi = {10.1109/ICRA.2016.7487237},`
- `bogenberger2026glasses`: `pages = {3342--3349},` and `doi = {10.1109/LRA.2026.3656790},`
- `lee2026ltmem`: `note = {Accepted to IROS 2026 (per arXiv comment)},`
- `huang2026robostream`: `note = {Accepted to ECCV 2026 (per arXiv comment)},`

## What this audit did not establish

- **ECCV 2026 (RoboStream) and IROS 2026 (LT-Mem):** these acceptances come only from the authors' arXiv comments. No proceedings record was available to confirm them. The LR's wording ("arXiv record reports ...") is already accurate. Recheck before the final report.
- **Venue discovery:** published venues were found through the Semantic Scholar batch API and then confirmed against Crossref, the RSS proceedings, or the ICLR proceedings. DBLP returned a bot challenge and OpenReview returned HTTP 403, so neither was used. A venue for SceneDiff, PreSIST, LT-Mem or RoboStream that post-dates these indexes could still be missing.
- **Depth of reading:** characterizations were checked against every abstract. Full-text checks covered DynaMem, Bogenberger et al., POCD, Khronos and SceneDiff. Other body-text details, such as the exact missed-detection model in Rosen et al., were not read end to end.

## Reproduction

These commands were run on the GB10 on 2026-09-26. The working files are in the session scratchpad, not the repo.

```bash
# arXiv abs pages (the export.arxiv.org API returned HTTP 406 to curl, so abs pages were parsed instead)
for id in 2512.16908 2607.04057 2608.19059 2402.13817 2603.12939 2404.06609 2401.17270 \
          2408.00714 2205.01202 1712.08409 2411.04999 2411.19162 2509.19851 2307.14105; do
  curl -sS -A 'Mozilla/5.0' -o abs_$id.html https://arxiv.org/abs/$id   # all 14 returned HTTP 200
done   # then parse <meta name="citation_title|citation_author|citation_date">, comments, jref, submission history

# Publisher metadata
curl -sS https://api.crossref.org/works/10.1109/TRO.2018.2876111        # likewise for LRA.2025.3544518, LRA.2026.3656790,
                                                                       # ICRA.2016.7487237, ICRA55743.2025.11127619,
                                                                       # SII58957.2024.10417695, CVPR52733.2024.01549, CVPR52733.2024.01599
curl -sS -X POST -H 'Content-Type: application/json' -d @ids.json \
  'https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,venue,year,journal,externalIds,abstract'
curl -sS https://www.roboticsproceedings.org/rss18/p013.html           # POCD, RSS 2022
curl -sS https://www.roboticsproceedings.org/rss20/p081.html           # Khronos, RSS 2024
curl -sS https://proceedings.iclr.cc/paper_files/paper/2025            # SAM 2 listed

# Full-text spot checks
curl -sS -L -o dynamem.pdf https://arxiv.org/pdf/2411.04999v2 && pdftotext -layout dynamem.pdf dynamem.txt
curl -sS -L -o pocd.pdf    https://arxiv.org/pdf/2205.01202v2 && pdftotext pocd.pdf pocd.txt
curl -sS -L -o khronos.pdf https://arxiv.org/pdf/2402.13817v2 && pdftotext khronos.pdf khronos.txt
curl -sS -L https://arxiv.org/html/2509.19851v2 ; curl -sS -L https://arxiv.org/html/2512.16908v2

# Bib parse and author-list comparison (inline script: pybtex parses the bib, and each author list
# is compared with the arXiv citation_author list, exactly and with diacritics folded)
~/.local/bin/uv run --no-project --with pybtex==0.24.0 python - < compare_authors_inline.py
```

## Evidence check (September 26, 2026, 13:15–13:25)

A second pass tried to refute this audit.

- **MAJOR / NOT FOUND entries.** There are none to re-check (0 of each).
- **Three OK entries, chosen at random** with `python3 -c "import random; random.seed(20260926); print(random.sample([...the 8 OK refs...], 3))"`, which picked [3] Bore, [1] POCD and [10] Bogenberger. Each was re-checked independently:
  - **[3] Bore et al.** Crossref `10.1109/TRO.2018.2876111` gives T-RO 35(1):231-247, February 2019, with the same four authors. The arXiv 1712.08409 abs page gives the title "... Large 3D Maps" (v1 22 Dec 2017) and the same authors. OK is confirmed.
  - **[1] POCD.** The arXiv 2205.01202 abs page gives six authors in the bib order, with the comment "Published in Robotics: Science and Systems (RSS) 2022". roboticsproceedings.org/rss18/p013 gives DOI 10.15607/RSS.2022.XVIII.013. "TorWIC" appears in the full text (pdftotext) but not in the abstract. OK is confirmed.
  - **[10] Bogenberger et al.** Crossref `10.1109/LRA.2026.3656790` gives RA-L 11(3):3342-3349, March 2026, with the same seven authors. The arXiv abs page gives v1 24 Sep 2025 and v2 14 Jan 2026, and the abstract states "95% of map changes" and "more than 29%". OK is confirmed.
- **Three MINOR entries, spot-checked on Crossref.** The DynaMem ICRA 2025 pages (13346-13355), the Terashima SII 2024 pages (1079-1084, with Crossref's given and family names swapped) and Lost & Found RA-L 10(4):3739-3746 all match.
- **Other checks.**
  - The 9 unique links in `AI_AUDIT.md` are all bib URLs.
  - The status counts (8 OK, 7 MINOR) match the table.
- **Correction.** One summary sentence was corrected inline: four papers, not five, have a published version the repo never mentions.
