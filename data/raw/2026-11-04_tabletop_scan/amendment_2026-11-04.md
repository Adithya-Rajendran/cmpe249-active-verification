# Board height: post-hoc development amendment (2026-11-04)

**This is a post-hoc development amendment by the researcher, made after inspecting the scan results.** It is not the
outcome of the predeclared rule. Machine-readable record: `amendment_2026-11-04.json` (hashes of every file below).

**Original outcome (unchanged, the record):** "no height satisfies (a) and (b)" (`decision.json`, commit a63ad64; no H_T,
no IND28 width). `decision.json`, `results.json`, `cells.jsonl`, `scan_plan.json` and `base_config.json` are not modified;
the original statistical output stays in `decision.json`.

**Amendment:** H_T = 0.18 m for S8, M16 and IND28. IND28 = tabletop variant 0.36 x 0.18 x 0.005 m. ycb_box: all four
choices the scan kept. Parking: the parked occluder's base centre in frame at >= 1 stop. This is the rule the scan already
used; full footprint on the table, clearance at every stop, shadow screen and area-light ray check unchanged.

**Why**
- The earlier rule conflated identity support with placement rates. Rule (b) compared each target with its best cell
  across different IND28 widths. Soup and meat peak at 0.28 m wide; brick, tuna and mug peak at 0.36 m.
- All five targets are admitted at 0.18 m (S8, M16-L/R, IND28 at both widths).
- Within width 0.36 m, 0.18 m has at least as many observed IND28 successes as every other height, for every target.
- 0.36 m is chosen for the design-case geometry (design mug at 0.18 m: 0.28 m cannot hide it, nominal margin -48.8 px;
  0.36 m fits with 3.4 px at gap 0.005 m, shift -0.02 m; fixed TDES3 revisit poses keep analytic coverage >= 0.998).
  It is not chosen because its total (283) exceeds 0.28 m's (279).

**Statistics disclosure**
- The Fisher comparisons treat shared-seed outcomes as independent; a paired comparison would be needed.
- A non-significant difference is not equivalence.
- Each cell was compared with a maximum selected from the same scan.
- This decision rests on observed support and geometry; the original statistical output is retained.

**Evidence:** the researcher's review `runs/2026-11-03-tabletop-scan-review/{audit,ind_design_margin,ind_design_revisit}.json`
(not committed; sha256 in the JSON).

**Fixed before the fresh seeds:** committed before the class matrix runs on the pre-planned probe seeds 7200-7299
(`scan_plan.json` `after_the_scan.class_matrix`, unchanged rule: >= 1 success in 100 tries = in the class's target list).
`make_tabletop_config.py final --amendment amendment_2026-11-04.json` takes H_T and the IND28 width from this file; without
it, a decision with no height is still refused.
