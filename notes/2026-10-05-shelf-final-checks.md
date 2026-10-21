# Final shelf pose: checks and geometry (2026-10-05)

Workflow wf_ef645745-caf: Pi checks, geometry re-fit at the actual height, and a skeptic audit. The audit's corrections come first, then its consolidated summary. Evidence: data/raw/2026-10-05_shelf-final/.

**Corrections**

Base path: `data/raw/2026-10-05_shelf-final/`. I recomputed the registration and the log stats from `pi/all_logs_pi.jsonl`. The camera was not used.

PI report:
1. **"No scale change" holds only against `ref_lp1.45`.** Affine scale for frame e against that reference is 0.9997/0.9996. Against the 1.10 reference it is 1.0015/1.0019, and against 1.05 it is 1.0018/1.0023. The LensPosition change itself does this (focus breathing): `ref_lp1.10` against `ref_lp1.45` has scale 1.0019/1.0023. That moves the frame corners by 2.4–2.7 px and the centre by only 0.17 px. My recompute matches `pi/lp1.05/lp105_results.json` (ref145_to_ref105 scale 0.9979/0.9974).
   - The PI's "1.10→1.45 moves dx −0.68, dy −0.30..−0.38 px" is translation-only and understates the effect at the corners about 4×.
   - The 3.39/3.55 px figures against the 1.10/1.05 references are camera shift plus breathing, not shift alone. The camera's own rigid shift is the same-LensPosition number: 2.80 px (dx 0.50, dy 2.75), roll 0.046°, corner displacement 2.1–4.3 px.
   - Frame e and the geometry edges were taken at 1.45. Against `ref_final` they are off by up to 5.3 px at the corners (recomputed: 4.63/5.34/1.79/3.57 px at the corners, 3.26 px at the centre).
2. **The focus "unexplained shift" is mostly explained by geometry.**
   - The modelled slant from P3 to the mug is 95.0 cm, i.e. 1.05 D (`geometry/arc_plan_final.txt`, R=90 [fit] P3). The measured study-mug peak is 1.00 (`pi/focus2_results.json`, peak_lp_parabolic 1.0017). So the dioptre scale agrees with geometry to about 5%.
   - The old 1.45 lock is the anomaly: 1/1.45 = 69 cm, which is closer than the 74 cm to the shelf front face.
   - The PI's "geometry predicts ~0.1 D change" should read "the current peak matches geometry; the earlier peak did not".
3. **1.10 was not "at least as sharp as 1.05 for 5 of 6 props".** Four props are strictly sharper: plush 1.036, look-alike mug 1.019, Switch 1.023, front edge 1.061. The study mug is 0.9999 (a tie) and the figurines 0.989 (`pi/lp1.05/lp105_results.json` focus_bracket).
4. **The first bracket's best value (1.25) was its lowest tested value**, so it was an edge result and not a peak (`pi/final_results.json` focus, lps 1.25–1.65).
5. **Ranges stated slightly wrong:**
   - AE colour temperature before lock was 3432–3438 K, not 3435–3438; the third AE read 3432 (`pi/all_logs_pi.jsonl` ae_state_for_lock).
   - Sensor temperature over the 96 frames was 38–46 °C; the whole log is 36–46 °C (recomputed). Lux 201.41–203.40 (0.99%), 96 frames over 8.28 min: confirmed.
6. **Glare component bbox:** the 2574/648 px component is at x844–1002, **y370–405** (`pi/lp1.10/highlight_components.json`). y362 belongs to a separate 108 px component with no clipped pixels. The threshold is max channel ≥240, not grey level.
7. **Whole camera session was 9.7 min**, and each locked run lasted at most 55 s. The stability numbers say nothing about drift beyond one minute.

GEO report:
8. **The occluder set mixes mug depths.**
   - S9/S10 and M18 need the mug at z=26 with the occluder at the front lip.
   - L28 "CpHHH" needs the mug at z=20 (`geometry/occ_set_check_n33.txt`).
   - Moving the mug between conditions changes mug pixel size and viewpoint, which is a confound. At z=26, L28 gives CCHHH with P5 at 0.03 for an 8 cm mug and CCHHp (P5 0.06) for a 9 cm mug.
   - IND28 works at both depths, with its back face 6 cm in front of the mug axis.
9. **The handle advice conflicts with the deep layout.** "Point the handle straight back" needs 26 + 4 + 2.7 = 32.7 cm of depth. The fitted depth is 29.16 cm (28.07–30.01) and the student's figure is 31 cm. The handle should point toward the camera instead, which keeps it inside the body silhouette from P3. This orientation was not modelled.
10. **Some quoted numbers mix nominal and f+5% rows.** The P1/P5 front-frame margin is 274 px nominal (258 is the f+5% value). P2 is 248/183 nominal (249/184 is f+5%). The tolerance values are 158/116/−105 px, not 159/117/−104 (`geometry/arc_plan_final.txt` lines 21–26).
11. **Two summary ranges are too narrow.** Distance is 70.58–78.23 cm and height 13.73–15.63 cm (`geometry/pose_fit_final.txt`), not 70.6–77.9 / 13.75–15.63. The tilt note in `positions_final_draft.json` gives 6.03–6.68°; the fit range is 6.03–6.79°.
12. **"The 120 Hz flicker fix is still unverified" is superseded.** The Pi bursts now test it (item B below).

**Consolidated summary**

**A. Lock** (`pi/ref_final.json`; `ref_final.jpg` is byte-identical to `pi/lp1.10/ref_lp1.10.jpg`, verified with cmp):
- Mode 2304:1296:10:P, ScalerCrop [0,0,4608,2592]
- AE and AWB off
- Exposure 25000 µs (sensor reports 24984)
- AnalogueGain 1.381 (reports 1.3801), DigitalGain 1.0008
- ColourGains [1.5753, 2.5111]
- AfMode Manual, LensPosition 1.10, approached from 0.3
- FrameDurationLimits 60000 µs
- At capture: 203.1 lux, 3433 K, 44 °C

**B. Flicker** (`pi/final_results.json` flicker): row-band median, with the detrended value in brackets.

| Exposure (µs) | Gain | Banding | Note |
|---|---|---|---|
| 10000 | 3.45 | 1.640% (1.557%) | positive control |
| 22392 | 1.54 | 1.098% (0.949%) | earlier lock |
| 8333 | 4.14 | 0.472% (0.214%) | gain over 4 |
| 16667 | 2.07 | 0.473% (0.212%) | |
| 25000 | 1.38 | 0.475% (0.213%) | frame-mean CV 0.014%, pixel noise 0.562 |
| 33333 | 1.12 | 0.440% (0.198%) | gain floor, mean brightness 138.5 vs about 133 |

The choice of 25000 µs holds. Each setting had only one burst.

**C. Stability** (ECC, frames about 5 s apart): every run stays within 0.27 px over 55 s, mostly dy. The previous pose reached 0.079 px max.

| Lock | Frames | Median | Max | Consecutive median / max | Source |
|---|---|---|---|---|---|
| 1.45 | 10 | 0.148 px | 0.221 px | 0.043 px | `pi/final_results.json` |
| 1.05 | 10 | 0.092 px | 0.151 px | | `pi/lp1.05/lp105_results.json` |
| 1.10 | 12 over 55 s | 0.166 px | 0.267 px | 0.054 / 0.160 px | `pi/lp1.10/lp110_results.json` |

During the 1.10 run the sensor warmed from 38 to 44 °C. Hysteresis from approach direction is under 0.1 px of translation.

**D. Focus** (`pi/focus2_results.json`):
- Fitted peaks: study mug 1.00, plush 1.12, look-alike mug 1.05, figurines 1.01, Switch 1.10, front edge 1.12.
- The worst-prop score at each value peaks at 1.05 (0.976) and is 0.964 at 1.15.
- In the same session, 1.45 relative to 1.10 was 0.76–0.87 (`pi/lp1.10/lp110_results.json`).
- Geometry cross-check: mug slant 95.0 cm = 1.05 D. In the deep layout the slant is about 101 cm (~0.99 D), and 1.10 is still about 0.96–0.99 of peak on the sweep curve.
- **A LensPosition change rescales the image by about 0.2%**, so once the reference is frozen, LensPosition must never change.

**E. Saturation at 1.10** (`pi/lp1.10/lp110_results.json`, `pi/lp1.10/highlight_components.json`):

| Region | Result |
|---|---|
| Mug-compartment interior | 0.0044% ≥250, 27 px fully clipped (median), p99 197 |
| Study mug box | 0.092% ≥250; one glint at x622–627, y690–697 (28 px ≥240, 16 clipped) |
| Look-alike mug | 23 clipped px at x1336–1341, y307–314 |
| Plush | max 194, no clipping |
| Middle-shelf front edge | max 222, no clipping |
| Glare on upper board front face | x844–1002, y370–405: 2574 px ≥240, 648 clipped |
| Full frame | 0.016% ≥250 |

The glare sits just above the opening's top edge, which is measured at y393–408 (`geometry/check_projection.txt`).

**F. Pose** (`geometry/pose_fit_final.txt`, from frame e, f=1774 px, not calibrated, nothing tape-measured):
- Distance to the front plane 74.26 cm (70.58–78.23)
- Lateral +0.14 cm (−0.28..+0.51)
- Lens 14.55 cm above the mug-shelf top (13.73–15.63)
- Pitch 6.35° down (6.03–6.79), yaw 1.85° right (1.66–2.05), roll −0.49°
- Fit RMS 1.32 px, max 4.83 px over 200 points
- Inner depth from the back-joint fit 29.16 cm (28.07–30.01), against the student's 31 cm
- Since frame e the camera has rigidly shifted 2.80 px (about 0.09°) with 0.046° of roll. That is negligible for geometry but above the proposed 2 px cold-start limit.

**G. Arc** (R=90, centred 15.5 cm behind the front, ±45/±22.5; `geometry/floor_template.txt`). Lens over the floor; lateral is from the compartment centre. I recomputed these analytically and they match.

| Stop | Lateral (cm) | Out from front (cm) |
|---|---|---|
| P1 | −63.64 | 48.14 |
| P2 | −34.44 | 67.65 |
| P3 | 0 | 74.50 |
| P4 | +34.44 | 67.65 |
| P5 | +63.64 | 48.14 |

- Chord 35.12 cm, arc step 35.343 cm, P1 to P5 127.28 cm. P1/P5 are 22.3 cm beyond the unit's outer side.
- Nominal margins: opening 199/248/203 px and front frame 274/183/156 px (P1/P2/P3); mug 149–151 px.
- Yaw limits at P3 for a 50 px margin: ±3.6°. Lateral placement error ±3 cm leaves 116 px.
- Caps 35.34 / 70.69 / 106.03 / 141.37 / 212.06 cm; `positions_final_draft.json` loads (`geometry/verify_positions_final.json`).
- The current spot works as P3 if a tape check confirms 74.5 ± 1 cm.

**H. Occluders** (`geometry/occ_extend.txt`, `occ_set_check_n33.txt`, `occ_parking*.txt`):
- A 12 cm height hides the mug top from P3 (sight line crosses at 9.35–9.82 cm).
- With the mug at z=20, no configuration gives CCHCC, CCHHC or CHHCC.
- **Recommended: use one mug depth, z=26**, with the handle pointing toward the camera:

| Occluder | Placement | Pattern | Kept under ±1 cm |
|---|---|---|---|
| S 9–10 cm | front lip, back face z=2, centred | CCHCC | 0.75 |
| M 18 cm | front lip, shifted ±4 | CCHHC / CHHCC | 0.67 |
| L 28 cm | front lip, shifted +10 | CCHHH, P5 0.03–0.06 (marginal) | |
| IND 28 cm | back face z=20 | HHHHH | |

- Parking for the visible condition, beside the mug: occluder centre at least 11 / 15 / 20 cm from the mug centre for widths 9–10 / 18 / 28 cm. Only the 9–10 cm occluder can park at the front lip (|x| ≥ 31 cm).
- Mark every occluder position on the shelf.

**I. Risks / next steps**
1. Measure floor space at P1/P5 against the right-hand wall; the ±30 fallback has no usable occluder patterns.
2. Tape-measure the inner depth: z=26 needs at least 30 cm, and the fit says 29.16 cm. If it is shorter, the fallback is z=24 with gap 22, where CCHCC was not found.
3. Tape-check P3 (74.5 cm), the lens height and the chords; f is uncalibrated (±5% means ±3.7 cm in distance).
4. Freeze LensPosition 1.10 and re-make every mask and reference from `ref_final`, not frame e (up to 5.3 px off at the corners).
5. Run a cold-start re-check against `ref_final` (limit ≤2 px) and a 10-minute locked drift series with sensor temperature logged.
6. Mask the glare at x844–1002, y370–405, or move the lamp.
7. G2's "nothing changeable visible" rule still fails: the figurines, the look-alike mug and the spray can are in compartment #4.
8. `configs/positions.json` (R=45, ±60/±30) and the PREREGISTRATION §6 caps (23.56 cm per step) need updating if this draft is adopted. I did not edit them.