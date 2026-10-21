# Bookshelf as the study scene: assessment (2026-10-05, reviewed)

## 1. Verdict: possibly, but not from where the camera is now, and not until the floor space is measured

The bookshelf and the closet could serve as the study scene, but the current camera position can't. In this frame the lighting is steady and the image doesn't move. The shelves are too far away, though: about 1.35–1.7 m slant distance. At that distance the mug is only about 90 px wide, and the lens is at its focus end stop. Our geometry model (not a test) predicts that a ±40 cm side-to-side move from here gives about ±17° of view change. That is too little to reveal most hidden targets, so G4 would likely fail.

The fix is to pick one compartment and move the camera onto a close arc in front of it. Whether that arc fits in this closet is the deciding question. In the photo, the right wall/pillar and boxes sit tight against the shelf's right side, and the left shelving sits tight against its left. The proposed end stops (56.6 cm to each side of centre, 43.6 cm out) may collide with them. The proposal's scope already says "one tabletop or shelf", and D2b already lists a shelf-style layout as an alternative. So this is a declared layout change, not a protocol break. It must go into the PREREGISTRATION draft before any pilot data.

## 2. What the measurements say

**Lighting** (Pi, 17 frames over 240 s, `pi/log.jsonl`):
- Lux 49.15–50.80 (std 0.37); 50.39–50.80 after the first frame. Mean luma std 0.32. Colour temperature 3530–3555 K. 0.01–0.04% saturated pixels.
- Auto exposure stayed at 49995 µs with gain 2.52–2.56. It looks pinned at about 50 ms, so the scene is light-limited.
- The sensor warmed from 36 to 50 °C and was still rising, so this is not a warmed-up steady state.
- Flicker tests (gain 8): frame-mean CV 0.027% at 1/100 s and 0.013% at 1/120 s; row-band std 0.007–0.008 DN. The 1/120 s test cannot see 120 Hz flicker, so the 1/100 s result is the meaningful one. No flicker was detected at this sensitivity.
- The light is steady but dim. Locked frames show 0.73 DN temporal noise after the ISP.

**Stability** (all settings locked, 10 frames over 45 s, `locked_ecc.json`): ECC shift was 0.021 px median and 0.042 px max. This covers only 45 s with nothing moving. It says nothing about the earlier 19 px jump, about returning to the same pose, about footsteps or floor flex, or about thermal drift.

**Focus** (`focus_results.json`, `pi/focus*.npy`):
- In both sweeps, every prop was sharpest at LensPosition 0.0 D, the end stop, and sharpness was still rising there.
- The current 0.284 D gives about 0.51–0.70 of the sharpness at 0.0.
- The camera's own autofocus settled at 0.19 D and 0.30 D, so autofocus can't be trusted here.
- The two sweeps differ by 13–37% in measured sharpness, even though both approached 0.0 from above. The cause is unknown: lens repeatability, ISP state, or settling time (frames were about 1 s apart). Lens repeatability is not established.
- The dioptre scale is unreliable: best focus at about 1.4–1.7 m sits at or beyond the nominal infinity setting. Check it against a tape measure.

**Distance** (`fit/pose_from_frame.txt`, `fit/shelf_geometry.txt`):
- From the shelf geometry: pitch about 32° down; mug-shelf front edge about 116 cm out horizontally and 72 cm below the lens (about 136 cm slant).
- From object size (Switch 271 px, mug 90 px): about 156–167 cm.
- Focal length is f ≈ 1774 px from the 66° HFOV spec; the lens/pixel spec gives 1693 px, so allow about ±5%.
- Either way, the shelves are about 2.3–2.8x farther than the planned 59 cm, at about 1.1 px/mm against the planned 3.0.

**Prop sizes** (by eye, ±5 px): mug 90×93 (112 wide with handle), plush 113×159, Switch 271×113, chess clock 158×92, egg 79×94, figurines about 90–100 px each, spray can 68×263, keyboard 376×135, boots 275×248. On the tabletop rig the mug would have been about 256 px.

**Detection** (one frame, run on the GB10, not the Hailo; hand boxes ±10 px; `perception/detections.json`):
- **Mug:** 0.10–0.36 at full frame with 640 input. 0.68–0.84 at 1280. 0.61–0.84 on a shelf crop at 640, and 0.85–0.94 on a crop at 1280.
- **Keyboard:** 0.66–0.93 at 1280 or on a crop, but 0.37–0.55 at full-frame 640.
- **Spray can:** detected as "bottle" at 0.43–0.92.
- **Boots:** "shoe" or "work boots" at 0.27–0.76 from the open-vocab models only. The closed-set models don't detect them.
- **Chess clock and egg:** only YOLOE finds them (clock 0.36–0.94, egg 0.31–0.64).
- **Switch:** every model misses it. The best was 0.25; COCO models call it "cell phone" at up to 0.90.
- **Plush:** unreliable. 0.28–0.38 as a plush; YOLOE calls it "mug" (0.51) or "anime figure" (0.64–0.91).
- **Under pasted occlusion:** with 50% of the mug covered, scores are 0.00–0.66 depending on model and side. This is the regime the study needs.
- **Not tested:** imgsz 1280 on the Hailo-8L. Hailo models are normally compiled at 640, so the compartment crop is the realistic route.

**Re-identification** (`perception/embeddings.json`):
- Against unlike objects, DINOv2 separates the mug: its closest other crop scores 0.226, while its own shifted or rescaled crops score 0.931–0.977.
- **This is not the test that matters.** The only look-alike pair present, left vs right boot, scores 0.943 (DINOv2) and 0.951 (CLIP), which overlaps the mug's self-similarity.
- Under 50% occlusion, the mug's self-similarity falls to 0.685–0.74 (DINOv2) and 0.83–0.84 (CLIP).
- Telling two look-alike mugs apart with whole-crop embeddings is therefore at real risk and must be measured with a real second mug.
- CLIP is weaker in general. Mug vs empty-shelf crops score 0.58–0.62, which suggests the white background dominates the crop; that cause is a guess.

## 3. Required changes (most important first)

1. **Measure the floor first, then re-rig close to one compartment.**
   - Planned arc (model only): radius 80 cm centred on the compartment's middle, stops at −45, −22.5, 0, +22.5, +45°.
   - The end stops sit 56.6 cm to each side and 43.6 cm out from the shelf front; the middle stop is 67.0 cm out. Adjacent stops are 31.2 cm apart (chord).
   - Lens 25–30 cm above that shelf's floor, pitched about 15° down.
   - The model gives a mug about 182 px wide. The smallest frame margin is 36 px at the end stops, which is within the ±15% geometry error. G2 requires everything fully in frame from all 5 stops, so verify this with real test frames.
   - From the photo, the right pillar/boxes and the left shelving may block the end stops. If they do, the fallbacks are weaker. An R=70 arc at ±40° clips the compartment in the model. A 55 cm rail at ±30.5° gives less parallax and needs code changes.
   - Prefer a compartment at a height a short, stiff post can reach.
2. **Mark the stops rigidly.** G2 requires median ≤5 px and max ≤10 px over 10 returns to P3. At about 83 cm, 5 px is about 2.3 mm of translation or about 0.16° of yaw.
   - Tape on the floor won't hold that. Use a plywood or MDF floor template with 5 L-stops that fix both position and yaw.
   - Use one heavy base with a fixed-height post and a rigid wedge at about 15°.
   - Tape-measure the arcs. Test the returns before anything else.
   - Exposures are 50 ms, so keep footsteps and floor flex away during shots.
3. **Focus.** At the new pose, sweep focus at P1 and P5 for the front and back of the compartment. Use a consistent approach direction, and repeat the sweep to measure how well the lens repeats. Do the tape check of the dioptre scale. There is no lock value for the new pose yet.
4. **Lighting.**
   - Add a fixed LED lamp aimed into the compartment from the side. L1 = ceiling light + lamp, L2 = ceiling light only (the lamp is switched off, not moved).
   - Watch for glare and clipping on the white semi-gloss shelf.
   - Stand at a fixed spot outside the frame during shots.
   - Re-run the G2 warm-up and the 10-minute drift test (the sensor was still warming).
   - Check gain stays ≤4 in the shaded compartment.
5. **Background.**
   - Drape the left shelving, boxes and yoga mat with matte cloth, or freeze them and record that.
   - Restrict detection and matching to the compartment.
   - The textureless white interior gives ECC little to lock onto. Use a static textured mask outside the compartment, or add a fixed textured marker.
   - The shelves above and below must be cleared or frozen. From a close, low camera, the top shelf and its contents are out of view anyway.
6. **Props.**
   - Targets: the study mug, plus the chess clock, egg or figurines if we use YOLOE. Don't use the plush or the Switch as prompt-detected targets.
   - **Add a second look-alike mug.** Without it the identity test can't run. Test DINOv2/CLIP separation on the real pair before relying on it (see §2).
   - **Occluders:** 10–18 cm wide and at least about 14 cm tall, placed 10–14 cm in front of the target centre, with the target about 18 cm back. On a 26 cm shelf that leaves about 4–8 cm for the occluder's thickness. Upright books, thin boards or flat boxes fit. Boots (too deep) and a standing Switch (about 10 cm tall) don't fit this geometry. The spray can is too narrow and the keyboard lies flat.
   - **Indistinguishable subset:** a 30–40 cm wide board or row of books close in front of the target (model).
   - **Include off-centre targets** so the first diagnostic stop varies (G4 iii). In the model, many off-centre settings give only 1 clear stop, so check G4 (ii) per arrangement.
   - **Detection:** use a crop of the compartment, which is Hailo-compatible. Imgsz 1280 has only been tested on the GB10.

## 4. Doc and code changes (list only, nothing edited)

- `configs/positions.json`:
  - radius_cm 80.0; azimuths ±45 / ±22.5 / 0; start P3; F order P1, P2, P4, P5.
  - `height_cm` and `tilt_deg` should be redefined relative to the shelf floor.
  - Draft: `data/raw/2026-10-05_shelf-eval/fit/positions_shelf_draft.json`. Verified: it loads with `verify.schema.Positions` (`from_dict`), and `g_lam_breakpoints` gives the same 6 breakpoints as the current file.
- PREREGISTRATION:
  - D2b becomes the shelf layout, with an entry in the §0 deviations table.
  - The §6 travel caps in cm become 31.42 / 62.83 / 94.25 / 125.66 / 188.50 (arc lengths). The caps in steps and the lambda grid are unchanged.
  - D3 L1/L2 are redefined for the closet.
  - The G2 framing and background clauses become compartment-based.
  - The Appendix C occluder sizes are replaced by the width / height / gap / thickness rule above.
  - Anything that turns pixels into mm (3.0 px/mm) must be re-derived. The G2 5/10 px thresholds are in image space; decide explicitly whether to keep them, since at the new pose they mean about 2.3/4.7 mm.
- D2 (moving stand on an arc), D4 and D5/D6 keep their meaning.
- verify/ needs no code change for the arc layout. The rail fallback would need one.

## 5. Open questions for you

1. Is there about 1.15 m of clear floor width centred on the compartment, at least 44 cm in front of the shelf, with the right pillar/boxes and the left shelving out of the way at both end stops? Please tape-measure. From the photo this looks tight.
2. Which compartment will you use? What are its inner width, depth and clear height, and how high is its floor above the room floor (all tape-measured)? Can the shelf pins give at least 22–25 cm of clear height?
3. What is the ceiling light: LED, CFL or fluorescent? Do you have a lamp that can stay fixed in place as L1?
4. Can you get a second mug that looks like the study mug?
5. Can the stand hold the lens 25–30 cm above that shelf at about 15° pitch without wobble? Does the floor flex when you walk nearby? At 50 ms exposures, any vibration blurs the image.

**Caveats:**
- There is one photo, and all detection and embedding results come from that single frame.
- The focal length comes from spec sheets (±5%, no calibration).
- The shelf is assumed to be Billy-like (76.4 × 26 cm).
- The occlusions were pasted onto the image, not real.
- The geometry and visibility tables are a 2-D model, roughly right to about ±15%.
- The stability and lighting data cover only 4 minutes, and the camera was still warming up.

Evidence is in `data/raw/2026-10-05_shelf-eval/`:
- `pi/log.jsonl`
- `locked_ecc.json`
- `focus_results.json`
- `pi/focus*.npy` and `pi/focusrev*.npy`
- `fit/shelf_geometry.txt`
- `fit/pose_from_frame.txt`
- `fit/clear_height_sensitivity.txt`
- `fit/positions_shelf_draft.json`
- `perception/detections.json`
- `perception/embeddings.json`