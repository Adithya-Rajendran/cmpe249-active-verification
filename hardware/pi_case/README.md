# Pi 5 + AI HAT+ + Camera Module 3 case

A rigid case for the capture rig, revision 2. It replaces the MakerWorld case, whose friction-pivot camera arm let the camera tilt. This revision follows the student's feedback on the first design:

- **Pi stack (not screwed to the case).** The Pi 5 with the AI HAT+ on top drops into a snug cradle. Four flat posts carry it, under the heads of the HAT kit's own bottom spacer screws. Low fences stand 0.5 mm from the PCB edge (`PI_CLR`). The closed lid holds the stack down with four pads that press 0.3 mm (`PRESS`) on the kit's top screw heads, so the load runs straight down the spacer columns. No screw goes into the Pi or the spacers.
- **Camera (screwed to the front wall).** Four M2.5 x 8 screws go through the board's 21 x 12.5 mm holes and four 2.5 mm standoffs into M2.5 nuts captured in hex pockets in the front face. The board's front face sits on the four coplanar standoff faces, which form the datum plane. That plane is parallel to the front face and perpendicular to the bottom face. So the lens axis is normal to the front face and parallel to the bottom. There are no pivots, arms or clamp bars.
- **Lid.** Four M2.5 x 8 screws hold it to the base. They go into M2.5 nuts captured in side-loaded slots in four corner towers.
- **Mounting.** A 1/4"-20 UNC hex nut is captured in the floor for a tripod or selfie stick. The two long sides are flat, parallel and **72 mm** apart for a phone clamp.
- **Outer size:** 113 x 72 x 41.7 mm with the lid on. The lens axis is centred across the case and 23.4 mm above the bottom face.

## Rebuild

Run these from the repo root. `make_pi_case.py` holds every dimension as a named parameter. It runs the self-check (last result below), refuses to write the STLs if any check fails, and picks each screw length from the kit's list.

```
uv run --no-sync --with trimesh --with shapely --with mapbox-earcut --with manifold3d python hardware/pi_case/make_pi_case.py
uv run --no-sync --with trimesh --with shapely --with mapbox-earcut --with manifold3d --with matplotlib --with networkx --with scipy python hardware/pi_case/preview.py
```

| File | Print | Orientation |
|---|---|---|
| `pi_case_base.stl` | 1 | as exported: floor on the bed |
| `pi_case_lid.stl` | 1 | as exported: top face on the bed, lip and press pads up |

The first design's `pi_case_cam_clamp.stl` is gone: the camera is now screwed on directly.

`preview.py` and `pi_case_preview.png` are local-only (gitignored by `hardware/.gitignore`, not in the repo); the second command above works only where `preview.py` exists.

`pi_case_preview.png` has four panels:
- plan sections: the floor, the cradle at the Pi's height, and the nut slots in the corner towers;
- the base in 3D;
- the front face, showing the lens window, the four camera nut pockets and the board outline;
- a side section with the stack, camera, kit screw heads and every M2.5 screw and nut drawn in.

## Print settings

- **Material:** PLA or PETG.
- **Layers:** 0.2 mm layers, 0.4 mm nozzle, 4 walls/perimeters, 5 top/bottom layers, 25–40 % gyroid infill. Use 100 % infill, or a modifier, around the four camera standoffs and the front wall. They carry the camera screws.
- **Supports:** none. Every overhang is one of these:
  - a bridge of 17 mm or less: the Ethernet window (17 mm), the USB-A windows (15.5 mm), the USB-C window (13 mm), the lens window (12 mm), the tripod-nut pocket roof (11.4 mm), the micro-HDMI relief pockets (9 mm wide, 1.8 mm deep), the lid-nut slot roofs (5.3 mm) and the camera nut-pocket tops (3.1 mm; the hexes print with a flat on top). Turn on bridge settings in the slicer;
  - a short horizontal cylinder: the camera standoffs, 2.5 mm long;
  - a 45° corbel: the undersides of the four corner towers.

  The cradle fences and posts rise from the floor.
- **Bed:** use a flat bed. The bottom face is the tripod and level reference. A brim helps the base in PETG.
- **Squareness:** the front face's squareness to the bottom depends on the printer's Z axis. Check it with a square (fit check 2).
- **Nut pockets:** they are 0.3 mm over the nut across flats (`NUT_CLR_AF`) and 0.3 mm over its thickness (`NUT_CLR_H`).
  - If nuts fall out, reduce `NUT_CLR_AF` to 0.2.
  - If they will not go in, raise it to 0.4.
  - Print a base first and test the nuts before printing the lid.

## Hardware (BOM)

Screws are M2.5 socket head from the kit (4/6/8/10/12/16/20 mm). `make_pi_case.py` picks the shortest one that gives full nut engagement and stays inside the part, and prints the reasoning.

| Location | Qty | Fastener | Why this length |
|---|---|---|---|
| Camera to front wall | 4 | **M2.5 x 8** socket head + **M2.5 nut** in the front-face pocket. **No washer.** | The head sits on the camera's ø4.75 hole pad, behind the board. The grip is 1.12 (board) + 2.5 (standoff) + 2.2 (wall behind the nut) = 5.82 mm. That gives the full 2.0 mm nut engagement, and the tip ends 0.62 mm behind the front face. A 6 mm screw only reaches 0.2 mm into the nut. A 10 mm screw sticks out of the front face. |
| Lid to base | 4 | **M2.5 x 8** socket head + **M2.5 nut** in the corner-tower slot. **Optional flat washer** under the head. | The grip is 2.5 (lid) + 2.0 (slot roof) = 4.5 mm, or 5.0 mm with a washer. Either way there is full 2.0 mm engagement, with 1.5 (or 1.0) mm of screw past the nut. The tip ends inside the tower, above the Ethernet jack. |
| Pi 5 / AI HAT+ | 0 | none from the M2.5 kit | Keep **all eight** HAT-kit screws fitted as Raspberry Pi installs them. The four under the Pi stand on the posts; the four on top take the lid pads. |
| Tripod / selfie stick | 1 | **1/4"-20 UNC finished hex nut**: 7/16" (11.1 mm) across flats, 7/32" (5.6 mm) thick | The pocket is 11.4 AF x 5.9 mm. A thinner jam nut also fits. A heavy hex nut (1/2" AF) does not. |
| Camera cable | 1 | Pi 5 camera cable, 22-pin to 15-pin, 200 mm | The 15-to-15-pin cable that ships with Camera Module 3 does not fit the Pi 5. Skip this if the camera already works on this Pi. |

**Totals:** 8 x M2.5 x 8 socket head screws and 8 x M2.5 nuts, plus 0–4 M2.5 flat washers (lid only). No lock washers: the nuts are captured and cannot turn, and the lid screws are re-opened often. Tools: a 2 mm hex key.

**Camera holes:** the official drawing gives ø2.2 mm, which is under M2.5's 2.5 mm. The student reports that M2.5 screws fit this camera's holes. Try one screw through the bare board before printing. If it needs force, do not drill or ream the board: stop and rethink the fixing.

## Assembly order

1. **Tripod nut.** Slide the 1/4"-20 nut into the slot on the front side of the floor puck, flats parallel to the slot walls, until it drops into the hex pocket. The 1.5 mm roof above it stops a long tripod screw before it reaches the Pi.
2. **Lid nuts.** Push an M2.5 nut into each corner tower's slot from the inside of the case, flats parallel to the slot walls, until it sits over the screw hole.
   - The front slots face the camera bay; the back slots face the Pi bay.
   - To check alignment, run an M2.5 x 8 down through the tower hole and back it out. Do this now, while the bay is empty.
   - The slots are open on the inside, so a nut can slide back out while its screw is out. Keep the base upright when the lid is off, or hold the nut in with a dab of glue on the outer flat.
3. **Camera nuts.** Press an M2.5 nut into each of the four hex pockets in the front face. Put a strip of tape over them on the outside so they stay put while you screw.
4. **Camera.**
   - Hold the board on the four standoffs, lens into the window, cable-connector edge up (as the drawing's side view shows). This makes the sensor image **upside down**; it can't be flipped in the case (the lens is off-centre on the board), so captures use `rpicam-still --rotation 180` (`lock.rotation` in `configs/capture_lock.json`, pi_capture >= 0.4.0).
   - From inside the case, push an M2.5 x 8 through each board hole and standoff into its nut.
   - Tighten with a 2 mm hex key along the case axis; with the Pi bay still empty you can reach straight in. Go diagonally opposite, snug and even. The board must sit flat on all four standoff faces.
   - Remove the tape.
5. **Camera cable.** Plug the 15-pin end into the camera (contacts as marked on the connector).
6. **Pi stack.** Assemble the Pi 5, Active Cooler (optional), stacking header, spacers and AI HAT+ following Raspberry Pi's instructions, **with all eight kit screws**. Plug the cable's 22-pin end into a CAM/DISP connector first: this is easier with the HAT off.
   - The HAT has a 10 x 16 mm notch over those connectors, 47.5–57.5 mm from its SD-card edge. Route the cable up through that notch, over the HAT, and down the front of the case to the camera's top edge.
   - Keep the cable inboard of the front-left press pad (over the hole nearest the SD/USB-C corner).
7. **Into the base.** Lower the stack into the cradle, SD-card edge toward the camera, USB-C toward the side wall with the port window. It drops onto the four posts between the fences; nothing is screwed.
8. **Lid.** Lay the lid on; its lip drops inside the walls and is notched round the towers. Drive 4 x M2.5 x 8 (optional flat washers) into the tower nuts until the lid sits down on the wall tops. The pads then press 0.3 mm on the HAT's top screw heads.

**Access:**
- **USB-C:** a 13 x 8 mm window in the side wall; the Pi sits 1 mm from that wall.
- **Ports:** Ethernet and both USB-A stacks open in the back wall.
- **Micro-HDMI:** closed by default; set `CUT_HDMI = True` to open both. The micro-HDMI shells stick out 1.65 mm past the Pi's edge, more than the 1.0 mm gap to that wall. So the wall has two 1.8 mm-deep relief pockets behind them (`HDMI_RELIEF`), leaving 1.2 mm of wall there and at least 0.65 mm of clearance with the Pi pushed against the fence.
- **SD card:** reachable through the 16 x 36 mm floor opening without removing the lid.
- **Cable or stack:** take off the lid (4 screws) and lift the stack out.

**Airflow:**
- **Active Cooler:** fits under the AI HAT+ (product brief: the 16 mm header and spacers allow it). The cooler is 13.70 mm tall, leaving 2.3 mm to the HAT.
- **Vents:** side-wall slots at fan height (kept clear of the corner towers), lid slots over the HAT, and the open back-port windows.
- **Covering:** don't cover the lid. The AI HAT+ brief warns against covering a case it is used in.

## Fit checks

Do these with the printed parts. If one fails, change the named parameter and rebuild. The generator re-checks the geometry and picks the screws again.

1. **Nut pockets (print the base first).**
   - An M2.5 nut presses into a front-face pocket and stays.
   - Another slides into a tower slot and lines up with the hole: an M2.5 x 8 from above threads in by hand.
   - Change `NUT_CLR_AF` / `NUT_CLR_H` if needed.
2. **Front face square.** A square standing on the bottom face lies flat against the front face, with no light gap over 35 mm of height.
3. **Camera datum.**
   - With the four screws snug, the board touches all four standoff faces; a 0.05 mm feeler should not go in.
   - The sensor housing clears the 12 mm window (`WINDOW`) all round. By the drawing the lens tip sits only about 0.1 mm behind the front face (0.07 to 0.52 mm), so it is almost flush: don't put the case face down on a hard surface.
   - Nothing on the board's front touches the wall; there is 2.5 mm of room (`STANDOFF`).
4. **Camera screws.** The screws thread fully into the nuts, and no tip shows past the front face (`CAM_NUT_DEPTH`, `STANDOFF`, `FRONT`).
5. **Stack height.** Measure the assembled stack with calipers, from the bottom of the lower kit-screw heads to the top of the upper ones. The design assumes 23.2 mm: 2.0 head + 1.6 Pi + 16 spacer + 1.6 HAT + 2.0 head.
   - If it differs by more than 0.2 mm, set `BOT_HEAD_H` / `TOP_HEAD_H` to the measured head heights and rebuild.
   - Otherwise the lid either will not sit on the walls or will not press the stack.
6. **Pi cradle.**
   - The stack drops in without force and rests on all four posts.
   - With the lid off, it moves no more than about 1 mm in X or Y (`PI_CLR` = 0.5 per side). If it binds, raise `PI_CLR` to 0.6. If it rattles, lower it to 0.4.
   - The fences touch only the PCB edge, below the port overhangs.
7. **Lid closed.**
   - Close the lid and tighten the screws: it sits flat on all four walls.
   - Pressing on the lid, the stack does not move.
   - If the lid rocks on the pads, they press too hard: lower `PRESS`, or measure as in item 5.
   - If the stack still rattles, raise `PRESS` to 0.5.
8. **Tripod nut.** A tripod screw threads in and pulls the bottom face flat onto the tripod plate. The screw may stick up at most 8.4 mm above the plate; a longer one bottoms on the 1.5 mm pocket roof before the plate is tight (normal camera screws are about 4.5 to 6.5 mm). Only a screw over 12 mm that has broken through the roof could reach the Pi.
9. **Ports.** The USB-C plug seats fully (power LED on). The Ethernet and USB-A plugs latch. The SD card can be removed and reinserted through the floor.
10. **Phone clamp.** It grips the two 72 mm sides without covering the USB-C plug.
11. **On the Pi** (later, by the student):
    - take a test still;
    - check the image is upright with `--rotation 180` (it is upside down without it);
    - tap the case: the image must not shift.

## Dimensions and sources

All dimensions are from Raspberry Pi Ltd documents, fetched 2026-10-12 by the first design pass:

- Pi 5 mechanical drawing: <https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-mechanical-drawing.pdf>
  - board 85 x 56;
  - M2.5 holes 3.5 mm from the edges, 58 x 49 apart;
  - USB-C centre 11.2 mm from the SD edge (micro-HDMI at 25.8 and 39.2);
  - USB-A centres at 29.1 and 47, Ethernet centre at 10.2, from the USB-C edge;
  - the ports overhang the board edge by 3 mm.
- AI HAT+ product brief: <https://datasheets.raspberrypi.com/ai-hat-plus/raspberry-pi-ai-hat-plus-product-brief.pdf>
  - board 65 x 56.5, same 58 x 49 holes;
  - 16 mm stacking header and spacers, which allow the Active Cooler;
  - a 10 x 16 mm notch over the camera/display connectors.
- AI HAT+ installation steps (spacers fixed to the Pi with screws from below, HAT fixed on top with screws): <https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html>
- Camera Module 3 (standard) mechanical drawing: <https://datasheets.raspberrypi.com/camera/camera-module-3-standard-mechanical-drawing.pdf>
  - board 25 x 23.862 x 1.12;
  - holes ø2.2 on ø4.75 pads, 2.0 mm from the left, right and bottom edges, **21 x 12.5 apart**;
  - lens axis 12.5 / 14.4 mm from the left and bottom edges;
  - sensor housing 10.8 x 10.8;
  - 66° x 41° field of view.
- Camera Module 3 product brief (25 x 24 x 11.5 mm): <https://datasheets.raspberrypi.com/camera/camera-module-3-product-brief.pdf>
- Active Cooler product brief (13.70 mm tall, 63.5 x 42.5 footprint): <https://datasheets.raspberrypi.com/cooling/raspberry-pi-active-cooler-product-brief.pdf>

**Assumptions** (not on the drawings; all are named parameters):
- **PCB thickness:** 1.6 mm for both the Pi and the HAT.
- **HAT-kit screw heads:** 2.0 mm tall, ø4.5 (`BOT_HEAD_H`, `TOP_HEAD_H`, `KIT_HEAD_D`). Fit check 5 measures them.
- **Port jack envelopes**, used only by the clearance check: RJ45 16 x 13.5 x 21 mm, stacked USB-A 14.5 x 16 x 17.5 mm (`ETH_BOX`, `USB_BOX`).
- **Port window sizes:** the drawings give centres only, so the windows are sized with margin. The SD card's position across the board is not dimensioned, so the floor opening spans 36 mm.
- **Lens protrusion:** 6.98–7.43 mm in front of the PCB. The side view gives 6.98 mm from the sensor flex, and 11.3 mm overall = 6.98 + flex + 1.12 board + 2.75 back connector, so the flex is 0.45 mm and the tip is 7.43 mm from the PCB front face. The lens aperture radius in the field-of-view check is 2.0 mm (`LENS_R`).
- **USB-C-edge connectors:** the drawing dimensions only their centres and heights (USB-C 3.2, micro-HDMI 3.4). Their widths and how far they stick out past the board edge (USB-C 8.9 wide, 1.27 out; micro-HDMI 7.0 wide, 1.65 out; `USBC_BOX`, `HDMI_BOX`) were scaled off the 1:1 drawing. Check them with calipers.
- **Camera holes:** ø2.2 on the drawing; the student reports M2.5 fits (see the BOM).

## Self-check (last run, 2026-10-12)

`make_pi_case.py` checks each of these by probing the meshes or intersecting solids. It refuses to write STLs if any check fails. The values below are from its last run.

- **Meshes:** base and lid are watertight with consistent winding.
  - base: 113.0 x 72.0 x 39.2 mm, 69.6 cm³;
  - lid: 113.0 x 72.0 x 8.8 mm in print orientation, 20.1 cm³;
  - lid vs base overlap: 0.0000 mm³ (the lip is notched round the towers); the lid sits on the wall tops at z = 39.2;
  - each STL is reloaded after export and must be one watertight body. The float32 export split a 2-triangle, zero-volume sliver off the base at a lid-nut slot roof; the script now drops it.
- **Phone clamp:** the long sides are flat and parallel, 72.0 mm apart (target 70–85).
- **Pi cradle:**
  - the fence faces are 0.5 mm from the PCB edge at 8 places (gap empty 8/8, fence solid 8/8);
  - the fence tops are at z = 13.0, between the Pi underside (12.0) and its top (13.6);
  - the 4 flat posts top out at z = 10.0, with no holes;
  - the assumed stack envelope (Pi, cooler, HAT, spacers, RJ45/USB-A, USB-C and micro-HDMI) overlaps the base by 0.0000 mm³ and the lid by 0.0000 mm³;
  - the USB-C and micro-HDMI shells, with the Pi pushed into each corner of the cradle play (±0.5 mm), overlap the base by 0.0000 mm³. The worst-case gap to the HDMI relief is 0.65 mm. Without the relief they overlap by up to 55 mm³, so the stack would not drop in;
  - the lid pads overlap the 4 top screw heads by exactly the 0.3 mm `PRESS` (19.0 mm³ vs 19.1 expected).
- **Camera:**
  - screw holes at case (y, z) (46.5, 11.0), (25.5, 11.0), (46.5, 23.5), (25.5, 23.5): spacing **21.00 x 12.50**, matching the drawing's 21.00 x 12.50; holes open 4/4, standoff walls solid 4/4;
  - datum: the 4 standoff faces are coplanar at x = 7.50 (36.1 mm² of +X faces);
  - lens axis: 0.0000° from the front-face normal (90° to the front face) and 90.0000° to the bottom normal (parallel to the bottom); front face vs bottom face 90.0000°;
  - lens window: 144 field-of-view ray samples are all clear, with a worst-case diagonal margin of 3.61 mm; the lens tip is 0.07–0.52 mm behind the front face;
  - the board, sensor housing and lens envelope overlap the base by 0.0000 mm³.
- **Camera nut pockets:** 5.30 AF x 2.80 deep, for 5.0 AF x 2.0 nuts, with 2.20 mm of wall behind the nut. The M2.5 x 8 screws get full 2.0 mm engagement, and their tips end 0.62 mm behind the front face.
- **Lid nuts:** the slots are 5.30 wide x 2.30 tall at z 34.90–37.20, with a solid 2.0 mm roof. The corbels start at z = 24.40 and are full size from z = 33.90. The M2.5 x 8 screws get full 2.0 mm engagement, with or without a washer, and their tips end at z 33.7–34.2, inside the towers.
- **Tripod nut:** the 11.4 AF x 5.9 pocket is empty, the roof is solid and the screw hole is open. The puck top is at z = 9.9, below the Pi underside at 12.0.
- **Stack heights (z, mm):**

  | Item | z (mm) |
  |---|---|
  | Post top | 10.0 |
  | Pi underside | 12.0 |
  | Pi top | 13.6 |
  | Active Cooler top | 27.3 |
  | HAT underside | 29.6 |
  | HAT top | 31.2 |
  | Lid pad bottom | 32.9 |
  | Lid underside / wall top | 39.2 |
