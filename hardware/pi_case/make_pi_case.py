"""Rigid case for Raspberry Pi 5 + AI HAT+ (16 mm stack) + Camera Module 3 (standard).

Design (revision 2, after the student's feedback):
  * The Pi 5 + AI HAT+ stack is NOT screwed to the case. It stands on four flat posts
    (under the heads of the HAT kit's own bottom spacer screws) inside a snug cradle of
    fences that touch only the Pi PCB edge (PI_CLR per side). The closed lid holds it
    down with four pads that press PRESS mm on the HAT kit's top screw heads, so the
    clamping load runs straight down the spacer columns.
  * The camera board is screwed to the front wall: 4 x M2.5 through its 21 x 12.5 holes,
    board front face flat on four coplanar standoff faces (the datum plane, parallel to
    the front face), nuts captured in hex pockets in the front face. Lens axis = datum
    normal = front-face normal, parallel to the flat bottom face. No pivots, no arms.
  * The lid screws onto the base: 4 x M2.5 down into nuts captured in side-loaded slots in
    four corner towers (corbels with 45-degree undersides, so no supports).
  * The USB-C wall has relief pockets for the micro-HDMI shells, which overhang the board edge
    (1.65 mm on the drawing) by more than the 1.0 mm gap to that wall.
  * A 1/4"-20 UNC hex nut is captured in the floor (tripod / selfie stick); the two long
    sides are flat and parallel, W mm apart, for a phone clamp.

Case frame (mm): X along the case, the camera looks toward -X (front face at x = 0);
Y across (USB-C side at y = 0); Z up, bottom face at z = 0.
Pi 5 board frame (official drawing): x = 0 at the SD-card edge, y = 0 at the USB-C /
micro-HDMI edge. The board origin sits at (BX, BY, PI_Z) in the case frame.

Parts:
  pi_case_base.stl   floor, walls, cradle, camera standoffs + nut pockets, lid-nut towers,
                     tripod-nut pocket (print as exported, floor on the bed)
  pi_case_lid.stl    lid with alignment lip and press pads (exported upside down for printing)

Run (from the repo root):
  uv run --no-sync --with trimesh --with shapely --with mapbox-earcut --with manifold3d \
      python hardware/pi_case/make_pi_case.py
"""
import math
import warnings
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon

OUT = Path(__file__).resolve().parent

# ---- official dimensions (see README for the source drawings) ----
PI_L, PI_W, PI_T = 85.0, 56.0, 1.6          # Pi 5 board; 1.6 = standard PCB thickness (not on the drawing)
PI_HOLES = [(3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)]   # 58 x 49, from the board's SD/USB-C corner
USBC_X = 11.2                               # USB-C centre from the SD edge
HDMI_X = (25.8, 39.2)                       # micro-HDMI centres from the SD edge
# USB-C-edge connectors (w along the edge, h above the board top, overhang past the board edge).
# Overhangs and widths scaled off the 1:1 drawing (rendered at 400 dpi); heights are its 3.2 / 3.4 dims.
USBC_BOX = (8.9, 3.2, 1.27)
HDMI_BOX = (7.0, 3.4, 1.65)
PORT_OVERHANG = 3.0                         # Ethernet/USB-A stick out past the board edge
ETH_Y, USB_Y = 10.2, (29.1, 47.0)           # port centres from the USB-C edge
STACK = 16.0                                # AI HAT+ stacking header / spacer height
HAT_L, HAT_W, HAT_T = 65.0, 56.5, 1.6       # AI HAT+ board (thickness assumed standard)
CAM_W, CAM_H, CAM_T = 25.0, 23.862, 1.12    # Camera Module 3 board
CAM_HOLES = [(2.0, 2.0), (23.0, 2.0), (2.0, 14.5), (23.0, 14.5)]  # 21 x 12.5, from the board's lower-left corner
CAM_LENS = (12.5, 14.4)                     # lens axis on the board
CAM_PAD_D = 4.75                            # copper pad (keep-out) around each camera hole
# lens tip in front of the PCB front face. Side view: 6.98 from the housing's back (the sensor flex), and
# 11.3 overall = 6.98 + flex + 1.12 PCB + 2.75 back connector -> flex 0.45 -> 7.43 from the PCB front face.
LENS_PROTRUSION = (6.98, 7.43)
SENSOR_HOUSING = 10.8                       # square sensor housing on the board front
CAM_HFOV, CAM_VFOV = 66.0, 41.0             # Camera Module 3 standard, degrees
COOLER_H = 13.70                            # Raspberry Pi Active Cooler height above the board

# ---- assumed envelopes (not on the drawings; used only by the clearance self-check) ----
ETH_BOX = (16.0, 13.5, 21.0)                # RJ45 jack w x h x depth, depth measured back from its front face
USB_BOX = (14.5, 16.0, 17.5)                # stacked USB-A jack
KIT_HEAD_D = 4.5                            # HAT kit spacer-screw head diameter
BOT_HEAD_H = 2.0                            # head of the kit screw under the Pi (spacer to Pi)
TOP_HEAD_H = 2.0                            # head of the kit screw on top of the HAT (HAT to spacer)
LENS_R = 2.0                                # lens front aperture radius used for the field-of-view rays

# ---- fasteners ----
M25_HOLE = 2.8                              # clearance hole
M25_HEAD_D, M25_HEAD_H = 4.5, 2.5           # ISO 4762 socket head
NUT_AF, NUT_H = 5.0, 2.0                    # ISO 4032 M2.5 nut
NUT_CLR_AF, NUT_CLR_H = 0.3, 0.3            # pocket = nut + clearance (total, not per side)
SCREWS = (4, 6, 8, 10, 12, 16, 20)          # M2.5 socket head lengths in the kit
WASHER_T = 0.5                              # M2.5 flat washer (optional, lid screws only)
QTR_NUT_AF, QTR_NUT_H, QTR_HOLE = 11.4, 5.9, 6.8   # 1/4"-20 hex nut is 7/16" (11.11) AF x 7/32" (5.56)

# ---- printable-part parameters ----
WALL, FRONT, FLOOR = 3.0, 5.0, 3.0          # side/back wall, front (reference) wall, floor
BAY = 16.0                                  # front wall inner face -> Pi SD edge (camera + cable)
Y_GAP, IN_W = 1.0, 66.0                     # Pi edge to the USB-C wall; inner width -> outer 72 mm
BACK_GAP = 1.0                              # port fronts to the back wall
PI_Z = 12.0                                 # Pi underside above the case bottom
PI_CLR = 0.5                                # cradle fence to Pi PCB edge, per side
FENCE_T, FENCE_ENGAGE = 3.0, 1.0            # fence thickness; fence top this far above the Pi underside
POST_D = 6.0                                # flat posts under the bottom spacer-screw heads
HAT_CLEAR = 8.0                             # HAT top -> lid underside (header pins, camera ribbon)
PRESS = 0.3                                 # lid pads press this far into the nominal stack top
LID_PAD_D = 7.0
LID_T, LIP_H, LIP_T, LIP_CLR = 2.5, 3.0, 1.6, 0.4
TOWER = 9.0                                 # lid-nut corner tower, square
LNUT_ROOF, LNUT_FLOOR = 2.0, 1.0            # plastic above / below the lid-nut slot
CAM_VB = 9.0                                # camera board bottom edge above the case bottom
STANDOFF, STANDOFF_D = 2.5, 4.4             # camera standoffs (inside the 4.75 pad keep-out)
CAM_NUT_DEPTH = 2.8                         # camera nut pocket depth from the front face
TIP_RECESS_MIN = 0.3                        # camera screw tips stay this far behind the front face
WINDOW = 12.0                               # square lens window (sensor housing is 10.8)
NUT_FLOOR, NUT_ROOF, PUCK = 2.5, 1.5, 20.0  # tripod nut pocket
USB_WIN = (13.0, 8.0)
CUT_HDMI = False                            # headless; set True to open both micro-HDMI ports
HDMI_RELIEF = 1.8                           # (CUT_HDMI False) pocket in the USB-C wall's inner face for the
                                            # micro-HDMI overhang; leaves WALL - HDMI_RELIEF of wall there
VENT_W, VENT_PITCH = 2.5, 5.0

# ---- derived layout ----
IN_L = BAY + PI_L + PORT_OVERHANG + BACK_GAP
L, W = FRONT + IN_L + WALL, IN_W + 2 * WALL
BX, BY = FRONT + BAY, WALL + Y_GAP
PI_TOP = PI_Z + PI_T
POST_TOP = PI_Z - BOT_HEAD_H
FENCE_TOP = PI_Z + FENCE_ENGAGE
HAT_Z = PI_TOP + STACK                      # HAT underside
HAT_TOP = HAT_Z + HAT_T
LID_Z = HAT_TOP + HAT_CLEAR                 # lid underside = wall top (lid sits on the walls)
WALL_TOP = LID_Z
PAD_Z = HAT_TOP + TOP_HEAD_H - PRESS        # lid pad bottom
NUT_POCKET_AF, NUT_POCKET_H = NUT_AF + NUT_CLR_AF, NUT_H + NUT_CLR_H
LNUT_TOP = WALL_TOP - LNUT_ROOF             # nut pulled up against the slot roof
LNUT_SLOT0 = LNUT_TOP - NUT_POCKET_H
TOWER_FULL = LNUT_SLOT0 - LNUT_FLOOR        # tower is full size from here up
TOWER_Z0 = TOWER_FULL - (TOWER + 0.5)       # corbel tip (45-degree underside)
CAM_Y = W / 2                               # camera centred across the case
LENS_Z = CAM_VB + CAM_LENS[1]
DATUM_X = FRONT + STANDOFF                  # camera board front face
HEAD_SEAT_X = DATUM_X + CAM_T               # camera screw heads seat on the board back face
TOWERS = [(FRONT, WALL, 1, 1), (FRONT, W - WALL, 1, -1), (L - WALL, WALL, -1, 1), (L - WALL, W - WALL, -1, -1)]


def box(x0, y0, z0, x1, y1, z1):
    m = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    m.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return m


def cyl(d, p0, p1, sections=48):
    """Cylinder of diameter d from point p0 to point p1."""
    return trimesh.creation.cylinder(radius=d / 2, segment=[p0, p1], sections=sections)


def hexagon(af, a, b):
    r = af / math.sqrt(3)                   # circumradius; vertices on +-a, flats facing +-b
    return Polygon([(a + r * math.cos(t), b + r * math.sin(t)) for t in np.radians(np.arange(0, 360, 60))])


def hex_z(af, x, y, z0, z1):
    """Hex prism along Z, flats facing +-Y."""
    m = trimesh.creation.extrude_polygon(hexagon(af, x, y), z1 - z0)
    m.apply_translation((0, 0, z0))
    return m


def hex_x(af, y, z, x0, x1):
    """Hex prism along X, flats facing +-Z (flat 3 mm bridge on top when printed)."""
    m = trimesh.creation.extrude_polygon(hexagon(af, y, z), x1 - x0)
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]   # local (a, b, c) -> world (y, z, x)
    m.apply_transform(T)
    m.apply_translation((x0, 0, 0))
    return m


def union(ms):
    return trimesh.boolean.union(ms, engine="manifold")


def diff(a, ms):
    return trimesh.boolean.difference([a] + list(ms), engine="manifold")


def inter(a, b):
    return trimesh.boolean.intersection([a, b], engine="manifold")


def pi_xy(bx, by):
    return BX + bx, BY + by


def cam_yz(u, v):
    """Camera board point (u from the board's left edge seen from the front, v up) -> case (y, z)."""
    return CAM_Y - (u - CAM_W / 2), CAM_VB + v     # seen from the front, board +u runs toward -Y


def tower_xy(cx, cy, sx, sy):
    """Screw axis of a lid-nut tower whose inner corner (wall faces) is at (cx, cy)."""
    return cx + sx * TOWER / 2, cy + sy * TOWER / 2


def tower(cx, cy, sx, sy):
    """Corner tower: full TOWER x TOWER square from TOWER_FULL to the wall top, below that a
    corbel whose two undersides are 45-degree planes running into the two walls."""
    e = 0.5                                 # bury 0.5 mm into the walls
    x0, y0, E = cx - sx * e, cy - sy * e, TOWER + e
    x1, y1 = x0 + sx * E, y0 + sy * E
    prism = box(min(x0, x1), min(y0, y1), TOWER_FULL - 0.01, max(x0, x1), max(y0, y1), WALL_TOP)
    # pyramid: apex on the wall corner at TOWER_Z0, square base at TOWER_FULL; its section at
    # height z is the square [0, z - TOWER_Z0]^2 from the corner -> 45-degree undersides
    v = np.array([(x0, y0, TOWER_Z0), (x0, y0, TOWER_FULL), (x1, y0, TOWER_FULL), (x1, y1, TOWER_FULL), (x0, y1, TOWER_FULL)])
    f = np.array([(0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 1, 4), (1, 2, 3), (1, 3, 4)])
    pyr = trimesh.Trimesh(v, f, process=False)
    if pyr.volume < 0:
        pyr.invert()
    return union([prism, pyr])


def fence_boxes():
    """Cradle fences around the Pi PCB, floor to FENCE_TOP, PI_CLR from the board edge.
    Pieces avoid the SD card (middle of the SD edge) and the side-wall port windows."""
    c, t, z0, z1 = PI_CLR, FENCE_T, FLOOR - 0.5, FENCE_TOP
    out = []
    for y0, y1 in ((0.5, 8.0), (48.0, 55.5)):                 # SD edge, either side of the card
        out.append(box(BX - c - t, BY + y0, z0, BX - c, BY + y1, z1))
    for y0, y1 in ((4.0, 12.0), (40.0, 52.0)):                # back edge, under the port overhang
        out.append(box(BX + PI_L + c, BY + y0, z0, L - WALL + 0.5, BY + y1, z1))
    for x0, x1 in ((1.0, 5.0), (50.0, 80.0)):                 # USB-C edge (thin rib on the wall)
        out.append(box(BX + x0, WALL - 0.5, z0, BX + x1, BY - c, z1))
    for x0, x1 in ((5.0, 20.0), (60.0, 80.0)):                # GPIO edge (block to the wall)
        out.append(box(BX + x0, BY + PI_W + c, z0, BX + x1, W - WALL + 0.5, z1))
    return out


FENCE_PROBES = [  # (point on the board edge (Pi coords), outward unit normal) where a fence sits
    ((0.0, 4.0), (-1, 0)), ((0.0, 52.0), (-1, 0)), ((PI_L, 8.0), (1, 0)), ((PI_L, 46.0), (1, 0)),
    ((3.0, 0.0), (0, -1)), ((65.0, 0.0), (0, -1)), ((12.0, PI_W), (0, 1)), ((70.0, PI_W), (0, 1)),
]


def build_base():
    shell = box(0, 0, 0, L, W, WALL_TOP)
    cavity = box(FRONT, WALL, FLOOR, L - WALL, W - WALL, WALL_TOP + 1)
    solids = []
    # cradle: posts under the kit's bottom screw heads, fences at the PCB edge
    for hx, hy in PI_HOLES:
        x, y = pi_xy(hx, hy)
        solids.append(cyl(POST_D, (x, y, FLOOR - 0.5), (x, y, POST_TOP)))
    solids += fence_boxes()
    # tripod-nut puck at the case centre
    cx, cy = L / 2, W / 2
    solids.append(box(cx - PUCK / 2, cy - PUCK / 2, FLOOR - 0.5, cx + PUCK / 2, cy + PUCK / 2, NUT_FLOOR + QTR_NUT_H + NUT_ROOF))
    # camera standoffs: their +X faces are the datum plane x = DATUM_X
    for u, v in CAM_HOLES:
        y, z = cam_yz(u, v)
        solids.append(cyl(STANDOFF_D, (FRONT - 0.5, y, z), (DATUM_X, y, z)))
    # lid-nut corner towers
    solids += [tower(*t) for t in TOWERS]
    base = union([diff(shell, [cavity])] + solids)

    cuts = []
    # tripod nut: pocket + side slot opening toward the front (insert before the Pi), clearance hole
    cuts.append(hex_z(QTR_NUT_AF, cx, cy, NUT_FLOOR, NUT_FLOOR + QTR_NUT_H))
    cuts.append(box(cx - PUCK / 2 - 1, cy - QTR_NUT_AF / 2, NUT_FLOOR, cx, cy + QTR_NUT_AF / 2, NUT_FLOOR + QTR_NUT_H))
    cuts.append(cyl(QTR_HOLE, (cx, cy, -1), (cx, cy, NUT_FLOOR + 0.5)))
    # lens window; camera screw holes through standoff + wall; nut pockets in the front face
    cuts.append(box(-1, CAM_Y - WINDOW / 2, LENS_Z - WINDOW / 2, FRONT + 0.01, CAM_Y + WINDOW / 2, LENS_Z + WINDOW / 2))
    for u, v in CAM_HOLES:
        y, z = cam_yz(u, v)
        cuts.append(cyl(M25_HOLE, (-1, y, z), (DATUM_X + 1, y, z)))
        cuts.append(hex_x(NUT_POCKET_AF, y, z, -1, CAM_NUT_DEPTH))
    # lid-nut towers: screw hole (through the corbel) + nut slot opening into the case
    for cxx, cyy, sx, sy in TOWERS:
        x, y = tower_xy(cxx, cyy, sx, sy)
        cuts.append(cyl(M25_HOLE, (x, y, TOWER_Z0 - 1), (x, y, WALL_TOP + 1)))
        cuts.append(hex_z(NUT_POCKET_AF, x, y, LNUT_SLOT0, LNUT_TOP))
        xe = cxx + sx * (TOWER + 1)
        cuts.append(box(min(x, xe), y - NUT_POCKET_AF / 2, LNUT_SLOT0, max(x, xe), y + NUT_POCKET_AF / 2, LNUT_TOP))
    # SD card access through the floor (card is under the Pi at its front edge)
    cuts.append(box(FRONT + 6, BY + 10, -1, BX + 6, BY + 46, FLOOR + 1))
    # USB-C (and optionally micro-HDMI) on the y = 0 wall
    uz = PI_TOP + 1.6
    for px in [USBC_X] + (list(HDMI_X) if CUT_HDMI else []):
        x = BX + px
        cuts.append(box(x - USB_WIN[0] / 2, -1, uz - USB_WIN[1] / 2, x + USB_WIN[0] / 2, WALL + 1, uz + USB_WIN[1] / 2))
    if not CUT_HDMI:                        # the micro-HDMI shells overhang the board edge by more than Y_GAP
        for px in HDMI_X:
            x, hw = BX + px, HDMI_BOX[0] / 2 + PI_CLR + 0.5
            cuts.append(box(x - hw, WALL - HDMI_RELIEF, FENCE_TOP, x + hw, WALL + 0.5, PI_TOP + HDMI_BOX[1] + 1.0))
    # back wall: Ethernet + 2 x USB-A
    for y, w, h in [(ETH_Y, 17.0, 15.0)] + [(u, 15.5, 17.0) for u in USB_Y]:
        cuts.append(box(L - WALL - 1, BY + y - w / 2, PI_TOP - 0.5, L + 1, BY + y + w / 2, PI_TOP - 0.5 + h))
    # side vents at fan level (between the Pi and the HAT), kept clear of the corner towers
    z0, z1 = PI_TOP + 2.5, PI_TOP + STACK - 1.5
    xmax = L - WALL - TOWER - 1.5
    for x in np.arange(BX + 50, xmax - VENT_W, VENT_PITCH):
        cuts.append(box(x, -1, z0, x + VENT_W, WALL + 1, z1))
    for x in np.arange(BX + 5, xmax - VENT_W, VENT_PITCH):
        cuts.append(box(x, W - WALL - 1, z0, x + VENT_W, W + 1, z1))
    return diff(base, cuts)


def build_lid():
    """Assembly position; flipped for printing in main."""
    plate = box(0, 0, LID_Z, L, W, LID_Z + LID_T)
    o = WALL + LIP_CLR
    ring = diff(box(FRONT + LIP_CLR, o, LID_Z - LIP_H, L - o, W - o, LID_Z + 0.5),
                [box(FRONT + LIP_CLR + LIP_T, o + LIP_T, LID_Z - LIP_H - 1, L - o - LIP_T, W - o - LIP_T, LID_Z + 1)])
    notches = []                            # lip stops short of the corner towers
    for cx, cy, sx, sy in TOWERS:
        a, b = cx + sx * (TOWER + LIP_CLR), cy + sy * (TOWER + LIP_CLR)
        notches.append(box(min(cx - sx, a), min(cy - sy, b), LID_Z - LIP_H - 1, max(cx - sx, a), max(cy - sy, b), LID_Z + 0.25))
    ring = diff(ring, notches)
    pads = []
    for hx, hy in PI_HOLES:                 # press pads over the HAT kit's top screw heads
        x, y = pi_xy(hx, hy)
        pads.append(cyl(LID_PAD_D, (x, y, PAD_Z), (x, y, LID_Z + 0.5)))
    lid = union([plate, ring] + pads)
    cuts = []
    for t in TOWERS:
        x, y = tower_xy(*t)
        cuts.append(cyl(M25_HOLE, (x, y, LID_Z - 1), (x, y, LID_Z + LID_T + 1)))
    for x in np.arange(BX + 8, L - WALL - TOWER - 3, 6.0):
        cuts.append(box(x, BY + 12, LID_Z - 1, x + VENT_W, BY + 44, LID_Z + LID_T + 1))
    return diff(lid, cuts)


def edge_connectors(dx=0.0, dy=0.0):
    """USB-C and micro-HDMI shells on the y = 0 board edge (overhang from the drawing), Pi shifted by (dx, dy)."""
    ms = []
    for px, (w, h, o) in [(USBC_X, USBC_BOX)] + [(p, HDMI_BOX) for p in HDMI_X]:
        x = BX + dx + px
        ms.append(box(x - w / 2, BY + dy - o, PI_TOP, x + w / 2, BY + dy + 6.0, PI_TOP + h))
    return ms


def stack_envelope(heads=False):
    """Assumed envelope of the Pi 5 + Active Cooler + AI HAT+ stack (nominal position)."""
    ms = [box(BX, BY, PI_Z, BX + PI_L, BY + PI_W, PI_TOP),
          box(BX + 10, BY + 7, PI_TOP, BX + 10 + 63.5, BY + 7 + 42.5, PI_TOP + COOLER_H),
          box(BX, BY - (HAT_W - PI_W) / 2, HAT_Z, BX + HAT_L, BY + PI_W + (HAT_W - PI_W) / 2, HAT_TOP)]
    xf = BX + PI_L + PORT_OVERHANG
    w, h, d = ETH_BOX
    ms.append(box(xf - d, BY + ETH_Y - w / 2, PI_TOP, xf, BY + ETH_Y + w / 2, PI_TOP + h))
    w, h, d = USB_BOX
    for u in USB_Y:
        ms.append(box(xf - d, BY + u - w / 2, PI_TOP, xf, BY + u + w / 2, PI_TOP + h))
    ms += edge_connectors()
    for hx, hy in PI_HOLES:
        x, y = pi_xy(hx, hy)
        ms.append(cyl(3.0, (x, y, PI_Z), (x, y, HAT_Z)))          # spacer
        if heads:
            ms.append(cyl(KIT_HEAD_D, (x, y, POST_TOP), (x, y, PI_Z)))
            ms.append(cyl(KIT_HEAD_D, (x, y, HAT_TOP), (x, y, HAT_TOP + TOP_HEAD_H)))
    return union(ms)


def camera_envelope():
    """Camera board + sensor housing + lens (to the longest protrusion) + back-side screw heads."""
    y0, z0 = cam_yz(CAM_W, 0)
    ms = [box(DATUM_X, y0, z0, DATUM_X + CAM_T, y0 + CAM_W, z0 + CAM_H)]
    ly, lz = cam_yz(*CAM_LENS)
    s = SENSOR_HOUSING / 2
    ms.append(box(DATUM_X - LENS_PROTRUSION[1], ly - s, lz - s, DATUM_X, ly + s, lz + s))
    return union(ms)


def choose_screw(min_len, max_len):
    ok = [s for s in SCREWS if min_len <= s <= max_len]
    assert ok, f"no kit screw between {min_len:.2f} and {max_len:.2f} mm"
    return ok[0]


_MANIFOLDS = {}


def inside(m, pts, h=0.02):
    """Point-in-solid test without rtree: intersect a tiny cube at each point with the mesh
    (manifold3d) and call the point inside when the cube is mostly filled."""
    import manifold3d as mf
    key = id(m)
    if key not in _MANIFOLDS:
        _MANIFOLDS[key] = (m, mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, dtype=np.float32),
                                                 tri_verts=np.asarray(m.faces, dtype=np.uint32))))
    M = _MANIFOLDS[key][1]
    out = []
    for p in np.asarray(pts, dtype=float):
        c = mf.Manifold.cube([h, h, h], True).translate(list(p))
        out.append((M ^ c).volume() > 0.5 * h ** 3)
    return np.array(out)


def check(base, lid):
    ok = True

    def report(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    print("== meshes")
    for name, m in (("base", base), ("lid", lid)):
        e = m.bounds[1] - m.bounds[0]
        report(m.is_watertight and m.is_winding_consistent and m.volume > 0,
               f"{name}: watertight={m.is_watertight} winding={m.is_winding_consistent} extents {e[0]:.1f} x {e[1]:.1f} x "
               f"{e[2]:.1f} mm, {m.volume/1000:.1f} cm^3 (~{m.volume/1000*1.24:.0f} g PLA if solid)")
    print(f"  outer case {L:.1f} x {W:.1f} x {LID_Z + LID_T:.1f} mm (L x W x H, lid on); "
          f"phone-clamp sides {W:.1f} mm apart (target 70-85)")
    report(70 <= W <= 85, f"long sides flat and parallel, {W:.1f} mm apart")
    vbl = inter(base, lid).volume
    report(vbl < 1e-3, f"lid (lip notched round the towers) vs base: {vbl:.4f} mm^3 overlap; lid underside z={LID_Z:.2f} "
           f"sits on the wall top z={WALL_TOP:.2f}")

    print("== Pi cradle (no screws into the Pi)")
    pts = []
    for (bx, by), (nx, ny) in FENCE_PROBES:
        x, y = pi_xy(bx, by)
        pts += [(x + nx * (PI_CLR - 0.1), y + ny * (PI_CLR - 0.1), PI_Z + 0.5),
                (x + nx * (PI_CLR + 0.15), y + ny * (PI_CLR + 0.15), PI_Z + 0.5)]
    r = inside(base, pts)
    gap_free, fence_solid = ~r[0::2], r[1::2]
    report(gap_free.all() and fence_solid.all(),
           f"fence faces {PI_CLR} mm from the PCB edge at {len(FENCE_PROBES)} places: gap empty {int(gap_free.sum())}/8, "
           f"fence solid {int(fence_solid.sum())}/8 (total play {2*PI_CLR:.1f} mm in X and in Y)")
    report(FENCE_TOP < PI_TOP, f"fences touch only the PCB edge: fence top z={FENCE_TOP:.1f} (Pi underside {PI_Z:.1f}, "
           f"top {PI_TOP:.1f}; port overhangs start at the Pi top)")
    hp = [pi_xy(*h) for h in PI_HOLES]
    top = inside(base, [(x, y, POST_TOP - 0.1) for x, y in hp]); above = inside(base, [(x, y, POST_TOP + 0.1) for x, y in hp])
    report(top.all() and not above.any(), f"4 flat posts, top z={POST_TOP:.1f} = Pi underside {PI_Z} - kit screw head "
           f"{BOT_HEAD_H}; no screw holes in the posts")
    env = stack_envelope()
    vb, vl = inter(base, env).volume, inter(lid, env).volume
    report(vb < 1e-3 and vl < 1e-3, f"stack envelope (Pi, cooler, HAT, spacers, RJ45/USB-A, USB-C, micro-HDMI) vs base: "
           f"{vb:.4f} mm^3, vs lid: {vl:.4f} mm^3 overlap")
    worst = max(inter(base, union(edge_connectors(dx, dy))).volume
                for dx in (-PI_CLR, PI_CLR) for dy in (-PI_CLR, PI_CLR))
    report(worst < 1e-3, f"USB-C ({USBC_BOX[2]} mm) and micro-HDMI ({HDMI_BOX[2]} mm) overhangs with the Pi pushed to each "
           f"fence corner (+-{PI_CLR}): worst overlap {worst:.4f} mm^3 (HDMI relief {HDMI_RELIEF} mm deep, "
           f"{WALL - HDMI_RELIEF:.1f} mm of wall left; worst-case gap {Y_GAP - PI_CLR + HDMI_RELIEF - HDMI_BOX[2]:.2f} mm)")
    heads = stack_envelope(heads=True)
    vp = inter(lid, heads).volume
    exp = 4 * math.pi * (KIT_HEAD_D / 2) ** 2 * PRESS
    report(abs(vp - exp) / exp < 0.05, f"lid pads press the 4 top screw heads by {PRESS} mm (overlap {vp:.1f} mm^3, "
           f"expected {exp:.1f}); pad bottom z={PAD_Z:.2f}, HAT top {HAT_TOP:.2f}")
    print(f"  stack: post top {POST_TOP:.1f}, Pi {PI_Z:.1f}-{PI_TOP:.1f}, cooler top {PI_TOP + COOLER_H:.2f} "
          f"(gap to HAT {STACK - COOLER_H:.2f}), HAT {HAT_Z:.1f}-{HAT_TOP:.1f}, lid underside = wall top {LID_Z:.1f}")

    print("== camera")
    cam = [cam_yz(u, v) for u, v in CAM_HOLES]
    hole_empty = ~inside(base, [(DATUM_X - 1.0, y, z) for y, z in cam])
    boss_solid = inside(base, [(DATUM_X - 1.0, y + M25_HOLE / 2 + 0.4, z) for y, z in cam])
    dy, dz = abs(cam[1][0] - cam[0][0]), cam[2][1] - cam[0][1]
    report(hole_empty.all() and boss_solid.all() and abs(dy - 21) < 1e-9 and abs(dz - 12.5) < 1e-9,
           f"screw holes (case y,z) {[(round(y, 2), round(z, 2)) for y, z in cam]}: spacing {dy:.2f} x {dz:.2f} "
           f"(drawing 21.00 x 12.50); hole open {int(hole_empty.sum())}/4, standoff wall solid {int(boss_solid.sum())}/4")
    datum = inside(base, [(DATUM_X - 0.05, y + 1.8, z) for y, z in cam]); clear = inside(base, [(DATUM_X + 0.05, y + 1.8, z) for y, z in cam])
    fn, tc, ar = base.face_normals, base.triangles_center, base.area_faces
    dsel = (np.abs(tc[:, 0] - DATUM_X) < 1e-6) & (fn[:, 0] > 0.999999)
    report(datum.all() and not clear.any() and STANDOFF_D <= CAM_PAD_D,
           f"datum: 4 standoff faces coplanar at x={DATUM_X:.2f} ({ar[dsel].sum():.1f} mm^2 of +X faces), "
           f"standoff dia {STANDOFF_D} inside the {CAM_PAD_D} pad keep-out; board front components clear the wall by {STANDOFF}")
    front = (np.abs(tc[:, 0]) < 1e-6) & (fn[:, 0] < -0.999999)
    bottom = (np.abs(tc[:, 2]) < 1e-6) & (fn[:, 2] < -0.999999)
    def mean_n(sel):
        n = (fn[sel] * ar[sel, None]).sum(0); return n / np.linalg.norm(n)
    nf, nb, nd = mean_n(front), mean_n(bottom), mean_n(dsel)
    a_fb = math.degrees(math.acos(np.clip(abs(nf @ nb), 0, 1)))
    a_df = math.degrees(math.acos(np.clip(-nd @ nf, -1, 1)))          # lens axis (-datum normal) vs front-face normal
    a_db = math.degrees(math.acos(np.clip(abs(nd @ nb), 0, 1)))
    report(abs(a_fb - 90) < 1e-6 and a_df < 1e-6 and abs(a_db - 90) < 1e-6,
           f"lens axis (= -datum normal) is {a_df:.4f} deg from the front-face normal (i.e. 90 deg to the front face), "
           f"{a_db:.4f} deg to the bottom normal (parallel to the bottom); front vs bottom face {a_fb:.4f} deg; "
           f"axis at y={cam_yz(*CAM_LENS)[0]:.2f} (centre {W/2:.2f}), z={LENS_Z:.2f} above the bottom")
    # lens window: rays from the lens aperture edge at the half-diagonal field angle, from the lens tip to the face
    ly, lz = cam_yz(*CAM_LENS)
    th, tv = math.tan(math.radians(CAM_HFOV / 2)), math.tan(math.radians(CAM_VFOV / 2))
    pts = []
    for prot in LENS_PROTRUSION:
        xt = DATUM_X - prot
        for sy, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (0, 1), (-1, 0), (0, -1)):
            for t in np.linspace(0, 1, 9):
                x = xt * (1 - t) - 0.01 * t                       # lens tip -> just outside the front face
                d = xt - x
                pts.append((x, ly + sy * (LENS_R + d * th), lz + sz * (LENS_R + d * tv)))
    hit = inside(base, pts)
    margin = WINDOW / 2 - (LENS_R + (DATUM_X - LENS_PROTRUSION[0]) * math.hypot(th, tv))
    report(not hit.any(), f"lens window {WINDOW} x {WINDOW}: {len(pts)} field-of-view ray samples ({CAM_HFOV}x{CAM_VFOV} deg, "
           f"aperture r {LENS_R}) all clear; lens tip {DATUM_X - LENS_PROTRUSION[1]:.2f}-{DATUM_X - LENS_PROTRUSION[0]:.2f} mm "
           f"behind the front face; worst-case diagonal margin {margin:.2f} mm")
    vc = inter(base, camera_envelope()).volume
    report(vc < 1e-3 and WINDOW > SENSOR_HOUSING, f"camera board + {SENSOR_HOUSING} mm sensor housing + lens envelope vs base: "
           f"{vc:.4f} mm^3 overlap")
    # nut pockets: flats at AF/2, depth
    y, z = cam[0]
    a = NUT_POCKET_AF / 2
    p = inside(base, [(CAM_NUT_DEPTH - 0.1, y, z + a - 0.05), (CAM_NUT_DEPTH - 0.1, y, z + a + 0.1),
                      (CAM_NUT_DEPTH + 0.1, y + 2.0, z)])
    report(not p[0] and p[1] and p[2], f"camera nut pockets: {NUT_POCKET_AF:.2f} AF x {CAM_NUT_DEPTH:.2f} deep in the front face "
           f"for {NUT_AF} AF x {NUT_H} nuts ({NUT_CLR_AF:.2f} AF clearance); flat at {a:.3f} empty/solid as expected, pocket floor solid; "
           f"{FRONT - CAM_NUT_DEPTH:.2f} mm of wall behind the nut")
    nut_far = CAM_NUT_DEPTH - NUT_H            # nut pulled onto the pocket floor spans [nut_far, CAM_NUT_DEPTH]
    cs = choose_screw(HEAD_SEAT_X - nut_far, HEAD_SEAT_X - TIP_RECESS_MIN)
    tip = HEAD_SEAT_X - cs
    report(tip <= nut_far + 1e-9 and tip >= TIP_RECESS_MIN, f"camera screws M2.5x{cs}: head on the board back (x={HEAD_SEAT_X:.2f}), "
           f"grip board {CAM_T} + standoff {STANDOFF} + wall {FRONT - CAM_NUT_DEPTH:.1f} = {HEAD_SEAT_X - CAM_NUT_DEPTH:.2f}; "
           f"full {NUT_H} mm nut engagement, tip {tip:.2f} mm behind the front face")
    report(M25_HEAD_D <= CAM_PAD_D, f"socket head dia {M25_HEAD_D} sits inside the camera's {CAM_PAD_D} hole pad (no washer on the camera)")

    print("== lid screws")
    xs = [tower_xy(*t) for t in TOWERS]
    r = inside(base, [(x, y, (LNUT_SLOT0 + LNUT_TOP) / 2) for x, y in xs] + [(x, y, LNUT_TOP + LNUT_ROOF / 2 + 0.6) for x, y in xs]
               + [(x, y + NUT_POCKET_AF / 2 + 0.1, (LNUT_SLOT0 + LNUT_TOP) / 2) for x, y in xs])
    roof_solid = inside(base, [(x + 1.6, y, LNUT_TOP + LNUT_ROOF / 2) for x, y in xs])   # just beside the screw hole
    report(not r[:4].any() and not r[4:8].any() and r[8:].all(),
           f"4 corner towers at {[(round(x, 1), round(y, 1)) for x, y in xs]}: nut slot {NUT_POCKET_AF:.2f} wide x "
           f"{NUT_POCKET_H:.2f} tall (z {LNUT_SLOT0:.2f}-{LNUT_TOP:.2f}), screw hole open, slot walls solid")
    report(roof_solid.all(), f"slot roof {LNUT_ROOF} mm solid under the lid; corbel 45-degree underside from z={TOWER_Z0:.2f}, "
           f"full size from z={TOWER_FULL:.2f}")
    for washer in (0.0, WASHER_T):
        head = LID_Z + LID_T + washer
        nut_bot = LNUT_TOP - NUT_H                 # nut pulled up against the roof
        ls = choose_screw(head - nut_bot, head - (TOWER_Z0 + TOWER / 2 + M25_HOLE))   # tip stays inside the corbel
        tip = head - ls
        report(tip <= nut_bot + 1e-9, f"lid screws M2.5x{ls}{' + flat washer' if washer else ''}: grip lid {LID_T}"
               f"{' + washer ' + str(washer) if washer else ''} + roof {LNUT_ROOF} = {head - LNUT_TOP:.2f} mm, "
               f"engagement {min(NUT_H, LNUT_TOP - tip):.2f}/{NUT_H} mm, {nut_bot - tip:.2f} mm past the nut, tip z={tip:.2f} "
               f"inside the tower (corbel tip z={TOWER_Z0:.2f}, port jacks top z={PI_TOP + max(ETH_BOX[1], USB_BOX[1]):.2f})")

    print("== tripod nut")
    cx, cy = L / 2, W / 2
    pk = inside(base, [(cx, cy, NUT_FLOOR + QTR_NUT_H / 2), (cx, cy, NUT_FLOOR + QTR_NUT_H + NUT_ROOF / 2), (cx, cy, 1.0),
                       (cx + QTR_NUT_AF / math.sqrt(3) + 0.6, cy, NUT_FLOOR + 1)])
    report(not pk[0] and pk[1] and not pk[2] and pk[3],
           f"1/4\"-20 pocket {QTR_NUT_AF} AF x {QTR_NUT_H} (nut 11.11 x 5.56) at x={cx:.1f} y={cy:.1f}: pocket empty, "
           f"roof solid, screw hole open, wall solid; puck top z={NUT_FLOOR + QTR_NUT_H + NUT_ROOF:.1f} < Pi underside {PI_Z}")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return ok, cs, ls


def export_clean(m, path):
    """Export, reload the STL (float32, vertices merged) and drop zero-volume slivers that the float32
    rounding can split off the boolean result; refuse to keep a file that is not one watertight body."""
    m.export(path)
    r = trimesh.load(path)
    parent = np.arange(len(r.faces))            # union-find over shared edges (no scipy / networkx needed)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in r.face_adjacency:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(len(r.faces))])
    comps = [np.where(roots == k)[0] for k in np.unique(roots)]
    vols = [abs(r.submesh([c], append=True).volume) for c in comps]
    keep = [c for c, v in zip(comps, vols) if v > 1e-3]
    if len(keep) != len(comps):
        print(f"  {path.name}: dropped {len(comps) - len(keep)} zero-volume sliver(s) "
              f"({sum(len(c) for c, v in zip(comps, vols) if v <= 1e-3)} faces) left by the float32 STL export")
        r = r.submesh([np.concatenate(keep)], append=True)
        r.export(path)
        r = trimesh.load(path)
    assert r.is_watertight and r.is_winding_consistent and len(keep) == 1, f"{path.name} is not one watertight body"
    print(f"  {path.name}: reloaded STL is one watertight body, {r.volume / 1000:.1f} cm^3")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"trimesh\.triangles")  # trimesh: empty intersections
    base, lid = build_base(), build_lid()
    ok, _, _ = check(base, lid)
    assert ok
    lid_print = lid.copy()
    lid_print.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    lid_print.apply_translation(-lid_print.bounds[0])
    for m, name in ((base, "pi_case_base.stl"), (lid_print, "pi_case_lid.stl")):
        export_clean(m, OUT / name)
    print(f"wrote {OUT / 'pi_case_base.stl'}, {OUT / 'pi_case_lid.stl'}")
