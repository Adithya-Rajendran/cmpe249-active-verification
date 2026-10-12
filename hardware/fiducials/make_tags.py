"""Printable ArUco corner tags for the shelf rig (compartment #3).

Writes tags_letter.pdf / tags_letter.png (US Letter, 600 dpi, print at 100% / actual size).
Each card: a 5.0 cm DICT_4X4_50 marker with a 1.5 cm white margin, plus a 2 cm labelled
flap that gets taped to the front face of a shelf board so the marker sits in the
shelf's front plane. A 10 cm scale bar checks the print scale.

Run: uv run --no-sync python hardware/fiducials/make_tags.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 600
CM = DPI / 2.54
TAG_CM = 5.0
MARGIN_CM = 1.5
FLAP_CM = 2.0
PAGE_W, PAGE_H = int(8.5 * DPI), int(11 * DPI)
OUT = Path(__file__).resolve().parent

# id, short name, flap side, placement text
CARDS = [
    (0, "LOWER LEFT", "top", "tape flap to FRONT FACE of the cup shelf board, left end; card hangs DOWN"),
    (1, "LOWER RIGHT", "top", "tape flap to FRONT FACE of the cup shelf board, right end; card hangs DOWN"),
    (2, "UPPER LEFT", "bottom", "tape flap to FRONT FACE of the board above the cups, left end; card stands UP"),
    (3, "UPPER RIGHT", "bottom", "tape flap to FRONT FACE of the board above the cups, right end; card stands UP"),
    (4, "SPARE", "bottom", "spare - do not place unless asked"),
    (5, "SPARE", "bottom", "spare - do not place unless asked"),
]


def px(cm):
    return int(round(cm * CM))


def font(size_cm):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, px(size_cm))
        except OSError:
            pass
    return ImageFont.load_default()


def card(tag_id, name, flap, text):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, tag_id, px(TAG_CM))
    side = px(TAG_CM + 2 * MARGIN_CM)
    h = side + px(FLAP_CM)
    im = Image.new("L", (side, h), 255)
    top = px(FLAP_CM) if flap == "top" else 0
    im.paste(Image.fromarray(marker), (px(MARGIN_CM), top + px(MARGIN_CM)))
    dr = ImageDraw.Draw(im)
    fy = 0 if flap == "top" else side
    dr.line([(0, fy + (px(FLAP_CM) if flap == "top" else 0)), (side, fy + (px(FLAP_CM) if flap == "top" else 0))],
            fill=170, width=4)  # fold/flap line
    dr.text((px(0.2), fy + px(0.15)), f"ID {tag_id}  {name}", font=font(0.42), fill=0)
    words, lines, cur = text.split(), [], ""
    for w in words:
        if dr.textlength(cur + " " + w, font=font(0.24)) > side - px(0.4):
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    for k, ln in enumerate(lines[:3]):
        dr.text((px(0.2), fy + px(0.75) + k * px(0.32)), ln, font=font(0.24), fill=0)
    dr.rectangle([0, 0, side - 1, h - 1], outline=120, width=3)  # cut line
    return im


def page_with(cards_spec, title):
    page = Image.new("L", (PAGE_W, PAGE_H), 255)
    dr = ImageDraw.Draw(page)
    x0, y0 = px(1.0), px(1.0)
    dr.text((x0, y0), title, font=font(0.4), fill=0)
    # 10 cm scale bar
    by = y0 + px(0.8)
    dr.rectangle([x0, by, x0 + px(10), by + px(0.3)], fill=0)
    for k in range(11):
        dr.line([(x0 + px(k), by + px(0.3)), (x0 + px(k), by + px(0.6))], fill=0, width=4)
    dr.text((x0 + px(10.3), by - px(0.05)), "must measure 10.0 cm", font=font(0.3), fill=0)
    cards = [card(*c) for c in cards_spec]
    cw, ch = cards[0].size
    gx, gy = px(0.6), px(0.6)
    sy = by + px(1.2)
    for k, im in enumerate(cards):
        r, c = divmod(k, 2)
        x, y = x0 + c * (cw + gx), sy + r * (ch + gy)
        assert x + cw <= PAGE_W - px(0.6) and y + ch <= PAGE_H - px(0.6), "card off page"
        page.paste(im, (x, y))
    return page


def main():
    t = "CMPE249 shelf corner tags - print at 100% (Actual size), no scaling"
    p1 = page_with(CARDS[:4], t + " - page 1")
    p2 = page_with(CARDS[4:], t + " - page 2 (spares)")
    p1.save(OUT / "tags_letter.png", dpi=(DPI, DPI))
    p1.convert("RGB").save(OUT / "tags_letter.pdf", resolution=DPI, save_all=True,
                           append_images=[p2.convert("RGB")])
    page = Image.fromarray(np.vstack([np.array(p1), np.array(p2)]))

    # self-check: detect every marker on the rendered page
    g = np.array(page)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = det.detectMarkers(cv2.resize(g, None, fx=0.25, fy=0.25))
    found = sorted(int(i) for i in ids.ravel()) if ids is not None else []
    side_cm = [float(np.linalg.norm(c[0][0] - c[0][1]) * 4 / CM) for c in corners]
    report = {"dictionary": "DICT_4X4_50", "tag_cm": TAG_CM, "margin_cm": MARGIN_CM, "flap_cm": FLAP_CM,
              "dpi": DPI, "ids_expected": [c[0] for c in CARDS], "ids_detected": found,
              "detected_side_cm": [round(s, 2) for s in side_cm],
              "placement": {c[0]: c[3] for c in CARDS}}
    (OUT / "tags_letter.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
