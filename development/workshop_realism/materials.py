"""Build the workshop's color maps from pinned CC0 photographs and original labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SOURCES = {
    "concrete_floor_worn_001": "6a3645f19f2ac4b7f60f43419c5d09ec",
    "painted_brick": "512a7cccab9caf88e7a7833fb0f28d81",
    "painted_metal_shutter": "71f3093741ac82b1e4ab10fd25bc367d",
    "plywood": "ce20176e90b0984ca7dec366832e8198",
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def recolor(image, center, gain):
    luminance = np.asarray(ImageOps.grayscale(image), dtype=float)
    neutral = luminance - np.mean(luminance)
    return Image.fromarray(
        np.clip(np.asarray(center) + neutral[:, :, None] * gain, 0, 255).astype("uint8")
    )


def sign(size, base, band, title, lines):
    plate = Image.new("RGB", size, base)
    d = ImageDraw.Draw(plate)
    w, h = size
    d.rounded_rectangle((4, 4, w - 5, h - 5), radius=10, outline=(59, 63, 62), width=6)
    d.rectangle((10, 10, w - 11, int(h * 0.35)), fill=band)
    d.text(
        (w * 0.5, h * 0.18),
        title,
        anchor="mm",
        font=ImageFont.truetype(FONT, int(h * 0.19)),
        fill="#faf8eb",
    )
    for idx, line in enumerate(lines):
        d.text(
            (w * 0.5, h * (0.52 + 0.24 * idx)),
            line,
            anchor="mm",
            font=ImageFont.truetype(FONT, int(h * 0.14)),
            fill="#262e30",
        )
    for x in (15, w - 16):
        for y in (15, h - 16):
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#8f999e", outline="#4b5355")
    return plate


def build(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    photos, inventory = {}, {}
    for name, expected in SOURCES.items():
        path = source / (name + ".jpg")
        data = path.read_bytes()
        if hashlib.md5(data).hexdigest() != expected:
            raise ValueError(f"Photograph differs from the supplier file: {name}")
        with Image.open(path) as image:
            photos[name] = image.convert("RGB").resize(
                (1024, 1024),
                Image.Resampling.LANCZOS
                if hasattr(Image, "Resampling")
                else Image.LANCZOS,
            )
        inventory[name] = {
            "asset": "https://polyhaven.com/a/" + name,
            "url": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/"
            + name
            + "/"
            + name
            + "_diff_2k.jpg",
            "source_sha256": hashlib.sha256(data).hexdigest(),
        }
    brick = photos["painted_brick"]
    metal = photos["painted_metal_shutter"]
    maps = {
        "concrete": recolor(photos["concrete_floor_worn_001"], (144, 139, 127), 2.15),
        "upper_brick": recolor(brick, (195, 191, 173), 0.95),
        "lower_brick": recolor(brick, (87, 115, 109), 0.69),
        "door_skin": recolor(metal, (116, 137, 140), 1.85),
        "ply": photos["plywood"],
        "service_plate": sign(
            (640, 256), "#e8e4cf", "#37575a", "FACILITIES", ["SERVICE STORES", "BAY 04"]
        ),
        "door_plate": sign(
            (600, 260), "#e6e3d5", "#9b6228", "CAUTION", ["SLIDING DOOR", "KEEP CLEAR"]
        ),
        "cabinet_plate": sign(
            (512, 256), "#eee6cd", "#475960", "ELECTRICAL", ["PANEL 04", "208 V AC"]
        ),
    }
    carton = recolor(photos["plywood"], (174, 138, 91), 0.25).resize((720, 420))
    d = ImageDraw.Draw(carton)
    d.rectangle((326, 0, 380, 420), fill=(164, 126, 79))
    d.rectangle((39, 68, 272, 278), fill=(226, 224, 207))
    d.text((53, 81), "STORES", font=ImageFont.truetype(FONT, 28), fill="#343737")
    d.text((53, 122), "PARTS / 04", font=ImageFont.truetype(FONT, 20), fill="#43464a")
    for i in range(39):
        x = 54 + 5 * i
        d.rectangle((x, 180, x + 1 + (i * 7 % 3), 244), fill="#43423c")
    d.text((470, 270), "↑   ↑", font=ImageFont.truetype(FONT, 49), fill="#554b35")
    maps["carton_face"] = carton
    mask = np.asarray(photos["concrete_floor_worn_001"].convert("L"), dtype=float)
    worn = np.clip((mask - np.percentile(mask, 20)) / 17, 0, 1)[..., None]
    maps["floor_paint"] = Image.fromarray(
        (
            np.asarray(maps["concrete"]) * (1 - worn) + np.array([199, 153, 53]) * worn
        ).astype("uint8")
    )
    for name, im in maps.items():
        im.save(dest / (name + ".png"), optimize=True)
    (dest / "sources.json").write_text(
        json.dumps(
            {
                "license": "https://polyhaven.com/license",
                "originals": inventory,
                "maps": {
                    n: hashlib.sha256((dest / (n + ".png")).read_bytes()).hexdigest()
                    for n in sorted(maps)
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    build(args.source, args.destination)
