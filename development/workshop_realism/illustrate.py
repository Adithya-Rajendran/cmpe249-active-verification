"""Show the first declared workshop layout at the scored camera positions."""

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from development.workshop_realism.release import OLD, read, sha, write_once

DARK = (25, 42, 48)
SECOND = (57, 78, 87)
BG = (249, 248, 244)
FONTS = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    return ImageFont.truetype(
        str(FONTS / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size
    )


def tile(canvas, path, x, y, w, h):
    with Image.open(path) as im:
        canvas.paste(
            im.convert("RGB").resize(
                (w, h), getattr(Image, "Resampling", Image).LANCZOS
            ),
            (x, y),
        )


def appearance(run, out):
    layout = "COR01"
    view = "P3"
    w, h = 480, 270
    pad = 20
    step = w + 12
    img = Image.new("RGB", (3 * step + 2 * pad - 12, 741), BG)
    label = ImageDraw.Draw(img)
    label.text(
        (pad, 12), "The same camera, doorway and bottle", font=font(32, True), fill=DARK
    )
    label.text(
        (pad, 56),
        "First declared layout  /  Frontal revisit  /  Bottle retained in every panel",
        font=font(19),
        fill=SECOND,
    )
    titles = [
        "Previous corridor · door open",
        "Furnished workshop · door open",
        "Furnished workshop · door closed",
    ]
    for j, title in enumerate(titles):
        label.text((pad + step * j, 102), title, font=font(19, True), fill=DARK)
    files = []
    for i, (engine, title) in enumerate([("mujoco", "MuJoCo"), ("isaac", "Isaac Sim")]):
        y = 139 + 300 * i
        for j, (base, door) in enumerate(
            [(OLD, "clear"), (run, "clear"), (run, "placed")]
        ):
            p = (
                base
                / "operator"
                / engine
                / layout
                / "D6/rgb"
                / f"{door}_present_{view}.png"
            )
            tile(img, p, pad + step * j, y, w, h)
            files.append(
                {
                    "engine": engine,
                    "layout": layout,
                    "view": view,
                    "door": door,
                    "source": str(p.relative_to(base)),
                    "study": base.name,
                    "sha256": sha(p),
                }
            )
        label.text((pad, y + h + 3), title, font=font(16, True), fill=SECOND)
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / "appearance-comparison.jpg", quality=92, subsampling=0)
    write_once(out / "appearance-sources.json", files)


def navigation(run, out):
    folder = run / "robot/navigation-demo"
    evaluation = read(folder / "summary.json")
    image = Image.new("RGB", (700, 520), BG)
    labels = ImageDraw.Draw(image)
    labels.text(
        (18, 10), "The robot's first camera image", fill=DARK, font=font(26, True)
    )
    labels.text(
        (18, 49),
        "MuJoCo  /  Same start pose in two reserved layouts",
        fill=SECOND,
        font=font(16),
    )
    for i, layout in enumerate(evaluation["reserved_layouts"]):
        for j, opened in enumerate((False, True)):
            case = next(
                r
                for r in evaluation["episodes"]
                if r["layout"] == layout
                and r["door_open"] == opened
                and r["lateral_m"] == -0.08
            )
            path = folder / f"{layout}-{int(opened)}-0.png"
            if sha(path) != case["starting_frame_sha256"]:
                raise ValueError("Saved frame differs from the evaluated policy input")
            x, y = 18 + j * 344, 78 + i * 222
            with Image.open(path) as frame:
                image.paste(frame, (x, y))
            outcome = {"safe_stop": "stopped safely", "crossed": "crossed the doorway"}[
                case["outcome"]
            ]
            labels.text(
                (x, y + 184),
                f"{layout}   {case['selected_action'].upper()}   ·   {outcome}",
                fill=DARK,
                font=font(15),
            )
    image.save(out / "training-observations.jpg", quality=94, subsampling=0)


def partial(run, out):
    folder = run / "robot/partial-frames"
    rows = read(run / "robot/partial-openings.json")["cases"]
    img = Image.new("RGB", (700, 553), BG)
    labels = ImageDraw.Draw(img)
    labels.text(
        (18, 10), "How much room does the robot have?", fill=DARK, font=font(26, True)
    )
    labels.text(
        (18, 49),
        "MuJoCo  /  Reserved layout COR07  /  Same fitted policy",
        fill=SECOND,
        font=font(16),
    )
    for j, (fraction, title) in enumerate(
        [(0.5, "Door halfway open"), (0.75, "Door three-quarters open")]
    ):
        x = 18 + 344 * j
        labels.text((x, 82), title, fill=DARK, font=font(17, True))
        for i, lateral in enumerate([-0.08, 0.08]):
            case = next(
                r
                for r in rows
                if (r["layout"], r["door_fraction"], r["lateral_m"])
                == ("COR07", fraction, lateral)
            )
            y = 109 + 218 * i
            with Image.open(
                folder / f"COR07-{int(100 * fraction):03d}-{i}.png"
            ) as frame:
                if (
                    hashlib.sha256(frame.tobytes()).hexdigest()
                    != case["initial_rgb_pixels_sha256"]
                ):
                    raise ValueError(
                        "Partial-opening observation differs from the tested input"
                    )
                img.paste(frame, (x, y))
            outcome = {"contact": "hit the door", "crossed": "crossed"}[
                case["policy"]["outcome"]
            ]
            labels.text(
                (x, y + 184),
                f"Start {100 * lateral:+.0f} cm, yaw {case['heading_deg']:+}°  ·  {outcome}",
                fill=DARK,
                font=font(15),
            )
    img.save(out / "partial-openings.jpg", quality=94, subsampling=0)


def exposed(run, out):
    from development.paper_audit.inputs_build import box_from_mask

    folder = run / "scoring/fresh-packet"
    cases = {r["id"]: r for r in read(folder / "cases.json")}
    with (out / "decisions.csv").open() as source:
        selected = [
            r
            for r in csv.DictReader(source)
            if (r["method"], r["layout"], r["view"], r["state"])
            == ("vggt_qwen", "COR04", "P4", "clear_present")
        ]
    if len(selected) != 2 or any(r["answer"] != "REMOVED" for r in selected):
        raise ValueError("The shared exposed-drill example did not reproduce")
    sheet = Image.new("RGB", (982, 581), BG)
    labels = ImageDraw.Draw(sheet)
    labels.text(
        (16, 8),
        "A visible drill was called removed in both simulators",
        font=font(29, True),
        fill=DARK,
    )
    labels.text(
        (16, 54),
        "COR04   /   95 cm mast   /   Same open doorway and retained drill",
        font=font(17),
        fill=SECOND,
    )
    for x, title in [(16, "Earlier image"), (371, "Revisit"), (726, "Revisit detail")]:
        labels.text((x, 92), title, fill=DARK, font=font(18, True))
    files = []
    for i, row in enumerate(selected):
        e, uid = row["engine"], row["id"]
        case = cases[uid]
        path = run / "operator" / e / "COR04/D6/masks/target_clear_present_P4.png"
        box = box_from_mask(path)
        y = 122 + 214 * i
        for col, (role, target_box) in enumerate(
            [("reference", case["box_xyxy"]), ("query", box)]
        ):
            raw = folder / case[role]
            if sha(raw) != case[role + "_sha256"]:
                raise ValueError("Changed example image")
            with Image.open(raw) as im:
                x = 16 + col * 355
                sheet.paste(
                    im.resize((336, 189), getattr(Image, "Resampling", Image).LANCZOS),
                    (x, y),
                )
                l, t, r, b = target_box
                labels.rectangle(
                    (
                        x + l * 336 / 1200,
                        y + t * 189 / 675,
                        x + r * 336 / 1200,
                        y + b * 189 / 675,
                    ),
                    outline=(255, 214, 55),
                    width=2,
                )
                if role == "query":
                    crop = im.crop((l - 13, t - 13, r + 13, b + 13))
                    zoom = ImageOps.contain(
                        crop,
                        (235, 189),
                        method=getattr(Image, "Resampling", Image).LANCZOS,
                    )
                    sheet.paste(
                        zoom,
                        (726 + (235 - zoom.width) // 2, y + (189 - zoom.height) // 2),
                    )
        name = "Isaac Sim" if e == "isaac" else "MuJoCo"
        labels.text(
            (16, y + 191),
            f"{name}   ·   Qwen answered REMOVED",
            font=font(15, True),
            fill=DARK,
        )
        files.append(
            {
                "engine": e,
                "id": uid,
                "reference_sha256": case["reference_sha256"],
                "query_sha256": case["query_sha256"],
                "evaluator_query_mask_sha256": sha(path),
                "evaluator_query_box": box,
            }
        )
    labels.text(
        (16, 555),
        "Outlines and detail are for inspection. The verifier received the original revisit.",
        font=font(15),
        fill=SECOND,
    )
    sheet.save(out / "exposed-drill.jpg", quality=94, subsampling=0)
    write_once(out / "exposed-drill-sources.json", files)


def approaches(run, out):
    with (out / "pairs.csv").open(newline="") as stream:
        results = {
            r["view"]: r
            for r in csv.DictReader(stream)
            if (r["layout"], r["engine"], r["door"], r["method"])
            == ("COR02", "mujoco", "placed", "vggt_qwen")
        }
    packet = run / "scoring/fresh-packet"
    cases = {c["id"]: c for c in read(packet / "cases.json")}
    w, h = 420, 236
    pad = 18
    step = w + 12
    img = Image.new("RGB", (3 * step + 2 * pad - 12, 961), BG)
    label = ImageDraw.Draw(img)
    label.text(
        (pad, 10), "One drill, three camera mounts", font=font(32, True), fill=DARK
    )
    label.text(
        (pad, 54),
        "MuJoCo  /  First drill layout, COR02  /  Each revisit was scored separately",
        font=font(18),
        fill=SECOND,
    )
    for j, title in enumerate(
        [
            "Earlier view · drill marked",
            "Retained behind the door",
            "Removed behind the door",
        ]
    ):
        label.text((pad + j * step, 94), title, font=font(18, True), fill=DARK)
    selected = []
    for i, (v, desc) in enumerate(
        [("P3", "Straight ahead"), ("P4", "95 cm mast"), ("P5", "48 cm mount")]
    ):
        r = results[v]
        held = cases[r["present_id"]]
        gone = cases[r["removed_id"]]
        y = 138 + i * 272
        for j, (case, role) in enumerate(
            [(held, "reference"), (held, "query"), (gone, "query")]
        ):
            p = packet / case[role]
            tile(img, p, pad + step * j, y, w, h)
            if role == "reference":
                b = case["box_xyxy"]
                label.rectangle(
                    (
                        pad + b[0] * w / 1200,
                        y + b[1] * h / 675,
                        pad + b[2] * w / 1200,
                        y + b[3] * h / 675,
                    ),
                    outline=(255, 207, 40),
                    width=3,
                )
            selected.append(
                {"view": v, "role": role, "id": case["id"], "rgb_sha256": sha(p)}
            )
        label.text((pad, y + h + 4), f"{v}   {desc}", font=font(17, True), fill=DARK)
        for j, member in [(1, "present"), (2, "removed")]:
            c = r[member + "_call"]
            a = r[member + "_answer"]
            phrase = (
                "Geometry check → UNKNOWN"
                if c == "registration_gate_unknown"
                else f"Qwen → {a}"
            )
            label.text(
                (pad + step * j, y + h + 4), phrase, font=font(17, True), fill=SECOND
            )
    img.save(out / "robot-camera-decisions.jpg", quality=92, subsampling=0)
    write_once(out / "robot-camera-sources.json", selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    figures = {
        "exposed": exposed,
        "partial": partial,
        "appearance": appearance,
        "approaches": approaches,
        "navigation": navigation,
    }
    parser.add_argument("figure", choices=figures)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    figures[a.figure](a.run, a.out)
