"""Shared helpers for the perception smoke test and the reference-crop matching check.

Image entries come from a JSON manifest. Each entry names a source JPEG, a list of image
operations ("ops") and an optional hand-annotated target box in *source* pixel coordinates
("gt_src"). The same ops are applied to the image and to the box, so every derived variant
(crop, blur, roll, ...) carries a box in its own pixel frame.

Supported ops, applied in order:
  ["rot180"]                    lossless 180 degree rotation (camera mounted upside down)
  ["crop", x0, y0, x1, y1]      crop to that rectangle (pixel coords of the current image)
  ["blur", sigma]               Gaussian blur, sigma in pixels (synthetic defocus)
  ["roll", deg]                 rotate counter-clockwise by deg, canvas expanded, grey fill
  ["gain", factor]              multiply RGB values by factor, clip to [0, 255]
  ["fill", x0, y0, x1, y1, r, g, b]
                                paint a flat rectangle. Used for synthetic truncation (grey 114,
                                the same value ultralytics uses for letterbox padding, from the
                                frame edge) and synthetic occlusion (a dark block over the target).
                                If the rectangle spans the box's full height and covers its left or
                                right end, the box shrinks to the still-visible part (modal box).
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKPOINTS = REPO / "checkpoints"


def configure_ultralytics() -> None:
    """Keep ultralytics settings, weights and runs inside the repo; no telemetry, no auto-install.

    Must run before `import ultralytics`. The settings file lives at
    checkpoints/ultralytics_config/Ultralytics/settings.json (gitignored) with sync=false and
    weights_dir=checkpoints/.
    """
    os.environ.setdefault("YOLO_CONFIG_DIR", str(CHECKPOINTS / "ultralytics_config"))
    # ultralytics falls back to /tmp/Ultralytics if the directory does not exist yet
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"


def pin_ultralytics_settings() -> None:
    """Call right after importing ultralytics: ultralytics (re)creates its settings file with
    sync=true and weights_dir=<cwd>/weights when the file is missing or from another version, so
    force the project values before any predict call (analytics events fire only on predict,
    train, val and export)."""
    from ultralytics import settings

    want = {"sync": False, "weights_dir": str(CHECKPOINTS), "runs_dir": str(REPO / "runs" / "ultralytics"),
            "datasets_dir": str(REPO / "data" / "datasets"), "clearml": False, "comet": False, "dvc": False,
            "mlflow": False, "raytune": False, "tensorboard": False, "wandb": False}
    if any(settings.get(k) != v for k, v in want.items()):
        settings.update(want)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


PROVENANCE_PACKAGES = ["torch", "torchvision", "ultralytics", "clip", "transformers", "open_clip_torch",
                       "huggingface_hub", "timm", "numpy", "pillow", "opencv-python", "matplotlib"]


def provenance(manifest: str | Path | None = None) -> dict:
    """Identifiers needed to reproduce a run: sha256 of the perception code, uv.lock and the manifest
    (the code is not necessarily committed, so a git hash alone would not identify it), the git HEAD,
    and the installed versions of every library that can change the numbers."""
    import importlib.metadata
    import subprocess

    def version(pkg):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except OSError:
        head = None
    code = sorted(REPO.glob("perception/**/*.py"))
    out = {"git_head": head,
           "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in code},
           "uv_lock_sha256": sha256(REPO / "uv.lock") if (REPO / "uv.lock").exists() else None,
           "packages": {p: version(p) for p in PROVENANCE_PACKAGES}}
    if manifest is not None:
        out["manifest"] = {"path": str(Path(manifest)), "sha256": sha256(manifest)}
    return out


# ----------------------------------------------------------------------------- manifest


def load_manifest(path: str | Path) -> dict:
    """Expand glob entries and return {"catalog": [...], "images": [entry, ...]}."""
    m = json.loads(Path(path).read_text())
    out = []
    for e in m["images"]:
        if "glob" in e:
            files = sorted(glob.glob(str(REPO / e["glob"])))
            if not files:
                raise FileNotFoundError(f"glob matched nothing: {e['glob']}")
            for f in files:
                x = {k: v for k, v in e.items() if k not in ("glob", "id_prefix")}
                x["id"] = f"{e['id_prefix']}{Path(f).stem}"
                x["path"] = str(Path(f).relative_to(REPO))
                out.append(x)
        else:
            out.append(dict(e))
    ids = [e["id"] for e in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate image ids: {sorted(dup)}")
    m["images"] = out
    return m


# ----------------------------------------------------------------------------- image ops


def apply_ops(img, ops, box=None):
    """Apply ops to a PIL image and (optionally) an axis-aligned box [x0, y0, x1, y1].

    For "roll" the returned box is the axis-aligned bounding box of the rotated rectangle, which
    is looser than a tight box around the rotated object.
    """
    from PIL import Image, ImageFilter

    for op in ops or []:
        name, *args = op
        W, H = img.size
        if name == "rot180":
            img = img.transpose(Image.Transpose.ROTATE_180)
            if box is not None:
                x0, y0, x1, y1 = box
                box = [W - x1, H - y1, W - x0, H - y0]
        elif name == "crop":
            x0, y0, x1, y1 = [int(round(a)) for a in args]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            img = img.crop((x0, y0, x1, y1))
            if box is not None:
                bx0, by0, bx1, by1 = box
                box = [max(bx0 - x0, 0), max(by0 - y0, 0), min(bx1 - x0, x1 - x0), min(by1 - y0, y1 - y0)]
                if box[2] <= box[0] or box[3] <= box[1]:
                    box = None
        elif name == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=float(args[0])))
        elif name == "fill":
            from PIL import ImageDraw

            fx0, fy0, fx1, fy1 = [float(a) for a in args[:4]]
            color = tuple(int(c) for c in args[4:7])
            img = img.copy()
            ImageDraw.Draw(img).rectangle([fx0, fy0, fx1 - 1, fy1 - 1], fill=color)
            if box is not None and fy0 <= box[1] and fy1 >= box[3]:
                if fx0 <= box[0] < fx1:
                    box = [fx1, box[1], box[2], box[3]]
                elif fx0 < box[2] <= fx1:
                    box = [box[0], box[1], fx0, box[3]]
                if box[2] <= box[0]:
                    box = None
        elif name == "gain":
            f = float(args[0])
            img = img.point(lambda v: max(0, min(255, int(round(v * f)))))
        elif name == "roll":
            deg = float(args[0])
            img2 = img.rotate(deg, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(114, 114, 114))
            if box is not None:
                # PIL rotates counter-clockwise about the centre; expand=True recentres on the new canvas
                cx, cy = W / 2, H / 2
                ncx, ncy = img2.size[0] / 2, img2.size[1] / 2
                t = math.radians(deg)
                pts = []
                for px, py in [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]:
                    dx, dy = px - cx, py - cy
                    # image y points down, so a CCW rotation on screen is (x, y) -> (x cos + y sin, -x sin + y cos)
                    pts.append((ncx + dx * math.cos(t) + dy * math.sin(t), ncy - dx * math.sin(t) + dy * math.cos(t)))
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                box = [min(xs), min(ys), max(xs), max(ys)]
            img = img2
        else:
            raise ValueError(f"unknown op {op}")
    return img, box


def load_entry(e: dict):
    """Return (PIL RGB image after ops, gt box after ops or None)."""
    from PIL import Image

    img = Image.open(REPO / e["path"]).convert("RGB")
    img, box = apply_ops(img, e.get("ops", []), e.get("gt_src"))
    return img, box


# ----------------------------------------------------------------------------- boxes


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def round_box(b, nd=1):
    return None if b is None else [round(float(v), nd) for v in b]
