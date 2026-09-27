"""Perception front-end smoke test: open-vocabulary and closed-set detectors on a list of images.

For every image in a manifest (see perception/common.py) and every detector, this records all
detections above a low confidence floor, whether a target-class box overlaps the hand-annotated
target box (IoU >= 0.5), and the warm per-image latency on the GPU. It writes:

  <out>/env.json            library versions, GPU, checkpoint and image sha256
  <out>/detections.jsonl    one record per (image, model) with every detection
  <out>/detections.csv      one row per detection
  <out>/summary.csv         one row per (image, model): hit confidence, false positives, latency
  <out>/overlays/<model>/<image_id>.jpg   downscaled overlays (hand box dashed, detections >= 0.10)

This is a smoke test on one object in one scene. The confidences are not calibrated probabilities
and the numbers here say nothing about accuracy on a real arrangement set.

Usage (from the repo root):
  uv run python perception/smoke_test.py --manifest perception/manifests/2026-09-26_smoke.json \
      --out runs/2026-09-26-perception-smoke/detect
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CHECKPOINTS, REPO, configure_ultralytics, iou, load_entry, load_manifest,  # noqa: E402
                    pin_ultralytics_settings, provenance, round_box, sha256)

configure_ultralytics()
import ultralytics  # noqa: E402

pin_ultralytics_settings()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import ImageDraw, ImageFont  # noqa: E402

MODELS = {
    # name: (kind, weights file, role)
    "yolov8s-worldv2": ("world", "yolov8s-worldv2.pt", "open vocabulary; the proposal's 'small YOLO-World checkpoint'"),
    "yoloe-26s-seg": ("yoloe", "yoloe-26s-seg.pt", "open vocabulary; YOLOE-26 small, the successor in ultralytics 8.4 (text prompts, boxes only used)"),
    "yolo11s": ("coco", "yolo11s.pt", "closed-set COCO-80 control"),
    "yolo26s": ("coco", "yolo26s.pt", "closed-set COCO-80 control, current ultralytics generation"),
}
HIT_IOU = 0.5
FP_IOU = 0.1


def load_model(name: str, catalog: list[str]):
    from ultralytics import YOLO, YOLOE

    kind, weights, _ = MODELS[name]
    path = CHECKPOINTS / weights
    if kind == "world":
        m = YOLO(str(path))
        m.set_classes(catalog)
    elif kind == "yoloe":
        m = YOLOE(str(path))
        m.set_classes(catalog, m.get_text_pe(catalog))
    else:
        m = YOLO(str(path))
    return m


def predict(model, bgr, args):
    r = model.predict(bgr, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou, device=args.device,
                      max_det=args.max_det, verbose=False)
    return r[0]


def timed_predict(model, bgr, args):
    times = []
    res = None
    for _ in range(args.repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = predict(model, bgr, args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return res, times


def get_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_overlay(img, dets, gt, targets, title, out_path, max_w=1280, min_conf=0.10, top=8):
    s = min(1.0, max_w / img.width)
    im = img.resize((round(img.width * s), round(img.height * s))) if s < 1 else img.copy()
    d = ImageDraw.Draw(im)
    font = get_font(max(14, im.width // 55))
    if gt is not None:
        x0, y0, x1, y1 = [v * s for v in gt]
        for i in range(0, int(x1 - x0), 16):  # dashed hand box
            d.line([(x0 + i, y0), (min(x0 + i + 8, x1), y0)], fill=(255, 255, 255), width=2)
            d.line([(x0 + i, y1), (min(x0 + i + 8, x1), y1)], fill=(255, 255, 255), width=2)
        for i in range(0, int(y1 - y0), 16):
            d.line([(x0, y0 + i), (x0, min(y0 + i + 8, y1))], fill=(255, 255, 255), width=2)
            d.line([(x1, y0 + i), (x1, min(y0 + i + 8, y1))], fill=(255, 255, 255), width=2)
    shown = [x for x in sorted(dets, key=lambda x: -x["conf"]) if x["conf"] >= min_conf][:top]
    for x in reversed(shown):
        col = (255, 140, 0) if x["cls"] in targets else (60, 160, 255)
        b = [v * s for v in x["box"]]
        d.rectangle(b, outline=col, width=3)
        label = f"{x['cls']} {x['conf']:.2f}"
        tb = d.textbbox((0, 0), label, font=font)
        ty = b[1] - (tb[3] - tb[1]) - 6 if b[1] > (tb[3] - tb[1]) + 6 else b[1] + 2
        d.rectangle([b[0], ty, b[0] + tb[2] - tb[0] + 6, ty + tb[3] - tb[1] + 6], fill=col)
        d.text((b[0] + 3, ty + 1), label, fill=(0, 0, 0), font=font)
    tb = d.textbbox((0, 0), title, font=font)
    d.rectangle([0, im.height - (tb[3] - tb[1]) - 10, tb[2] + 10, im.height], fill=(0, 0, 0))
    d.text((5, im.height - (tb[3] - tb[1]) - 7), title, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.01, help="confidence floor for *recording* detections")
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats", type=int, default=5, help="timed runs per image (median reported)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--only", nargs="*", help="restrict to these image ids")
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(args.manifest)
    catalog = man["catalog"]
    entries = [e for e in man["images"] if not args.only or e["id"] in args.only]
    if not entries:
        sys.exit(f"no manifest entries selected (--only {args.only})")

    env = {
        "command": " ".join(sys.argv),
        "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ultralytics": ultralytics.__version__,
        "args": vars(args),
        "catalog": catalog,
        "hit_rule": f"target-class box with IoU >= {HIT_IOU} against the hand box; false positive = target-class box with IoU < {FP_IOU}",
        "models": {n: {"kind": MODELS[n][0], "weights": MODELS[n][1], "role": MODELS[n][2],
                       "sha256": sha256(CHECKPOINTS / MODELS[n][1])} for n in args.models},
        "text_encoder_sha256": None,  # filled in after the models load (first use downloads them)
        "provenance": provenance(args.manifest),
        "images": {},
    }
    for e in entries:
        env["images"].setdefault(e["path"], sha256(REPO / e["path"]))
    (out / "env.json").write_text(json.dumps(env, indent=1))

    # Pre-load every image once (ops applied) so disk and PIL time is outside the timed region.
    loaded = {}
    for e in entries:
        img, gt = load_entry(e)
        loaded[e["id"]] = (img, gt, np.ascontiguousarray(np.asarray(img)[:, :, ::-1]))

    records = []
    for name in args.models:
        kind = MODELS[name][0]
        targets = man["target_classes"]["coco" if kind == "coco" else "open_vocab"]
        t0 = time.perf_counter()
        model = load_model(name, catalog)
        load_s = time.perf_counter() - t0
        first = loaded[entries[0]["id"]][2]
        for _ in range(args.warmup):
            predict(model, first, args)
        print(f"[{name}] loaded in {load_s:.1f} s; classes: {len(model.names)}", flush=True)
        for e in entries:
            img, gt, bgr = loaded[e["id"]]
            res, times = timed_predict(model, bgr, args)
            names = res.names
            dets = []
            for xyxy, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                box = [float(v) for v in xyxy]
                dets.append({"cls": names[int(k)], "conf": round(float(c), 4), "box": round_box(box),
                             "iou_gt": round(iou(box, gt), 3) if gt is not None else None})
            tgt = [x for x in dets if x["cls"] in targets]
            hits = [x for x in tgt if (x["iou_gt"] or 0) >= HIT_IOU]
            fps = [x for x in tgt if (x["iou_gt"] or 0) < FP_IOU]
            on_mug = [x for x in dets if (x["iou_gt"] or 0) >= HIT_IOU]
            best_hit = max(hits, key=lambda x: x["conf"]) if hits else None
            best_on = max(on_mug, key=lambda x: x["conf"]) if on_mug else None
            rec = {
                "image_id": e["id"], "group": e["group"], "model": name, "source": e["path"], "ops": e.get("ops", []),
                "size": list(img.size), "gt_box": round_box(gt),
                "hit_conf": best_hit["conf"] if best_hit else 0.0,
                "hit_cls": best_hit["cls"] if best_hit else None,
                "hit_iou": best_hit["iou_gt"] if best_hit else None,
                "target_any_conf": max((x["conf"] for x in tgt), default=0.0),
                "target_fp_conf": max((x["conf"] for x in fps), default=0.0),
                "top_on_mug": f"{best_on['cls']} {best_on['conf']:.2f}" if best_on else None,
                "any_on_mug_conf": best_on["conf"] if best_on else 0.0,
                "n_dets_ge_025": sum(x["conf"] >= 0.25 for x in dets),
                "wall_ms_median": round(statistics.median(times), 2),
                "wall_ms": [round(t, 2) for t in times],
                "ultralytics_speed_ms": {k: round(v, 2) for k, v in res.speed.items()},
                "dets": dets,
            }
            records.append(rec)
            if not args.no_overlays:
                title = f"{name} | {e['id']} | hit {rec['hit_conf']:.2f}"
                draw_overlay(img, dets, gt, targets, title, out / "overlays" / name / f"{e['id']}.jpg")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # text encoders behind the open-vocabulary prompts (set_classes); their weights change the scores
    env["text_encoder_sha256"] = {p: sha256(CHECKPOINTS / p) for p in ("clip/ViT-B-32.pt", "mobileclip2_b.ts")
                                  if (CHECKPOINTS / p).exists()}
    (out / "env.json").write_text(json.dumps(env, indent=1))
    with open(out / "detections.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out / "detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group", "model", "cls", "conf", "x0", "y0", "x1", "y1", "iou_gt"])
        for r in records:
            for x in r["dets"]:
                w.writerow([r["image_id"], r["group"], r["model"], x["cls"], x["conf"], *x["box"], x["iou_gt"]])
    cols = ["image_id", "group", "model", "hit_conf", "hit_cls", "hit_iou", "target_any_conf", "target_fp_conf",
            "top_on_mug", "any_on_mug_conf", "n_dets_ge_025", "wall_ms_median"]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    # Console table: image x model hit confidence
    ids = [e["id"] for e in entries]
    by = {(r["image_id"], r["model"]): r for r in records}
    print("\nhit_conf (target class, IoU>=0.5 with hand box); [fp=max target-class conf elsewhere]")
    print(f"{'image':34s}" + "".join(f"{m:>22s}" for m in args.models))
    for i in ids:
        row = ""
        for m in args.models:
            r = by[(i, m)]
            fp = f" fp{r['target_fp_conf']:.2f}" if r["target_fp_conf"] >= 0.10 else ""
            row += f"{r['hit_conf']:>14.2f}{fp:>8s}"
        print(f"{i:34s}{row}")
    print("\nwarm latency per image, median over images of per-image medians (ms):")
    for m in args.models:
        v = [by[(i, m)]["wall_ms_median"] for i in ids]
        print(f"  {m:18s} median {statistics.median(v):6.2f}  min {min(v):6.2f}  max {max(v):6.2f}  (n={len(v)} images x {args.repeats} runs)")


if __name__ == "__main__":
    main()
