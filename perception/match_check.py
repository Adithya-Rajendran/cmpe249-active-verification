"""Reference-crop matching sanity check (one object, one scene; NOT an evaluation).

Takes the target crop from one reference frame, embeds it with frozen image encoders and reports
cosine similarity against:
  (i)   crops of the same mug from other real frames (pilot queries, later captures, other
        lighting, defocus, the earlier probe pose), at the hand-annotated box;
  (ii)  same-size background crops from the same frames (at most 10 % of the mug inside);
  (iii) augmented copies of the reference crop: digital brightness gain, Gaussian blur, and a
        pasted rectangle covering 25/50/75 % of the crop (flat dark, or a real background patch);
  (iv)  every YOLO-World proposal (any class, conf >= 0.01) in each real frame, to see whether the
        most similar proposal is the mug (IoU >= 0.5 with the hand box). The margin is the best
        mug-proposal similarity minus the best similarity among proposals with IoU < 0.1.

There is exactly one physical target and no second cup, so nothing here tests whether the encoder
can tell this mug from a similar one. Positives from the same camera pose are near-duplicates.

Encoders: DINOv2-small (transformers, CLS/pooler output) and CLIP ViT-B/32 (open_clip,
laion2b_s34b_b79k image tower). Crops are padded to a square with grey, resized to 224 px and
normalised with each model's mean/std.

Usage (from the repo root, after smoke_test.py so the proposals exist):
  uv run python perception/match_check.py --manifest perception/manifests/2026-09-26_smoke.json \
      --detections runs/2026-09-26-perception-smoke/detect/detections.jsonl \
      --out runs/2026-09-26-perception-smoke/match
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, apply_ops, iou, load_entry, load_manifest, provenance, sha256  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REAL_GROUPS = ["pilot", "now", "sweep_best", "sweep_series", "af_bright", "probe"]
PAD = (114, 114, 114)
# Hugging Face revisions of the encoder weights, pinned so that an upstream update cannot change the
# numbers silently (these are the revisions the 2026-09-26 run resolved to).
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"   # facebook/dinov2-small
CLIP_HF_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"         # open_clip ViT-B-32 laion2b_s34b_b79k
CLIP_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
CLIP_FILE = "open_clip_model.safetensors"


# ----------------------------------------------------------------------------- encoders


class Encoder:
    name: str

    def __init__(self, device):
        self.device = device

    def prep(self, crops):
        out = []
        for c in crops:
            w, h = c.size
            s = max(w, h)
            sq = Image.new("RGB", (s, s), PAD)
            sq.paste(c, ((s - w) // 2, (s - h) // 2))
            sq = sq.resize((224, 224), Image.Resampling.BICUBIC)
            a = (np.asarray(sq, dtype=np.float32) / 255.0 - self.mean) / self.std
            out.append(torch.from_numpy(a.transpose(2, 0, 1)))
        return torch.stack(out).to(self.device)

    @torch.no_grad()
    def embed(self, crops, bs=64):
        feats = []
        for i in range(0, len(crops), bs):
            feats.append(torch.nn.functional.normalize(self.forward(self.prep(crops[i:i + bs])).float(), dim=-1))
        return torch.cat(feats)


class DinoV2(Encoder):
    def __init__(self, device, repo="facebook/dinov2-small", revision=DINOV2_REVISION):
        super().__init__(device)
        from transformers import AutoImageProcessor, AutoModel

        self.name = "dinov2-small"
        self.model = AutoModel.from_pretrained(repo, revision=revision).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(repo, revision=revision)
        self.mean = np.array(proc.image_mean, dtype=np.float32)
        self.std = np.array(proc.image_std, dtype=np.float32)
        self.source = {"hf_repo": repo, "revision_requested": revision,
                       "revision": getattr(self.model.config, "_commit_hash", None),
                       "feature": "pooler_output (layer-normed CLS token), 384-d"}

    def forward(self, x):
        return self.model(pixel_values=x).pooler_output


class ClipB32(Encoder):
    def __init__(self, device, arch="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        super().__init__(device)
        import open_clip
        from huggingface_hub import hf_hub_download

        self.name = "clip-vit-b32"
        # open_clip resolves a pretrained tag to the latest HF revision; download the pinned revision
        # of the same file instead and load it with the tag's preprocessing constants.
        cfg = open_clip.pretrained.get_pretrained_cfg(arch, pretrained)
        path = hf_hub_download(CLIP_HF_REPO, CLIP_FILE, revision=CLIP_REVISION)
        self.model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=path, image_mean=cfg["mean"],
                                                                   image_std=cfg["std"])
        self.model = self.model.to(device).eval()
        norm = [t for t in pre.transforms if t.__class__.__name__ == "Normalize"][0]
        self.mean = np.array(norm.mean, dtype=np.float32)
        self.std = np.array(norm.std, dtype=np.float32)
        self.source = {"open_clip": open_clip.__version__, "arch": arch, "pretrained": pretrained,
                       "hf_repo": CLIP_HF_REPO, "revision": CLIP_REVISION, "file": CLIP_FILE,
                       "file_sha256": sha256(path), "feature": "image tower embedding, 512-d"}

    def forward(self, x):
        return self.model.encode_image(x)


# ----------------------------------------------------------------------------- crops


def crop(img, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return img.crop((max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)))


def background_boxes(img, gt, n, rng, max_frac=0.10, tries=5000):
    """Same-size boxes fully inside the frame with at most max_frac of the gt area inside them."""
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    area = w * h
    out = []
    for _ in range(tries):
        if len(out) == n:
            break
        if img.width - w < 1 or img.height - h < 1:
            break
        x0, y0 = rng.uniform(0, img.width - w), rng.uniform(0, img.height - h)
        b = [x0, y0, x0 + w, y0 + h]
        ix = max(0, min(b[2], gt[2]) - max(b[0], gt[0])) * max(0, min(b[3], gt[3]) - max(b[1], gt[1]))
        if ix / area <= max_frac and all(iou(b, o) < 0.5 for o in out):
            out.append(b)
    return out


def paste_rect(c, frac, style, patch=None):
    """Cover `frac` of the crop area. band = full-height band from the left; centre = centred
    rectangle with the crop's aspect ratio. Fill is flat dark grey or a resized background patch."""
    c = c.copy()
    w, h = c.size
    if style.startswith("band"):
        rect = (0, 0, round(w * frac), h)
    else:
        s = frac ** 0.5
        rw, rh = round(w * s), round(h * s)
        rect = ((w - rw) // 2, (h - rh) // 2, (w - rw) // 2 + rw, (h - rh) // 2 + rh)
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    if style.endswith("flat"):
        fill = Image.new("RGB", (rw, rh), (45, 45, 50))
    else:
        fill = patch.resize((rw, rh), Image.Resampling.BICUBIC)
    c.paste(fill, rect[:2])
    return c


def augmentations(ref_crop, patch):
    out = []
    for g in (0.5, 0.7, 1.4, 2.0):
        out.append((f"gain_x{g}", "brightness", g, ref_crop.point(lambda v, g=g: max(0, min(255, int(round(v * g)))))))
    from PIL import ImageFilter

    for s in (2, 4, 8, 16):
        out.append((f"blur_s{s}", "blur", s, ref_crop.filter(ImageFilter.GaussianBlur(s))))
    for style in ("band_flat", "band_texture", "centre_texture"):
        for f in (0.25, 0.5, 0.75):
            out.append((f"occl_{style}_{int(f * 100)}", f"occlusion_{style}", f, paste_rect(ref_crop, f, style, patch)))
    return out


# ----------------------------------------------------------------------------- stats


def auroc(pos, neg):
    """P(sim(pos) > sim(neg)) with ties counted half (Mann-Whitney U / (n_pos n_neg))."""
    if not pos or not neg:
        return None
    u = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return u / (len(pos) * len(neg))


def describe(v):
    if not v:
        return None
    return {"n": len(v), "min": round(min(v), 4), "median": round(statistics.median(v), 4), "max": round(max(v), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--detections", required=True, help="detections.jsonl from smoke_test.py (YOLO-World proposals)")
    ap.add_argument("--proposal-model", default="yolov8s-worldv2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-per-frame", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    man = load_manifest(args.manifest)
    rng = random.Random(args.seed)

    # Reference crop and a background patch (for textured occluders) from the reference frame.
    mref = man["match_reference"]
    ref_img, ref_box = apply_ops(Image.open(REPO / mref["path"]).convert("RGB"), mref.get("ops", []), mref["box"])
    ref_crop = crop(ref_img, ref_box)
    patch_box = [80, 300, 80 + (ref_box[2] - ref_box[0]), 300 + (ref_box[3] - ref_box[1])]  # dark monitor/puck/desk region, left of the mug
    patch = crop(ref_img, patch_box)
    ref_crop.save(out / "examples" / "reference_crop.jpg", quality=90)
    patch.save(out / "examples" / "occluder_patch.jpg", quality=90)

    # Real frames: positives at the hand box, same-size negatives, and detector proposals.
    dets = {}
    for line in open(args.detections):
        r = json.loads(line)
        if r["model"] == args.proposal_model:
            dets[r["image_id"]] = r
    items = []  # (set, group, frame_id, variant, crop, extra)
    # "*_crop" entries are skipped: their mug crop is pixel-identical to the full-frame entry's.
    frames = [e for e in man["images"] if e["group"] in REAL_GROUPS and e["path"] != mref["path"]
              and not e["id"].endswith("_crop")]
    for e in frames:
        img, gt = load_entry(e)
        items.append(("positive", e["group"], e["id"], "hand_box", crop(img, gt), {}))
        for k, b in enumerate(background_boxes(img, gt, args.neg_per_frame, rng)):
            items.append(("negative", e["group"], e["id"], f"bg{k:02d}", crop(img, b), {"box": [round(v) for v in b]}))
        r = dets.get(e["id"])
        if r:
            for k, d in enumerate(r["dets"]):
                items.append(("proposal", e["group"], e["id"], f"p{k:03d}", crop(img, d["box"]),
                              {"cls": d["cls"], "conf": d["conf"], "iou_gt": d["iou_gt"]}))
    augs = augmentations(ref_crop, patch)
    for name, kind, level, c in augs:
        items.append(("augment", kind, "reference_t01", name, c, {"level": level}))
    for name, _, _, c in augs:
        if name in ("gain_x0.5", "blur_s8", "occl_band_flat_50", "occl_band_texture_50", "occl_centre_texture_75"):
            c.resize((c.width // 3, c.height // 3)).save(out / "examples" / f"aug_{name}.jpg", quality=85)
    # Anchor sanity row: the reference against itself must give 1.0.
    items.append(("self", "reference", "reference_t01", "identity", ref_crop, {}))

    env = {
        "command": " ".join(sys.argv), "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(), "torch": torch.__version__, "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "reference": {"path": mref["path"], "sha256": sha256(REPO / mref["path"]), "box": ref_box, "crop_size": list(ref_crop.size)},
        "occluder_patch_box": patch_box, "neg_rule": "same size as the frame's hand box, <= 10% of the mug area inside, pairwise IoU < 0.5",
        "seed": args.seed, "n_items": len(items), "encoders": {},
        "detections_sha256": sha256(args.detections), "provenance": provenance(args.manifest),
    }

    rows = []
    summary = {}
    for Enc in (DinoV2, ClipB32):
        t0 = time.perf_counter()
        enc = Enc(args.device)
        load_s = time.perf_counter() - t0
        env["encoders"][enc.name] = enc.source | {"load_s": round(load_s, 2)}
        ref = enc.embed([ref_crop])
        # warm single-crop latency (prep + forward), GPU synchronised
        for _ in range(5):
            enc.embed([ref_crop])
        lat = []
        for _ in range(20):
            torch.cuda.synchronize() if args.device == "cuda" else None
            t = time.perf_counter()
            enc.embed([ref_crop])
            torch.cuda.synchronize() if args.device == "cuda" else None
            lat.append((time.perf_counter() - t) * 1e3)
        feats = enc.embed([it[4] for it in items])
        sims = (feats @ ref.T).squeeze(1).cpu().tolist()
        for it, s in zip(items, sims):
            rows.append({"encoder": enc.name, "set": it[0], "group": it[1], "frame": it[2], "variant": it[3],
                         "sim": round(s, 4), **{k: (json.dumps(v) if k == "box" else v) for k, v in it[5].items()}})

        er = [r for r in rows if r["encoder"] == enc.name]
        S = {"latency_ms_single_crop": {"median": round(statistics.median(lat), 2), "min": round(min(lat), 2), "n": len(lat)}}
        S["self_sim"] = [r["sim"] for r in er if r["set"] == "self"][0]
        pos = {g: [r["sim"] for r in er if r["set"] == "positive" and r["group"] == g] for g in REAL_GROUPS}
        neg = {g: [r["sim"] for r in er if r["set"] == "negative" and r["group"] == g] for g in REAL_GROUPS}
        S["positive"] = {g: describe(v) for g, v in pos.items()}
        S["negative"] = {g: describe(v) for g, v in neg.items()}
        allneg = [s for v in neg.values() for s in v]
        S["negative_all"] = describe(allneg)
        S["per_group"] = {}
        for g in REAL_GROUPS:
            if pos[g] and neg[g]:
                S["per_group"][g] = {"auroc_vs_same_frames_neg": round(auroc(pos[g], neg[g]), 4),
                                     "gap_min_pos_minus_max_neg": round(min(pos[g]) - max(neg[g]), 4),
                                     "n_pos_below_max_neg_all": sum(p <= max(allneg) for p in pos[g])}
        S["augment"] = {r["variant"]: r["sim"] for r in er if r["set"] == "augment"}
        S["augment_vs_max_negative"] = round(max(allneg), 4)
        # (iv) proposals: is the most similar proposal the mug?
        prop = {}
        for r in er:
            if r["set"] == "proposal":
                prop.setdefault(r["frame"], []).append(r)
        top1 = {}
        for f, ps in prop.items():
            g = ps[0]["group"]
            best = max(ps, key=lambda r: r["sim"])
            mug = [r for r in ps if (r["iou_gt"] or 0) >= 0.5]
            # proposals that barely touch the mug; partial boxes on the mug (0.1 <= IoU < 0.5) are
            # neither counted as the mug nor as a distractor
            other = [r for r in ps if (r["iou_gt"] or 0) < 0.1]
            ok = (best["iou_gt"] or 0) >= 0.5
            margin = (max(r["sim"] for r in mug) - max(r["sim"] for r in other)) if mug and other else None
            by_conf = max([r for r in ps if r["cls"] in ("cup", "mug")], key=lambda r: r["conf"], default=None)
            top1.setdefault(g, []).append({"frame": f, "n_props": len(ps), "top1_is_mug": ok, "has_mug_prop": bool(mug),
                                           "top1_overlaps": (best["iou_gt"] or 0) >= 0.1, "top1_iou": best["iou_gt"],
                                           "margin": round(margin, 4) if margin is not None else None,
                                           "top_conf_cupmug_is_mug": bool(by_conf and (by_conf["iou_gt"] or 0) >= 0.5)})
        S["proposals"] = {g: {"frames": len(v), "top1_sim_is_mug": sum(x["top1_is_mug"] for x in v),
                              "frames_with_mug_proposal": sum(x["has_mug_prop"] for x in v),
                              "top_cupmug_conf_is_mug": sum(x["top_conf_cupmug_is_mug"] for x in v),
                              "top1_sim_overlaps_mug_iou_ge_0.1": sum(x["top1_overlaps"] for x in v),
                              "margin_min": min((x["margin"] for x in v if x["margin"] is not None), default=None),
                              "margin_median": statistics.median([x["margin"] for x in v if x["margin"] is not None]) if any(x["margin"] is not None for x in v) else None,
                              "median_props": statistics.median([x["n_props"] for x in v])}
                          for g, v in top1.items()}
        S["proposal_frames"] = top1
        summary[enc.name] = S
        del enc
        torch.cuda.empty_cache() if args.device == "cuda" else None

    (out / "env.json").write_text(json.dumps(env, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = ["encoder", "set", "group", "frame", "variant", "sim", "cls", "conf", "iou_gt", "level", "box"]
    with open(out / "similarities.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    for name, S in summary.items():
        print(f"\n=== {name}  (self-sim {S['self_sim']:.4f}; single-crop warm latency {S['latency_ms_single_crop']['median']} ms)")
        print(f"{'group':14s} {'pos n':>5s} {'pos min':>8s} {'pos med':>8s} {'neg max':>8s} {'neg med':>8s} {'AUROC':>6s} {'gap':>7s}")
        for g in REAL_GROUPS:
            p, n, pg = S["positive"][g], S["negative"][g], S["per_group"].get(g, {})
            if p and n:
                print(f"{g:14s} {p['n']:5d} {p['min']:8.3f} {p['median']:8.3f} {n['max']:8.3f} {n['median']:8.3f} "
                      f"{pg['auroc_vs_same_frames_neg']:6.3f} {pg['gap_min_pos_minus_max_neg']:7.3f}")
        print(f"all negatives: {S['negative_all']}")
        print("augmentations (sim to reference):  " + ", ".join(f"{k} {v:.3f}" for k, v in S["augment"].items()))
        print("proposals (top-1 by similarity is the mug / frames):  " +
              ", ".join(f"{g} {v['top1_sim_is_mug']}/{v['frames']} (min margin {v['margin_min']})" for g, v in S["proposals"].items()))


if __name__ == "__main__":
    main()
