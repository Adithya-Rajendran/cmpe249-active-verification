"""Frozen two-image VGGT and zero-shot Qwen registered-place adaptation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['YOLO_AUTOINSTALL'] = 'false'

import cv2
import numpy as np
from PIL import Image, ImageDraw
from development.cross_sim.geoscd_geometry import GeometryProvider

SPEC_SHA = 'db57d40b5d55ba40b0b3fcdcb7a449c322bb861d4cb401c29361f6c5f5f045ed'
MODEL_REV = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
WEIGHT_SHA = 'f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_cases(path):
    root = path.parent.resolve()
    cases = json.loads(path.read_text())
    ids = set()
    for row in cases:
        if set(row) != {'id', 'reference', 'query', 'box_xyxy', 'reference_sha256', 'query_sha256'}:
            raise ValueError('unexpected case fields')
        if not isinstance(row['id'], str) or not row['id'].isalnum() or row['id'] in ids:
            raise ValueError('invalid or repeated case id')
        ids.add(row['id'])
        for key in ('reference', 'query'):
            p = (root / row[key]).resolve()
            if p.parent != root or sha(p) != row[key + '_sha256']:
                raise ValueError('image path or digest mismatch')
        box = row['box_xyxy']
        if len(box) != 4 or any(type(x) is not int for x in box):
            raise ValueError('box must have four integer coordinates')
    return cases


def make_images(ref, query, warped, valid, box):
    height, width = ref.shape[:2]
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('box outside reference')
    mark = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(mark).rectangle((x0, y0, x1 - 1, y1 - 1), outline='red', width=3)
    full1 = mark.resize((768, 432), Image.Resampling.LANCZOS)
    full2 = Image.fromarray(cv2.cvtColor(query, cv2.COLOR_BGR2RGB)).resize((768, 432), Image.Resampling.LANCZOS)
    bx = (max(0, int(np.floor(x0 - .25 * (x1-x0)))), max(0, int(np.floor(y0 - .25 * (y1-y0)))),
          min(width, int(np.ceil(x1 + .25 * (x1-x0)))), min(height, int(np.ceil(y1 + .25 * (y1-y0)))))
    clean = warped.copy()
    clean[~valid] = 0
    crops = []
    for arr in (ref, clean):
        image = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).crop(bx)
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        crops.append(image)
    return [full1, full2, *crops], bx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--geo-source', type=Path, required=True)
    p.add_argument('--geo-checkpoint', type=Path, required=True)
    a = p.parse_args()
    if sha(a.spec) != SPEC_SHA or a.model.name != MODEL_REV:
        raise ValueError('specification or model revision mismatch')
    spec = json.loads(a.spec.read_text())
    cases = load_cases(a.cases.resolve())
    if a.out.exists():
        raise FileExistsError('output directory already exists')
    a.out.mkdir(parents=True)
    import torch
    import transformers
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_files = sorted(a.model.glob('model.safetensors-*-of-*.safetensors'))
    if len(model_files) != 4 or any(not f.is_file() for f in model_files):
        raise ValueError('model shards missing')
    weights = {f.name: sha(f) for f in model_files}
    # Hash actual shard bytes; cache blob names need not be SHA-256 values.
    binding = {'spec_sha256': SPEC_SHA, 'cases_sha256': sha(a.cases), 'script_sha256': sha(__file__),
               'model_revision': MODEL_REV, 'model_shards': weights, 'torch': torch.__version__,
               'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
               'geometry_checkpoint_sha256': WEIGHT_SHA}
    (a.out / 'binding.json').write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')
    provider = GeometryProvider({'source_dir': str(a.geo_source), 'checkpoint': str(a.geo_checkpoint),
                                 'checkpoint_sha256': WEIGHT_SHA, 'device': 'cuda', 'precision': 'bfloat16'})
    processor = AutoProcessor.from_pretrained(str(a.model), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(str(a.model), local_files_only=True,
                                                         dtype=torch.bfloat16).to('cuda').eval()
    rows = a.out / 'records.jsonl'
    with rows.open('x', buffering=1) as stream:
        for case in cases:
            start = time.monotonic()
            result = {'id': case['id'], 'reference_sha256': case['reference_sha256'],
                      'query_sha256': case['query_sha256'], 'box_xyxy': case['box_xyxy'],
                      'answer': 'UNKNOWN', 'raw': None, 'failure': None}
            try:
                ref = cv2.imread(str(a.cases.parent / case['reference']), cv2.IMREAD_COLOR)
                query = cv2.imread(str(a.cases.parent / case['query']), cv2.IMREAD_COLOR)
                if ref is None or query is None or ref.shape != (675, 1200, 3) or query.shape != ref.shape:
                    raise ValueError('expected 1200x675 RGB images')
                box = case['box_xyxy']
                warped, valid, diag = provider.register(ref, query, box)
                x0, y0, x1, y1 = box
                fraction = float(valid[y0:y1, x0:x1].mean())
                result['valid_fraction'] = fraction
                if diag['registration_ok'] and fraction >= .60:
                    images, bounds = make_images(ref, query, warped, valid, box)
                    result['crop_bounds_xyxy'] = bounds
                    messages = [{'role': 'user', 'content': [*({'type': 'image', 'image': image} for image in images),
                                                              {'type': 'text', 'text': spec['prompt']}]}]
                    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                           return_dict=True, return_tensors='pt', enable_thinking=False).to('cuda')
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=False, max_new_tokens=16)
                    raw = processor.decode(generated[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                    result['raw'] = raw
                    if raw in ('PRESENT', 'REMOVED', 'UNKNOWN'):
                        result['answer'] = raw
                    else:
                        result['failure'] = 'invalid_model_output'
                else:
                    result['failure'] = 'registration_gate'
            except Exception as exc:
                result['failure'] = type(exc).__name__ + ': ' + str(exc)
            result['runtime_seconds'] = time.monotonic() - start
            stream.write(json.dumps(result, allow_nan=False, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
