"""Run frozen image evidence on bound, sanitized inputs. No task model is used."""
from __future__ import annotations
import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from development.stress.prepare_inputs import PROTOCOL, load, sha, STUDY
from development.stress.alignment_diagnostics import check_frozen, observe_fit, refusal_reason


def digest_tree(path):
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f'missing checkpoint: {path}')
    files = [path] if path.is_file() else sorted(x for x in path.rglob('*') if x.is_file())
    if not files:
        raise ValueError(f'empty checkpoint: {path}')
    return {str(f.relative_to(path) if path.is_dir() else f.name): sha(f) for f in files}


def run(prepared, output, analysis, device):
    # Offline mode makes missing checkpoint files an error instead of downloading them.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import cv2
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from perception import evidence as ev, evidence_v03 as v03
    from perception.common import CHECKPOINTS

    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    proto = load(PROTOCOL)
    spec = next(x for x in proto['analyses'] if x['id'] == analysis)
    pre = load(prepared / 'preparation.json')
    if pre['study_id'] != STUDY or pre['protocol_sha256'] != sha(PROTOCOL):
        raise ValueError('preparation study or protocol mismatch')
    frozen = check_frozen()
    if frozen != pre['frozen_hashes']:
        raise ValueError('frozen source binding changed')
    size = 'primary' if spec['size'] == proto['primary_size'] else 'native'
    folder = prepared / 'verifier' / size
    jobs_path = folder / 'jobs.json'
    if pre['jobs_sha256'][size] != sha(jobs_path):
        raise ValueError('prepared jobs binding mismatch')
    jobs = load(jobs_path)
    if len(jobs) != pre['count'] or len({x['id'] for x in jobs}) != len(jobs):
        raise ValueError('job count or ids mismatch')
    for job in jobs:
        for key in ('reference', 'query'):
            path = folder / job[key]
            if path.parent != folder or sha(path) != job[key + '_sha256']:
                raise ValueError(f'input hash mismatch: {path}')
    params, vparams = ev.Params(), v03.V03Params()
    align_params = v03.AlignParams(**{**proto['align_params'], 'max_shift_px': spec['max_shift_px']})
    snapshot = snapshot_download(ev.DINOV2_REPO, revision=ev.DINOV2_REVISION, local_files_only=True)
    weights = { 'dinov2': digest_tree(snapshot), 'yolo': digest_tree(CHECKPOINTS / ev.DETECTOR_WEIGHTS),
                'clip': digest_tree(CHECKPOINTS / 'clip/ViT-B-32.pt') }
    versions = {}
    for name in ('numpy', 'opencv-python', 'torch', 'transformers', 'ultralytics', 'huggingface-hub', 'Pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    binding = {'study_id': STUDY, 'analysis': analysis, 'protocol_sha256': sha(PROTOCOL),
               'preparation_sha256': sha(prepared / 'preparation.json'), 'jobs_sha256': sha(jobs_path),
               'runner_sha256': sha(__file__), 'alignment_diagnostics_sha256': sha(ROOT / 'development/stress/alignment_diagnostics.py'),
               'frozen_hashes': frozen, 'params': dataclasses.asdict(params), 'v03_params': dataclasses.asdict(vparams),
               'align_params': dataclasses.asdict(align_params), 'thresholds': proto['thresholds'],
               'checkpoints': weights, 'device': device, 'python': platform.python_version(),
               'versions': versions, 'runtime_module_hashes': {str(Path(m.__file__).resolve()): sha(Path(m.__file__).resolve()) for m in (cv2, np, torch, ev, v03) if getattr(m, '__file__', None)}, 'torch_cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(device) if device.startswith('cuda') and torch.cuda.is_available() else None}
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / 'manifest.json'
    if binding_path.exists():
        if load(binding_path) != binding:
            raise ValueError('resume binding mismatch')
    else:
        if any(output.iterdir()):
            raise ValueError('output exists without a run manifest')
        binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')
    result_dir = output / 'records'
    result_dir.mkdir(exist_ok=True)
    expected = {x['id'] for x in jobs}
    existing = {x.stem for x in result_dir.glob('*.json')}
    if not existing <= expected:
        raise ValueError('unexpected result records')
    for ident in existing:
        record = load(result_dir / (ident + '.json'))
        if record.get('id') != ident or record.get('binding_sha256') != sha(binding_path):
            raise ValueError('existing record binding mismatch')
    pending = [j for j in jobs if j['id'] not in existing]
    if not pending:
        return
    initializing = time.perf_counter()
    encoder, detector = ev.DinoV2Patches(device), ev.YoloWorld(device, p=params)
    with (output / 'initialization.jsonl').open('a') as f:
        f.write(json.dumps({'initialization_seconds': time.perf_counter() - initializing, 'pending_inputs': len(pending)}) + '\n')
    for job in pending:
        started = time.perf_counter()
        # Errors stop the run, leaving a visibly missing record for a safe resume.
        ref, query = ev.read_bgr(folder / job['reference']), ev.read_bgr(folder / job['query'])
        if list(ref.shape[:2][::-1]) != spec['size'] or list(query.shape[:2][::-1]) != spec['size']:
            raise ValueError('input dimensions changed')
        info = {}
        def align(r, q):
            accepted, registration, raw = observe_fit(v03, r, q, tuple(job['box']), align_params)
            info.update(registration=registration, raw_H=None if raw is None else raw.tolist(),
                        refusal=refusal_reason(registration))
            return accepted
        rec, diag = v03.evidence_for(ref, job['box'], query, 0, encoder, detector, params, vparams, align=align, ap=align_params)
        diag['registration'] = info['registration']
        record = {'id': job['id'], 'binding_sha256': sha(binding_path), 'evidence': dataclasses.asdict(rec),
                  'diagnostics': diag, 'raw_H_query_to_reference': info['raw_H'], 'refusal': info['refusal'],
                  'runtime_seconds': time.perf_counter() - started, 'failure': False}
        temp = result_dir / (job['id'] + '.tmp')
        temp.write_text(json.dumps(record, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)) + '\n')
        temp.replace(result_dir / (job['id'] + '.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--analysis', required=True, choices=['primary', 'native', 'native_normalized_guard', 'relaxed_guard'])
    parser.add_argument('--device', default='cuda')
    a = parser.parse_args()
    run(a.prepared, a.out, a.analysis, a.device)
