"""Compare frozen evidence with the observation wrapper on at most two prepared inputs."""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from development.stress.alignment_diagnostics import check_frozen, observe_fit
from development.stress.prepare_inputs import PROTOCOL, STUDY, load, sha
from development.stress.run_evidence import digest_tree


def compare(left, right):
    """Return numeric maximum difference and nonnumeric mismatches, including missing keys."""
    differences, mismatches = [], []
    def visit(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    mismatches.append(path + '.' + key)
                else:
                    visit(a[key], b[key], path + '.' + key)
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                mismatches.append(path + ':length')
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, f'{path}[{i}]')
        elif isinstance(a, bool) or isinstance(b, bool):
            if type(a) is not type(b) or a != b: mismatches.append(path)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isfinite(a) or not math.isfinite(b):
                mismatches.append(path + ':nonfinite')
            else:
                differences.append(abs(a - b))
        elif a != b:
            mismatches.append(path)
    visit(left, right, '$')
    return {'max_abs_difference': max(differences, default=0.0), 'nonnumeric_mismatches': mismatches}


def run(prepared, output, analysis, ids, device):
    if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids):
        raise ValueError('select one or two unique input IDs')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from huggingface_hub import snapshot_download
    from perception import evidence as ev, evidence_v03 as v03
    from perception.common import CHECKPOINTS
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    proto, pre = load(PROTOCOL), load(prepared / 'preparation.json')
    if pre['study_id'] != STUDY or pre['protocol_sha256'] != sha(PROTOCOL):
        raise ValueError('study or protocol mismatch')
    frozen = check_frozen()
    if frozen != pre['frozen_hashes']:
        raise ValueError('frozen sources mismatch')
    spec = next(x for x in proto['analyses'] if x['id'] == analysis)
    size = 'primary' if spec['size'] == proto['primary_size'] else 'native'
    folder = prepared / 'verifier' / size
    jobs_path = folder / 'jobs.json'
    if pre['jobs_sha256'][size] != sha(jobs_path):
        raise ValueError('prepared jobs mismatch')
    jobs = load(jobs_path)
    by_id = {j['id']: j for j in jobs}
    if len(by_id) != len(jobs) or any(ident not in by_id for ident in ids):
        raise ValueError('missing or duplicate input IDs')
    for ident in ids:
        for key in ('reference', 'query'):
            job = by_id[ident]
            path = folder / job[key]
            if path.parent != folder or sha(path) != job[key + '_sha256']:
                raise ValueError('input image hash mismatch')
    p, v = ev.Params(), v03.V03Params()
    ap = v03.AlignParams(**{**proto['align_params'], 'max_shift_px': spec['max_shift_px']})
    snapshot = snapshot_download(ev.DINOV2_REPO, revision=ev.DINOV2_REVISION, local_files_only=True)
    weights = {'dinov2': digest_tree(snapshot), 'yolo': digest_tree(CHECKPOINTS / ev.DETECTOR_WEIGHTS),
               'clip': digest_tree(CHECKPOINTS / 'clip/ViT-B-32.pt')}
    encoder, detector = ev.DinoV2Patches(device), ev.YoloWorld(device, p=p)
    records = []
    for ident in ids:
        job = by_id[ident]
        ref, query = ev.read_bgr(folder / job['reference']), ev.read_bgr(folder / job['query'])
        if list(ref.shape[:2][::-1]) != spec['size'] or list(query.shape[:2][::-1]) != spec['size']:
            raise ValueError('image dimensions mismatch')
        baseline, original = v03.evidence_for(ref, job['box'], query, 0, encoder, detector, p, v, ap=ap)
        observed = {}
        def align(r, q):
            accepted, info, raw = observe_fit(v03, r, q, tuple(job['box']), ap)
            observed['registration'] = info
            observed['raw_H'] = None if raw is None else raw.tolist()
            return accepted
        wrapped, diagnostic = v03.evidence_for(ref, job['box'], query, 0, encoder, detector, p, v, align=align, ap=ap)
        diagnostic['registration'] = observed['registration']
        records.append({'id': ident, 'evidence_comparison': compare(dataclasses.asdict(baseline), dataclasses.asdict(wrapped)),
                        'alignment_info_comparison': compare(original['registration'], observed['registration']),
                        'alignment_ok_equal': original['align_ok'] == diagnostic['align_ok'],
                        'baseline_evidence': dataclasses.asdict(baseline), 'wrapped_evidence': dataclasses.asdict(wrapped),
                        'baseline_registration': original['registration'], 'wrapped_registration': observed['registration'],
                        'raw_H_query_to_reference': observed['raw_H']})
    result = {'study_id': STUDY, 'analysis': analysis, 'protocol_sha256': sha(PROTOCOL),
              'preparation_sha256': sha(prepared / 'preparation.json'), 'jobs_sha256': sha(jobs_path),
              'parity_script_sha256': sha(__file__), 'frozen_hashes': frozen, 'params': dataclasses.asdict(p),
              'v03_params': dataclasses.asdict(v), 'align_params': dataclasses.asdict(ap),
              'device': device, 'python': platform.python_version(), 'checkpoints': weights, 'records': records}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--analysis', required=True, choices=['primary', 'native', 'native_normalized_guard', 'relaxed_guard'])
    parser.add_argument('--id', action='append', required=True, dest='ids')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args.prepared, args.out, args.analysis, args.ids, args.device)
