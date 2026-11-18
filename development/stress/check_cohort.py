"""Validate every full-cohort render against its declared scene and masks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from development.stress.alignment_diagnostics import check_frozen
from development.stress.prepare_inputs import PROTOCOL, load, validate_render
from sim import arrange as A
from sim import isaac_batch as B


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(exist_ok=False)
    frozen, protocol = check_frozen(), load(PROTOCOL)
    manifest = load(root / 'geometry-v3/cohort_manifest.json')
    records = manifest['records']
    audit = load(root / 'reviewer-geometry-audit.json')
    if len(records) != 120 or audit['result'] != 'PASS' or audit['records'] != 120:
        raise ValueError('wrong cohort or geometry audit')
    names = {p.name for p in (root / 'renders').iterdir() if p.is_dir()}
    ids = {r['arrangement_id'] for r in records}
    if names != ids or len(ids) != 120:
        raise ValueError('missing, extra, or partial render folders')
    results, base = {}, {}
    for i, rec in enumerate(records, 1):
        ident = rec['arrangement_id']
        source = root / 'geometry-v3' / rec['arrangement_file']
        if sha(source) != audit['arrangement_file_sha256'][ident]:
            raise ValueError(f'geometry audit hash mismatch: {ident}')
        arr, lab, folder = validate_render(rec, root / 'geometry-v3', protocol, frozen)
        expected_state = {'reference': (True, 'parked', 'nominal'), 'visible': (True, 'parked', 'revisit'),
                          'occluded': (True, 'placed', 'revisit'), 'removed': (False, 'placed', 'revisit')}
        state = {x['phase']: (x['target_present'], x['occluder_pose'], x['camera']) for x in arr['phases']}
        if state != expected_state:
            raise ValueError(f'phase state mismatch: {ident}')
        fingerprint = A.sha256_json({'objects': arr['objects'], 'cameras': arr['scene']['cameras']})
        old = base.setdefault(arr['base_id'], fingerprint)
        if old != fingerprint:
            raise ValueError(f'base scene differs across levels: {ident}')
        stops = lab['stops']
        files = sorted((folder / 'masks').glob('*.png'))
        if len(files) != 45:
            raise ValueError(f'mask count mismatch: {ident}')
        masks = {}
        for path in files:
            with Image.open(path) as image:
                if list(image.size) != protocol['render_size']:
                    raise ValueError(f'mask dimensions: {path}')
                masks[path.stem] = np.asarray(image).copy()
        target, occluder = {}, {}
        for phase in expected_state:
            for stop in stops:
                classes = masks[f'classes_{phase}_{stop}']
                allowed = {0, 1, 2} | set(range(10, 10 + len(arr['objects']['distractors'])))
                if not set(np.unique(classes)) <= allowed:
                    raise ValueError(f'unexpected class: {ident} {phase} {stop}')
                if phase != 'removed':
                    target[phase, stop] = classes == 1
                    if not np.array_equal(masks[f'target_{phase}_{stop}'] != 0, target[phase, stop]):
                        raise ValueError(f'target mask differs: {ident} {phase} {stop}')
                elif np.count_nonzero(classes == 1) != lab['removed_target_px'][stop] or lab['removed_target_px'][stop] != 0:
                    raise ValueError(f'removed target pixels: {ident} {stop}')
                if phase in ('occluded', 'removed'):
                    occluder[phase, stop] = classes == 2
        alone = {s: masks[f'target_alone_{s}'] != 0 for s in stops}
        revisit = {s: masks[f'target_alone_revisit_{s}'] != 0 for s in stops}
        computed = B.compute_labels(stops, target, alone, revisit, occluder)
        for key, value in computed.items():
            if lab[key] != value:
                raise ValueError(f'mask-derived label differs: {ident} {key}')
        for stop in stops:
            if lab['reference_box']['L1'][stop] != B.mask_box(target['reference', stop]):
                raise ValueError(f'reference box differs: {ident} {stop}')
        results[ident] = {'rgb_count': 20, 'mask_count': len(files), 'label_sanity': lab['label_sanity']['ok'],
                          'source_sha256': sha(source), 'labels_sha256': sha(folder / 'labels.json'),
                          'render_sha256': sha(folder / 'render.json'),
                          'rgb_sha256': {p.name: sha(p) for p in sorted((folder / 'rgb').glob('*.png'))},
                          'mask_sha256': {p.name: sha(p) for p in files}}
        print(f'{i}/120 {ident} passed', flush=True)
    result = {'status': 'PASS', 'arrangements': len(results), 'bases': len(base),
              'rgb_count': sum(x['rgb_count'] for x in results.values()),
              'mask_count': sum(x['mask_count'] for x in results.values()),
              'frozen_count': len(frozen), 'validator_sha256': sha(__file__), 'records': results}
    with (out / 'cohort-validation.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    run(a.root, a.out)
