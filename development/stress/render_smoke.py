"""Validate and launch only the four approved development smoke arrangements."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

from sim import arrange as A
from sim import isaac_batch as B
from sim import isaac_rig as R

STUDY = 'support-stress-dev-v1'
FREEZE_SHA256 = 'bb7baa02e721d6a8759d6e19d2e8e4c0e0c719536312e8e77eae18d323a9196b'
IDS = ('SXS01D0', 'SXS01D6', 'SXT01D0', 'SXT01D6')
ROOT = Path(__file__).resolve().parents[2]


def digest_tree(root):
    root = Path(root).resolve(strict=True)
    return {str(p.relative_to(root)): A.sha256_file(p) for p in sorted(root.rglob('*')) if p.is_file()}


def preflight(geometry, output, assets, tags, textures):
    geometry, output = Path(geometry).resolve(), Path(output).resolve()
    protocol = json.loads((ROOT / 'development/stress/protocol.json').read_text())
    if protocol.get('study_id') != STUDY or protocol.get('status') != 'development':
        raise ValueError('invalid study ID or development status')
    summary = json.loads((geometry / 'summary.json').read_text())
    if set(summary) != {'shelf', 'tabletop'} or not all(x['passes'] for x in summary.values()):
        raise ValueError('geometry quota has not passed')
    provenance = json.loads((geometry / 'provenance.json').read_text())
    if provenance['protocol_sha256'] != A.sha256_file(ROOT / 'development/stress/protocol.json'):
        raise ValueError('protocol provenance mismatch')
    if provenance['builder_sha256'] != A.sha256_file(ROOT / 'development/stress/build_cohort.py'):
        raise ValueError('builder provenance mismatch')
    if output != geometry.parent / 'renders':
        raise ValueError('render output must be the isolated sibling renders directory')
    records = json.loads((geometry / 'cohort_manifest.json').read_text())['records']
    by_id = {x['arrangement_id']: x for x in records}
    if not all(x in by_id for x in IDS):
        raise ValueError('smoke arrangement missing')
    paths, items = [], []
    for ident in IDS:
        record = by_id[ident]
        path = geometry / record['arrangement_file']
        arr = json.loads(path.read_bytes())
        A.check_dev_seed(arr['seed'])
        env = 'shelf' if ident.startswith('SXS') else 'tabletop'
        lo = protocol['seed_starts'][env]
        if not lo <= arr['seed'] < lo + protocol['seeds_per_slot']:
            raise ValueError('smoke seed is outside its first slot')
        if arr.get('study_id') != STUDY or arr.get('split') != 'dev' or arr['arrangement_id'] != ident:
            raise ValueError('invalid arrangement identity')
        if arr['arrangement_sha256'] != A.arrangement_hash(arr):
            raise ValueError('arrangement hash mismatch')
        if arr['stress_provenance'] != provenance:
            raise ValueError('arrangement source provenance mismatch')
        source = arr['provenance']
        for key, source_path in (('generator', 'sim/arrange.py'), ('rig_py', 'sim/rig.py')):
            if source[key]['sha256'] != A.sha256_file(ROOT / source_path):
                raise ValueError(f'{key} source provenance mismatch')
        if env == 'tabletop' and source['tabletop_py']['sha256'] != A.sha256_file(ROOT / 'sim/tabletop.py'):
            raise ValueError('tabletop source provenance mismatch')
        if arr['environment'] != env or arr['level'] != ident[-2:]:
            raise ValueError('unexpected environment or level')
        paths.append(path)
        items.append((arr, path.read_bytes()))
    freeze = ROOT / 'docs/FREEZE.md'
    if A.sha256_file(freeze) != FREEZE_SHA256:
        raise ValueError('freeze record mismatch')
    match = re.search(r'^```json\s*\n(.*?)^```', freeze.read_text(), re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError('freeze hash manifest missing')
    frozen = json.loads(match.group(1))['files']
    required = ('sim/arrange.py', 'sim/rig.py', 'sim/tabletop.py', 'sim/isaac_batch.py',
                'sim/isaac_rig.py', 'configs/sim/arrange_shelf_v3.json',
                'configs/sim/arrange_tabletop_v1.json', 'configs/rig_geometry.json',
                'configs/sim/tabletop_geometry.json', 'configs/sim/ycbv_hulls.json')
    for path in required:
        if A.sha256_file(ROOT / path) != frozen[path]:
            raise ValueError(f'frozen source mismatch: {path}')
    B.check_render_inputs(items, ['L1'], textures)
    inputs = {'study_id': STUDY, 'arrangements': {p.name: A.sha256_file(p) for p in paths},
              'freeze_sha256': FREEZE_SHA256, 'frozen_inputs': {path: frozen[path] for path in required},
              'source': {str(p.relative_to(ROOT)): A.sha256_file(p) for p in
                         (ROOT / 'sim/isaac_batch.py', ROOT / 'sim/isaac_rig.py', Path(__file__).resolve(),
                          ROOT / 'development/stress/protocol.json')},
              'assets': digest_tree(assets), 'textures': digest_tree(textures),
              'isaacsim': B.isaac_version(), 'python': sys.version, 'lighting': ['L1'],
              'renderer': 'RaytracedLighting', 'rt_subframes': 16,
              'settle': R.SETTLE_UPDATES, 'capture_updates': B.CAPTURE_UPDATES,
              'assets_dir': str(Path(assets).resolve()), 'textures_dir': str(Path(textures).resolve()),
              'tags_dir': str(Path(tags).resolve()), 'output_dir': str(output)}
    return paths, items, inputs


def main():
    ap = argparse.ArgumentParser()
    for key in ('geometry', 'output', 'assets', 'tags', 'textures'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--capsule-out')
    a = ap.parse_args()
    paths, items, inputs = preflight(a.geometry, a.output, a.assets, a.tags, a.textures)
    if a.execute and (sys.version_info[:2] != (3, 12) or B.isaac_version() != '6.2.0.0'):
        raise ValueError('execution requires Isaac Sim 6.2.0.0 and Python 3.12')
    if a.capsule_out:
        with Path(a.capsule_out).open('x') as f:
            json.dump(inputs, f, indent=2)
            f.write('\n')
    output = Path(a.output).resolve()
    capsule = output / 'smoke_inputs.json'
    if output.exists() and not capsule.is_file():
        raise ValueError('existing output has no input capsule; refusing overwrite')
    if capsule.exists() and json.loads(capsule.read_text()) != inputs:
        raise ValueError('resume input provenance mismatch')
    pending = []
    for (arr, data), path in zip(items, paths):
        dest = output / arr['arrangement_id']
        if (output / (arr['arrangement_id'] + '.partial')).exists():
            raise ValueError(f'incomplete output at {dest}; refusing overwrite')
        if dest.exists():
            settings = B.render_settings(arr, renderer='RaytracedLighting', rt_subframes=16, settle=R.SETTLE_UPDATES,
                                         capture_updates=B.CAPTURE_UPDATES, lightings=['L1'], assets_dir=a.assets,
                                         tags_dir=a.tags, isaacsim=B.isaac_version(), textures_dir=a.textures)
            expect = {'isaac_batch_sha256': A.sha256_file(ROOT / 'sim/isaac_batch.py'),
                      'isaac_rig_sha256': A.sha256_file(ROOT / 'sim/isaac_rig.py'),
                      'lightings': ['L1'], 'render_settings_sha256': A.sha256_json(settings)}
            redo, why = B.needs_render(dest, data, expect)
            if redo:
                raise ValueError(f'resume mismatch at {dest}: {why}')
            for stop in arr['stops']:
                names = [f'target_{ph}_{stop}.png' for ph in ('reference', 'visible', 'occluded')]
                names += [f'classes_{ph}_{stop}.png' for ph in ('reference', 'visible', 'occluded', 'removed')]
                names += [f'target_alone_{stop}.png', f'target_alone_revisit_{stop}.png']
                if any(not (dest / 'masks' / name).is_file() for name in names):
                    raise ValueError(f'resume masks incomplete at {dest}')
        else:
            pending.append(path)
    if not a.execute or not pending:
        print(json.dumps({'valid': True, 'pending': [p.name for p in pending], 'would_start_gpu': bool(pending and a.execute)}))
        return
    output.mkdir(parents=True, exist_ok=True)
    if not capsule.exists():
        with capsule.open('x') as f:
            json.dump(inputs, f, indent=2)
            f.write('\n')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    cmd = [sys.executable, '-m', 'sim.isaac_batch', '--arrangements', *map(str, pending),
           '--assets-dir', a.assets, '--tags-dir', a.tags, '--textures-dir', a.textures,
           '--out', str(output), '--lighting', 'L1', '--skip-existing']
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1')
    started = time.monotonic()
    with (output / f'launch-{stamp}.log').open('x') as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    with (output / f'launch-{stamp}.json').open('x') as f:
        json.dump({'command': cmd, 'start_utc': stamp, 'exit_code': code,
                   'elapsed_s': time.monotonic() - started}, f, indent=2)
        f.write('\n')
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
