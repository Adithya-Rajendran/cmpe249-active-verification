"""Preflight and, when authorized, render the reviewed development cohort."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from development.stress import render_smoke as S
from sim import arrange as A
from sim import isaac_batch as B
from sim import isaac_rig as R

AUDIT_SHA256 = '12409ccd9195324bf9423077195e10d3c6ad67035ee2ac7308adcf0ec3d71250'
SMOKE_CAPSULE_SHA256 = '86ee86125f50ac6a1156ca179ab156598c137ce4fe19244bcac22a8aadeb1f8c'


def preflight(geometry, output, assets, tags, textures, audit_path):
    geometry, output, audit_path = Path(geometry).resolve(), Path(output).resolve(), Path(audit_path).resolve()
    _, _, smoke_inputs = S.preflight(geometry, output, assets, tags, textures)
    smoke_capsule = output / 'smoke_inputs.json'
    if A.sha256_file(smoke_capsule) != SMOKE_CAPSULE_SHA256 or json.loads(smoke_capsule.read_text()) != smoke_inputs:
        raise ValueError('smoke capsule differs from reviewed inputs')
    if A.sha256_file(audit_path) != AUDIT_SHA256:
        raise ValueError('geometry audit differs from reviewed record')
    audit = json.loads(audit_path.read_text())
    if audit['result'] != 'PASS' or audit['records'] != 120 or audit['base_arrangements'] != 24:
        raise ValueError('geometry audit did not pass')
    if audit['protocol_sha256'] != smoke_inputs['source']['development/stress/protocol.json']:
        raise ValueError('geometry audit protocol differs')
    data = json.loads((geometry / 'cohort_manifest.json').read_text())
    records = data['records']
    expected = {f'SX{env}{i:02d}{level}' for env in ('S', 'T') for i in range(1, 13)
                for level in ('D0', 'D1', 'D2', 'D4', 'D6')}
    if data['study_id'] != S.STUDY or len(records) != 120 or {r['arrangement_id'] for r in records} != expected:
        raise ValueError('cohort IDs do not match the reviewed 24 by 5 design')
    if set(audit['arrangement_file_sha256']) != expected:
        raise ValueError('audit arrangement set mismatch')
    paths, items = [], []
    protocol = json.loads((S.ROOT / 'development/stress/protocol.json').read_text())
    for rec in records:
        ident = rec['arrangement_id']
        path = geometry / rec['arrangement_file']
        if path.resolve().parent != geometry or path.name != ident + '.json':
            raise ValueError('unexpected arrangement path')
        if A.sha256_file(path) != audit['arrangement_file_sha256'][ident]:
            raise ValueError(f'arrangement differs from audit: {ident}')
        arr = json.loads(path.read_bytes())
        env = 'shelf' if ident.startswith('SXS') else 'tabletop'
        slot = int(ident[3:5])
        level = ident[5:]
        start = protocol['seed_starts'][env] + (slot - 1) * protocol['seeds_per_slot']
        A.check_dev_seed(arr['seed'])
        if not start <= arr['seed'] < start + protocol['seeds_per_slot']:
            raise ValueError(f'seed outside assigned slot: {ident}')
        if (arr['arrangement_id'] != ident or arr['study_id'] != S.STUDY or arr['split'] != 'dev'
                or arr['environment'] != env or arr['level'] != level
                or rec['base_id'] != arr['base_id'] or rec['seed'] != arr['seed']
                or rec['environment'] != env or rec['level_id'] != level
                or arr['arrangement_sha256'] != A.arrangement_hash(arr)):
            raise ValueError(f'arrangement identity mismatch: {ident}')
        paths.append(path)
        items.append((arr, path.read_bytes()))
    B.check_render_inputs(items, ['L1'], textures)
    for p in output.iterdir():
        if p.is_dir() and (p.name not in expected or p.name.endswith('.partial')):
            raise ValueError(f'unexpected or incomplete render folder: {p}')
    pending = []
    for (arr, data_bytes), path in zip(items, paths):
        dest = output / arr['arrangement_id']
        if not dest.exists():
            if arr['arrangement_id'] in S.IDS:
                raise ValueError('reviewed smoke output is missing')
            pending.append(path)
            continue
        settings = B.render_settings(arr, renderer='RaytracedLighting', rt_subframes=16, settle=R.SETTLE_UPDATES,
                                     capture_updates=B.CAPTURE_UPDATES, lightings=['L1'], assets_dir=assets,
                                     tags_dir=tags, isaacsim=B.isaac_version(), textures_dir=textures)
        expect = {'isaac_batch_sha256': A.sha256_file(S.ROOT / 'sim/isaac_batch.py'),
                  'isaac_rig_sha256': A.sha256_file(S.ROOT / 'sim/isaac_rig.py'), 'lightings': ['L1'],
                  'render_settings_sha256': A.sha256_json(settings)}
        redo, why = B.needs_render(dest, data_bytes, expect)
        if redo:
            raise ValueError(f'existing output mismatch at {dest}: {why}')
        for stop in arr['stops']:
            names = [f'target_{ph}_{stop}.png' for ph in ('reference', 'visible', 'occluded')]
            names += [f'classes_{ph}_{stop}.png' for ph in ('reference', 'visible', 'occluded', 'removed')]
            names += [f'target_alone_{stop}.png', f'target_alone_revisit_{stop}.png']
            if any(not (dest / 'masks' / name).is_file() for name in names):
                raise ValueError(f'incomplete masks at {dest}')
    prior = {ident: S.digest_tree(output / ident) for ident in S.IDS}
    binding = {'study_id': S.STUDY, 'smoke_capsule_sha256': SMOKE_CAPSULE_SHA256,
               'smoke_inputs': smoke_inputs, 'geometry_audit_sha256': AUDIT_SHA256,
               'arrangements': audit['arrangement_file_sha256'], 'prior_smoke_files': prior,
               'cohort_launcher_sha256': A.sha256_file(__file__)}
    return pending, binding


def main():
    ap = argparse.ArgumentParser()
    for key in ('geometry', 'output', 'assets', 'tags', 'textures', 'audit'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--capsule-out')
    a = ap.parse_args()
    pending, binding = preflight(a.geometry, a.output, a.assets, a.tags, a.textures, a.audit)
    if a.execute and (sys.version_info[:2] != (3, 12) or B.isaac_version() != '6.2.0.0'):
        raise ValueError('execution requires Isaac Sim 6.2.0.0 and Python 3.12')
    output = Path(a.output).resolve()
    capsule = output / 'cohort_inputs.json'
    if capsule.exists() and json.loads(capsule.read_text()) != binding:
        raise ValueError('resume input provenance mismatch')
    if a.capsule_out:
        with Path(a.capsule_out).open('x') as f:
            json.dump(binding, f, indent=2)
            f.write('\n')
    if not a.execute or not pending:
        print(json.dumps({'valid': True, 'pending': len(pending), 'would_start_gpu': bool(a.execute and pending)}))
        return
    if not capsule.exists():
        with capsule.open('x') as f:
            json.dump(binding, f, indent=2)
            f.write('\n')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    cmd = [sys.executable, '-m', 'sim.isaac_batch', '--arrangements', *map(str, pending),
           '--assets-dir', a.assets, '--tags-dir', a.tags, '--textures-dir', a.textures,
           '--out', str(output), '--lighting', 'L1', '--skip-existing']
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1')
    started = time.monotonic()
    with (output / f'cohort-launch-{stamp}.log').open('x') as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=S.ROOT)
    with (output / f'cohort-launch-{stamp}.json').open('x') as f:
        json.dump({'command': cmd, 'cwd': str(S.ROOT), 'PYTHONPATH': env.get('PYTHONPATH'),
                   'start_utc': stamp, 'exit_code': code,
                   'elapsed_s': time.monotonic() - started}, f, indent=2)
        f.write('\n')
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
