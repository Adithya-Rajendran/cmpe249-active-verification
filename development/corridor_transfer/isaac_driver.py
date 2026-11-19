"""Bind and capture a reviewed cross-simulator cohort with Isaac Sim."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys

import numpy as np

from sim import isaac_batch as batch
from sim import isaac_rig as rig
from development.corridor_transfer import isaac_adapter as adapter
from development.cross_sim import isaac_uv_fixture as uv_fixture


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()


def inspect(manifest_path, assets, tags, textures, declaration):
    manifest_path = Path(manifest_path).resolve()
    data = json.loads(manifest_path.read_bytes())
    declaration = Path(declaration)
    declaration_hash = digest(declaration)
    if not json.loads(declaration.read_text()).get('design_locked', False):
        raise ValueError('Capture design is not locked')
    if data['split'] not in ('qualification', 'pilot'):
        raise ValueError('Only development qualification and pilot are allowed')
    if data['states'] != list(adapter.STATES)[1:] or data['stops'] != ['P1', 'P2', 'P3']:
        raise ValueError('Unexpected stops or intervention states')
    expected_levels = ['D0'] if data['split'] == 'qualification' else ['D0', 'D6']
    if data['levels'] != expected_levels:
        raise ValueError('Unexpected level list')
    asset_files = {}
    for label, root in [('models', assets), ('tags', tags), ('textures', textures)]:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f'Missing {label} directory')
        for path in sorted(root.rglob('*')):
            if path.is_file():
                asset_files[f'{label}/{path.relative_to(root)}'] = digest(path)
    items, ids = [], set()
    for record in data['arrangements']:
        base = record['base_id']
        if not base or Path(base).name != base or base in ids:
            raise ValueError('Invalid or repeated base ID')
        ids.add(base)
        if set(record['levels']) != set(expected_levels):
            raise ValueError(f'Level mismatch: {base}')
        for level in expected_levels:
            spec = record['levels'][level]
            path = (manifest_path.parent / spec['path']).resolve()
            if not path.is_relative_to(manifest_path.parent):
                raise ValueError('Arrangement must be inside cohort folder')
            if digest(path) != spec['sha256']:
                raise ValueError(f'Arrangement hash mismatch: {path}')
            arr = json.loads(path.read_bytes())
            batch.check_arrangement(arr)
            batch.check_scene(arr)
            if arr['stops'] != data['stops'] or arr['seed'] != record['seed']:
                raise ValueError(f'Arrangement identity mismatch: {path}')
            if arr.get('base_id') != base or arr.get('level') != level or arr.get('environment') != record['environment']:
                raise ValueError(f'Arrangement metadata mismatch: {path}')
            batch.check_render_inputs([(arr, b'')], ['L1'], textures)
            items.append((base, level, path, arr))
    source = {str(Path(p).resolve()): digest(p) for p in [__file__, adapter.__file__, uv_fixture.__file__, batch.__file__, rig.__file__]}
    binding = {'qualification_declaration_sha256': declaration_hash, 'manifest_sha256': digest(manifest_path), 'study_id': data['study_id'],
               'split': data['split'], 'assets': asset_files, 'source': source,
               'runtime': {'python': sys.version, 'platform': platform.platform(), 'isaac': batch.isaac_version()},
               'settings': {'lighting': 'L1', 'renderer': 'RaytracedLighting', 'settle': rig.SETTLE_UPDATES,
                            'capture_updates': batch.CAPTURE_UPDATES, 'rt_subframes': 16,
                            'alone_updates': batch.ALONE_UPDATES}}
    return data, items, binding


def xyz_and_rotation(record):
    m = record['world_from_local_row']
    return {'position': m[3][:3], 'rotation': [[m[j][i] for j in range(3)] for i in range(3)]}


def world_bounds(records, prefix):
    clouds = []
    for path, rec in records.items():
        if path != prefix and not path.startswith(prefix + '/'):
            continue
        if 'points_local' in rec:
            points = np.asarray(rec['points_local'], dtype=float)
        elif 'cube_size' in rec:
            import itertools
            h = rec['cube_size'] / 2
            points = np.asarray(list(itertools.product((-h, h), repeat=3)), dtype=float)
        else:
            continue
        if points.size:
            transform = np.asarray(rec['world_from_local_row'], dtype=float)
            clouds.append(points @ transform[:3, :3] + transform[3, :3])
    if not clouds:
        raise ValueError(f'No compiled geometry under {prefix}')
    points = np.concatenate(clouds)
    return [points.min(axis=0).tolist(), points.max(axis=0).tolist()]


def unified(arr, observed):
    initial = observed.get('reference', observed['clear_present'])
    cams = {}
    for group, root, key in [('nominal', 'Cameras', 'cameras'), ('revisit', 'CamerasRevisit', 'revisit_cameras')]:
        cams[group] = {}
        for c in arr['scene'][key]:
            rec = initial[f"/World/{root}/{c['name']}"]
            m = rec['world_from_local_row']
            width, height = int(c['width']), int(c['height'])
            focal = rec['focal_length'] / rec['horizontal_aperture'] * width
            cams[group][c['name']] = {'position': m[3][:3], 'right': m[0][:3], 'up': m[1][:3],
                'forward': [-x for x in m[2][:3]], 'f_px': focal, 'width': width, 'height': height,
                'principal_point_px': [width * (0.5 - rec['horizontal_aperture_offset'] / rec['horizontal_aperture']),
                                       height * (0.5 + rec['vertical_aperture_offset'] / rec['vertical_aperture'])]}
    objects = {'target': xyz_and_rotation(initial['/World/Objects/target']),
               'blocker_parked': xyz_and_rotation(observed['clear_present']['/World/Objects/occluder']),
               'blocker_placed': xyz_and_rotation(observed['placed_present']['/World/Objects/occluder'])}
    for i in range(len(arr['objects']['distractors'])):
        objects[f'distractor_{i}'] = xyz_and_rotation(initial[f'/World/Objects/distractor_{i}'])
    for name, obj in objects.items():
        prefix = '/World/Objects/occluder' if name.startswith('blocker_') else f'/World/Objects/{name}'
        # Move compiled geometry from its first captured pose to each measured pose.
        local_bounds = world_bounds(initial, prefix)
        if name == 'blocker_placed':
            # Recompute from local vertices at the observed placed transform.
            placed = dict(initial)
            old = np.asarray(initial[prefix]['world_from_local_row'])
            new = np.asarray(observed['placed_present'][prefix]['world_from_local_row'])
            for key, rec in list(placed.items()):
                if key == prefix or key.startswith(prefix + '/'):
                    copy = dict(rec)
                    copy['world_from_local_row'] = (np.asarray(rec['world_from_local_row']) @ np.linalg.inv(old) @ new).tolist()
                    placed[key] = copy
            local_bounds = world_bounds(placed, prefix)
        obj['bounds'] = local_bounds
    static_bounds = {}
    for part in arr['scene'].get('boxes', []):
        prefix = f"/World/Shelf/{rig._safe(part['role'])}_{rig._safe(part['name'])}"
        static_bounds[part['name']] = world_bounds(initial, prefix)
    for part in arr['scene'].get('meshes', []):
        if rig.is_static_mesh(part):
            prefix = f"/World/Shelf/{rig._safe(part['role'])}_{rig._safe(part['name'])}"
            static_bounds[part['name']] = world_bounds(initial, prefix)
    return {'cameras': cams, 'objects': objects, 'static_bounds': static_bounds, 'usd_per_state': observed}


def outputs(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name != 'completion.json'}


def expected_files(stops, level, qualification):
    states = list(adapter.STATES)[1:]
    if level == 'D0':
        states.insert(0, 'reference')
    if qualification:
        states.extend(f'repeat{n}_{state}' for n in (1, 2) for state in list(adapter.STATES)[1:])
        states.extend(f'distractor_hidden_{state}' for state in list(adapter.STATES)[1:])
    names = {'observed.json'}
    for stop in stops:
        names.add(f'masks/alone_{stop}.png')
        for state in states:
            names.add(f'rgb/{state}_{stop}.png')
            for kind in ('target', 'blocker'):
                names.add(f'masks/{kind}_{state}_{stop}.png')
    return names


def main():
    ap = argparse.ArgumentParser()
    for name in ('manifest', 'assets', 'tags', 'textures', 'output', 'binding', 'declaration'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--approved-binding-sha256')
    args = ap.parse_args()
    data, items, binding = inspect(args.manifest, args.assets, args.tags, args.textures, args.declaration)
    binding_path = Path(args.binding)
    if not args.execute:
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        with binding_path.open('x') as f:
            json.dump(binding, f, indent=2, sort_keys=True)
            f.write('\n')
        print('BINDING_SHA256', digest(binding_path), flush=True)
        return
    if not args.approved_binding_sha256 or digest(binding_path) != args.approved_binding_sha256:
        raise ValueError('Reviewed binding hash required')
    if json.loads(binding_path.read_text()) != binding:
        raise ValueError('Current inputs, code, runtime or settings differ from binding')
    out = Path(args.output)
    for base, level, _, _ in items:
        final = out / base / level
        partial = final.with_name(level + '.partial')
        if partial.exists():
            raise ValueError(f'Incomplete capture requires review: {partial}')
        if final.exists():
            done = json.loads((final / 'completion.json').read_text())
            if (done['binding_sha256'] != args.approved_binding_sha256 or done['outputs'] != outputs(final)
                    or set(done['outputs']) != expected_files(data['stops'], level, data['split'] == 'qualification')):
                raise ValueError(f'Existing capture changed: {final}')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from isaacsim import SimulationApp
    cam = items[0][3]['scene']['cameras'][0]
    app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'width': cam['width'], 'height': cam['height']})
    try:
        rig._tonemap()
        if data['split'] == 'qualification':
            fixture = out / 'uv_fixture'
            if fixture.exists():
                done = json.loads((fixture / 'completion.json').read_text())
                if done['binding_sha256'] != args.approved_binding_sha256 or done['outputs'] != outputs(fixture):
                    raise ValueError('Existing UV fixture changed')
            else:
                partial_fixture = out / 'uv_fixture.partial'
                uv_fixture.capture(app, partial_fixture)
                (partial_fixture / 'completion.json').write_text(json.dumps({'binding_sha256': args.approved_binding_sha256,
                    'outputs': outputs(partial_fixture)}, indent=2) + '\n')
                partial_fixture.rename(fixture)
        for base, level, path, arr in items:
            final = out / base / level
            if final.exists():
                continue
            final.parent.mkdir(parents=True, exist_ok=True)
            partial = final.with_name(level + '.partial')
            observed = adapter.capture_arrangement(app, arr, partial, args.assets, args.tags, args.textures,
                settle=rig.SETTLE_UPDATES, qualification=data['split'] == 'qualification', include_reference=level == 'D0')
            record = unified(arr, observed)
            record['arrangement_sha256'] = digest(path)
            record['binding_sha256'] = args.approved_binding_sha256
            (partial / 'observed.json').write_text(json.dumps(record, separators=(',', ':')) + '\n')
            if set(outputs(partial)) != expected_files(data['stops'], level, data['split'] == 'qualification'):
                raise ValueError(f'Incomplete capture: {partial}')
            (partial / 'completion.json').write_text(json.dumps({'binding_sha256': args.approved_binding_sha256,
                                                                  'outputs': outputs(partial)}, indent=2) + '\n')
            partial.rename(final)
            print('CAPTURED', base, level, flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()
