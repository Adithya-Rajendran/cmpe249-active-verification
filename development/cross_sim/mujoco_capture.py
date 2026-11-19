"""Experimental capture runner. Do not use until the reviewer releases the manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
from development.cross_sim import mujoco_geometry, mujoco_scene
from sim import isaac_rig


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preflight(manifest_path, declaration_path, assets):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    decl = json.loads(Path(declaration_path).read_text())
    if manifest['study_id'] != decl['study_id'] or manifest['split'] not in ('qualification', 'pilot'):
        raise ValueError('unexpected study or split')
    expected_levels = decl[manifest['split'] + '_levels']
    if (manifest['stops'] != decl['stops'] or manifest['states'] != decl['states'] or
            manifest['levels'] != expected_levels or manifest['lighting'] != 'L1'):
        raise ValueError('manifest differs from declaration')
    seen = set()
    rows = []
    for entry in manifest['arrangements']:
        ident, seed = entry['base_id'], entry['seed']
        if not ident.isalnum() or ident in seen or type(seed) is not int or not 0 <= seed <= 9999:
            raise ValueError(f'reserved or invalid identifier/seed: {ident}, {seed}')
        seen.add(ident)
        if set(entry['levels']) != set(expected_levels):
            raise ValueError('missing or extra levels')
        for level in expected_levels:
            rec = entry['levels'][level]
            path = (manifest_path.parent / rec['path']).resolve()
            if not path.is_relative_to(manifest_path.parent) or sha(path) != rec['sha256']:
                raise ValueError(f'input hash or path mismatch: {path}')
            arr = json.loads(path.read_text())
            if (arr.get('schema') != 'cmpe249.sim-arrangement.v1' or arr.get('split') != 'dev' or
                    arr.get('seed') != seed or arr.get('environment') != entry['environment'] or
                    arr.get('arrangement_id') != ident + level):
                raise ValueError(f'arrangement identity mismatch: {path}')
            for cam in arr['scene']['cameras'] + arr['scene']['revisit_cameras']:
                if [cam['width'], cam['height']] != decl['image_size']:
                    raise ValueError('unexpected image size')
            xml, inventory = mujoco_scene.build_xml(arr, assets/'models', assets/'tags', assets/'textures')
            files = {p: sha(p) for p in sorted(set(inventory['files']))}
            rows.append((ident, level, arr, xml, inventory, files, rec['sha256']))
    return manifest, rows


def identity(manifest_path, declaration_path, input_hash, files):
    sources = [Path(__file__), Path(mujoco_scene.__file__), Path(mujoco_geometry.__file__), Path(isaac_rig.__file__)]
    return {'manifest_sha256': sha(manifest_path), 'declaration_sha256': sha(declaration_path),
            'arrangement_sha256': input_hash, 'assets_sha256': files,
            'sources_sha256': {str(p.resolve()):sha(p) for p in sources},
            'runtime': {'python':sys.version, 'mujoco':mujoco.__version__, 'numpy':np.__version__,
                        'platform':platform.platform(), 'gl':os.environ.get('MUJOCO_GL')}}


def files_hash(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'complete.json'}


def capture(row, manifest, ident, out):
    name, level, arr, xml, inventory, files, input_hash = row
    final = out / name / level
    partial = final.with_name(final.name + '.partial')
    if partial.exists():
        raise RuntimeError(f'partial output exists; inspect and preserve it: {partial}')
    if final.exists():
        marker = final / 'complete.json'
        if not marker.is_file():
            raise RuntimeError(f'incomplete output exists: {final}')
        saved = json.loads(marker.read_text())
        if saved['identity'] != ident or saved['outputs'] != files_hash(final):
            raise RuntimeError(f'input or output changed: {final}')
        return 'resumed'
    partial.mkdir(parents=True)
    (partial/'rgb').mkdir()
    (partial/'masks').mkdir()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    h, w = arr['scene']['cameras'][0]['height'], arr['scene']['cameras'][0]['width']
    renderer = mujoco.Renderer(model, h, w)
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    blocker_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'blocker')
    target_ids = np.flatnonzero(model.geom_bodyid == target_body)
    blocker_ids = np.flatnonzero(model.geom_bodyid == blocker_body)
    states = {}
    start = time.monotonic()

    def image(state, stop, camera, prefix=None):
        label = prefix or state
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=f'{camera}_{stop}')
        Image.fromarray(renderer.render().copy()).save(partial/'rgb'/f'{label}_{stop}.png')
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=f'{camera}_{stop}')
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        isgeom = seg[:,:,1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        for role, ids in [('target', target_ids), ('blocker', blocker_ids)]:
            mask = (isgeom & np.isin(seg[:,:,0], ids)).astype(np.uint8) * 255
            Image.fromarray(mask).save(partial/'masks'/f'{role}_{label}_{stop}.png')

    try:
        parked = mujoco_scene.apply_state(model, data, arr, 'reference')
        placed = mujoco_scene.apply_state(model, data, arr, 'placed_present')
        observation = {'cameras':parked['cameras'], 'objects':{k:v for k,v in parked['objects'].items() if k != 'blocker'},
                       'static_bounds':parked['static_bounds'], 'states':states, 'inventory':inventory}
        observation['objects']['blocker_parked'] = parked['objects']['blocker']
        observation['objects']['blocker_placed'] = placed['objects']['blocker']
        if level == 'D0':
            states['reference'] = mujoco_scene.apply_state(model,data,arr,'reference')['state']
            for stop in manifest['stops']:
                image('reference', stop, 'nominal')
        for state in manifest['states']:
            states[state] = mujoco_scene.apply_state(model,data,arr,state)['state']
            for stop in manifest['stops']:
                image(state, stop, 'revisit')
        if manifest['split'] == 'qualification':
            states['repeat_clear_present'] = mujoco_scene.apply_state(model,data,arr,'clear_present')['state']
            for stop in manifest['stops']:
                image('clear_present', stop, 'revisit', 'repeat_clear_present')
        mujoco_scene.apply_state(model,data,arr,'clear_present')
        saved = model.geom_group.copy()
        model.geom_group[:] = 5
        model.geom_group[target_ids] = 0
        try:
            renderer.enable_segmentation_rendering()
            for stop in manifest['stops']:
                renderer.update_scene(data, camera=f'revisit_{stop}')
                seg = renderer.render().copy()
                mask = ((seg[:,:,1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[:,:,0],target_ids)).astype(np.uint8)*255
                Image.fromarray(mask).save(partial/'masks'/f'alone_{stop}.png')
            renderer.disable_segmentation_rendering()
        finally:
            model.geom_group[:] = saved
        observation['capture_seconds'] = time.monotonic() - start
        (partial/'observed.json').write_text(json.dumps(observation, indent=2)+'\n')
    finally:
        renderer.close()
    marker = {'identity':ident, 'outputs':files_hash(partial)}
    (partial/'complete.json').write_text(json.dumps(marker, indent=2)+'\n')
    partial.rename(final)
    return 'captured'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--declaration', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    manifest, rows = preflight(a.manifest, a.declaration, a.assets)
    if a.preflight_only:
        print(f'preflight: {len(rows)} level inputs, all assets verified and scene XML built')
        return
    for row in rows:
        ident = identity(a.manifest, a.declaration, row[-1], row[-2])
        print(row[0], row[1], capture(row, manifest, ident, a.out), flush=True)

if __name__ == '__main__':
    main()
