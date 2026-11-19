"""Prepare blinded cross-simulator image inputs after independent acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
import secrets
from pathlib import Path
import numpy as np
from PIL import Image
from development.stress.prepare_inputs import clean_image, load, sha
from development.stress.alignment_diagnostics import check_frozen
from sim.arrange import arrangement_hash, check_dev_seed

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name('scoring_protocol.json')
STUDY = 'cross-sim-dev-v1'
STOPS = ['P1', 'P2', 'P3', 'P4', 'P5']
STATES = ['clear_present', 'clear_removed', 'placed_present', 'placed_removed']
ENGINES = ['isaac', 'mujoco']
QUALIFICATION_PROTOCOL = Path(__file__).with_name('qualification_v1.json')


def check_fixed_protocol():
    cross = load(PROTOCOL); stress = load(ROOT / 'development/stress/protocol.json')
    for key in ('thresholds','align_params','render_size','primary_size','resize'):
        if cross[key] != stress[key]: raise ValueError('fixed stress settings changed')
    expected = [x for x in stress['analyses'] if x['id'] in ('primary','relaxed_guard','native_normalized_guard')]
    if cross['analyses'] != expected: raise ValueError('fixed control settings changed')
    return cross


def identifier(engine, base, level, stop, state):
    return 'I' + hashlib.sha256(json.dumps([STUDY, engine, base, level, stop, state], separators=(',', ':')).encode()).hexdigest()[:24]


def box_from_mask(path):
    with Image.open(path) as im:
        if im.size != (2304, 1296): raise ValueError('invalid mask size')
        mask = np.asarray(im)
    if mask.ndim == 3:
        if mask.shape[2] not in (3, 4): raise ValueError('invalid mask channels')
        mask = np.any(mask[..., :3] != 0, axis=2)
    elif mask.ndim == 2:
        mask = mask != 0
    else: raise ValueError('invalid mask')
    ys, xs = np.nonzero(mask)
    if not len(xs): raise ValueError('empty reference mask')
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def validate_manifest(path):
    path = Path(path).resolve(); m = load(path)
    if m.get('study_id') != STUDY or m.get('split') != 'pilot' or m.get('stops') != STOPS or m.get('states') != STATES or m.get('levels') != ['D0','D4','D6']:
        raise ValueError('pilot manifest contract mismatch')
    records = m.get('arrangements', [])
    if len(records) != 10 or len({r['base_id'] for r in records}) != 10 or len({r['seed'] for r in records}) != 10:
        raise ValueError('invalid pilot cohort')
    if any(sum(r['environment'] == e for r in records) != 5 for e in ('shelf', 'tabletop')):
        raise ValueError('environment count')
    for r in records:
        check_dev_seed(r['seed'])
        if set(r['levels']) != set(m['levels']): raise ValueError('missing levels')
        objects = None
        for level, entry in r['levels'].items():
            file = (path.parent / entry['path']).resolve()
            if file.parent != path.parent or sha(file) != entry['sha256']: raise ValueError('arrangement hash mismatch')
            arr = load(file)
            if any((arr.get('base_id') != r['base_id'], arr.get('seed') != r['seed'], arr.get('level') != level, arr.get('study_id') != STUDY, arr.get('arrangement_sha256') != arrangement_hash(arr))): raise ValueError('arrangement identity mismatch')
            if objects is not None and arr['objects'] != objects: raise ValueError('level geometry mismatch')
            objects = arr['objects']
    return m


def verify_release(release_path, stage, manifest_path, roots=None):
    release_path = Path(release_path).resolve(); release = load(release_path)
    if (release.get('study_id') != STUDY or release.get('stage') != stage or release.get('status') != 'PASS'
        or release.get('manifest_sha256') != sha(manifest_path) or release.get('protocol_sha256') != sha(QUALIFICATION_PROTOCOL)
        or set(release.get('engines', {})) != set(ENGINES)):
        raise ValueError('release binding mismatch')
    for engine in ENGINES:
        item = release['engines'][engine]; root = Path(item['root']).resolve(); audit = Path(item['audit_path']).resolve()
        if not Path(item['root']).is_absolute() or not Path(item['audit_path']).is_absolute() or sha(audit) != item['audit_sha256']:
            raise ValueError('audit binding mismatch')
        if roots is not None and root != roots[engine]: raise ValueError('accepted root mismatch')
        files = load(audit).get('files')
        if not isinstance(files, dict) or not files: raise ValueError('empty audit')
        for name, digest in files.items():
            path = (root / name).resolve()
            if path == root or root not in path.parents or sha(path) != digest: raise ValueError('accepted file mismatch')
    return release


def prepare(manifest_path, qualification_manifest, engine_roots, output, qualification_release, pilot_release):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    qualification_manifest = Path(qualification_manifest).resolve()
    check_fixed_protocol()
    m = validate_manifest(manifest_path)
    qualification = load(qualification_manifest)
    qrows = qualification.get('arrangements', [])
    if (qualification.get('study_id') != STUDY or qualification.get('split') != 'qualification' or not qrows or
            {r['base_id'] for r in m['arrangements']} & {r['base_id'] for r in qrows} or
            {r['seed'] for r in m['arrangements']} & {r['seed'] for r in qrows}):
        raise ValueError('qualification and pilot cohorts overlap or qualification is invalid')
    if set(engine_roots) != set(ENGINES): raise ValueError('both engines required')
    roots = {k:Path(v).resolve() for k,v in engine_roots.items()}
    qrelease = verify_release(qualification_release, 'qualification-accepted', qualification_manifest)
    prelease = verify_release(pilot_release, 'pilot-inputs-accepted', manifest_path, roots)
    if output.exists(): raise FileExistsError(output)
    inputs = []
    for engine in ENGINES:
        for r in m['arrangements']:
            for level in m['levels']:
                folder = roots[engine] / r['base_id'] / level
                rgb = folder / 'rgb'
                expected = {f'{state}_{stop}.png' for state in STATES for stop in STOPS}
                if level == 'D0': expected |= {f'reference_{stop}.png' for stop in STOPS}
                if {x.name for x in rgb.glob('*.png')} != expected: raise ValueError(f'incomplete or extra RGB files: {rgb}')
                for stop in STOPS:
                    nominal = roots[engine] / r['base_id'] / 'D0'
                    reference = nominal / 'rgb' / f'reference_{stop}.png'
                    mask = nominal / 'masks' / f'target_reference_{stop}.png'
                    box = box_from_mask(mask)
                    for state in STATES:
                        query = rgb / f'{state}_{stop}.png'
                        with Image.open(query) as im:
                            if im.size != (2304,1296): raise ValueError('query dimensions')
                        with Image.open(reference) as im:
                            if im.size != (2304,1296): raise ValueError('reference dimensions')
                        files = load(prelease['engines'][engine]['audit_path'])['files']
                        for source in (reference, query, mask):
                            relative = source.relative_to(roots[engine]).as_posix()
                            if files.get(relative) != sha(source): raise ValueError('required source not accepted')
                        inputs.append((engine,r,level,stop,state,reference,query,mask,box))
    if len(inputs) != 1200: raise ValueError('pilot count mismatch')
    frozen = check_frozen(); output.mkdir(parents=True)
    for size in ('primary','native'): (output/'verifier'/size).mkdir(parents=True)
    jobs = {'primary':[], 'native':[]}; ledger=[]
    # Opaque random names and a shuffled presentation order prevent reversing a
    # public hash or inferring the engine and state from the job sequence.
    secrets.SystemRandom().shuffle(inputs)
    for engine,r,level,stop,state,reference,query,mask,box in inputs:
        ident = 'I' + secrets.token_hex(16)
        row={'id':ident,'engine':engine,'base_id':r['base_id'],'environment':r['environment'],'level':level,'stop':stop,'state':state,
             'physical_state':'REMOVED' if state.endswith('_removed') else 'PRESENT', 'pair_id':identifier(engine,r['base_id'],level,stop,state.split('_')[0]),
             'arrangement_sha256':r['levels'][level]['sha256'],'source_reference_sha256':sha(reference),'source_query_sha256':sha(query),'source_mask_sha256':sha(mask),'reference_box_native':box}
        ledger.append(row)
        for name,dimensions in [('primary',[1200,675]),('native',[2304,1296])]:
            dest=output/'verifier'/name; rn=ident+'_r.png'; qn=ident+'_q.png'
            clean_image(reference,dest/rn,dimensions); clean_image(query,dest/qn,dimensions)
            scaled=[round(x*dimensions[i%2]/[2304,1296][i%2]) for i,x in enumerate(box)]
            jobs[name].append({'id':ident,'reference':rn,'query':qn,'box':scaled,'reference_sha256':sha(dest/rn),'query_sha256':sha(dest/qn)})
    if len({x['id'] for x in ledger}) != 1200: raise ValueError('duplicate ids')
    (output/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    for size in jobs: (output/'verifier'/size/'jobs.json').write_text(json.dumps(jobs[size],indent=2)+'\n')
    record={'study_id':STUDY,'protocol_sha256':sha(PROTOCOL),'cohort_sha256':sha(manifest_path),'ledger_sha256':sha(output/'ledger.json'),
            'jobs_sha256':{size:sha(output/'verifier'/size/'jobs.json') for size in jobs},'scope':'pilot','count':1200,'frozen_hashes':frozen,
            'qualification_manifest':str(qualification_manifest), 'cohort_manifest':str(manifest_path),
            'qualification_release':str(Path(qualification_release).resolve()), 'qualification_release_sha256':sha(qualification_release),
            'pilot_release':str(Path(pilot_release).resolve()), 'pilot_release_sha256':sha(pilot_release), 'preparer_sha256':sha(__file__), 'blinding':'random opaque identifiers and shuffled job order'}
    (output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--isaac',required=True); ap.add_argument('--mujoco',required=True); ap.add_argument('--qualification-manifest',required=True); ap.add_argument('--qualification-release',required=True); ap.add_argument('--pilot-release',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); prepare(a.manifest,a.qualification_manifest,{'isaac':a.isaac,'mujoco':a.mujoco},a.out,a.qualification_release,a.pilot_release)
