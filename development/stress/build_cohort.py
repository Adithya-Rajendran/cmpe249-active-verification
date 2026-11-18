"""Generate the declared development displacement cohort without starting Isaac Sim."""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
from pathlib import Path

import numpy as np

from sim import arrange as A
from sim import tabletop as T

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'development/stress/protocol.json'
STUDY = 'support-stress-dev-v1'
CONFIGS = {'shelf': 'configs/sim/arrange_shelf_v3.json', 'tabletop': 'configs/sim/arrange_tabletop_v1.json'}


def signs(p):
    pairs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] * 3
    return {env: {stop: [list(pairs[i]) for i in np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([p['sign_seed'], ei, si]))).permutation(12)]
        for si, stop in enumerate(p['stops'])} for ei, env in enumerate(p['environments'])}


def check_protocol(p):
    if p['study_id'] != STUDY or p['status'] != 'development':
        raise ValueError('invalid development study ID or status')
    for env in p['environments']:
        for seed in range(p['seed_starts'][env], p['seed_starts'][env] + 12 * p['seeds_per_slot']):
            A.check_dev_seed(seed)


def motions(ctx, target, pairs, levels):
    result = {}
    for level in levels:
        stops, cameras = {}, []
        for cam in ctx.cams:
            st, sy = pairs[cam.name]
            h = math.radians(cam.heading_deg)
            d = {'dx_m': round(st * level['translation_m'] * math.cos(h), 9),
                 'dy_m': round(-st * level['translation_m'] * math.sin(h), 9),
                 'dz_m': 0.0, 'dyaw_deg': sy * level['yaw_deg'], 'dpitch_deg': 0.0, 'droll_deg': 0.0}
            actual = A.perturb_camera(cam, d)
            margin = A.frame_check(ctx, ctx.rig, actual, target.shape_points(), True)
            nominal_margin = A.frame_check(ctx, ctx.rig, cam, target.shape_points(), True)
            stops[cam.name] = {'attempt': 0, 'n_rejected': 0, 'rejected': [],
                               'nominal_margin_px': A._margin_rec(nominal_margin),
                               'delta': d, 'sign_translation': st, 'sign_yaw': sy,
                               'nominal': A._cam_pose(cam), 'pose': A._cam_pose(actual),
                               'margin_px': A._margin_rec(margin), 'margin_min_unrounded_px': float(margin['min'])}
            cameras.append(T.camera_spec(actual))
        result[level['id']] = (stops, cameras)
    return result


def generate(out, p):
    check_protocol(p)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    table = signs(p)
    (out / 'sign_table.json').write_text(json.dumps(table, indent=2) + '\n')
    provenance = {'study_id': STUDY, 'protocol_sha256': A.sha256_file(PROTOCOL),
                  'builder_sha256': A.sha256_file(__file__), 'python': platform.python_version(),
                  'numpy': np.__version__, 'configs': {e: A.sha256_file(ROOT / c) for e, c in CONFIGS.items()}}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    logs, summaries, manifest = [], {}, []
    for env in p['environments']:
        ctx = A.load_context(CONFIGS[env])
        old = A.slot_list(ctx)
        slots = []
        for key in p['classes']:
            matches = [s for s in old if A.class_key(s['family'], s['m16_side']) == key][:2]
            if len(matches) != 2:
                raise ValueError(f'missing configured slots for {env} {key}')
            slots.extend(matches)
        middle = 0
        for i, slot in enumerate(slots):
            base = f'STR-{env.upper()}-{i+1:02d}'
            pair = {stop: table[env][stop][i] for stop in p['stops']}
            candidates = []
            log = {'base_id': base, 'source_slot': slot, 'attempts': []}
            logs.append(log)
            for seed in range(p['seed_starts'][env] + i * p['seeds_per_slot'], p['seed_starts'][env] + (i+1) * p['seeds_per_slot']):
                for attempt in range(p['attempts_per_seed']):
                    try:
                        arr = A.generate(seed, slot['family'], slot['m16_side'], attempt, ctx=ctx, target=slot['target'])
                    except (A.PlacementFailure, RuntimeError) as exc:
                        log['attempts'].append({'seed': seed, 'attempt': attempt, 'status': 'rejected', 'reason': str(exc)})
                        continue
                    target = A.body_of(arr['objects']['target'], ctx=ctx)
                    poses = motions(ctx, target, pair, p['levels'])
                    floor = float(ctx.config['revisit']['min_margin_px'])
                    fails = [{'level': level, 'stop': stop, 'margin_px': rec['margin_px']} for level, (stops, _) in poses.items()
                             for stop, rec in stops.items() if rec['margin_min_unrounded_px'] < floor]
                    if fails:
                        log['attempts'].append({'seed': seed, 'attempt': attempt, 'status': 'rejected', 'reason': 'frame', 'failures': fails})
                        continue
                    occ = A.body_of(arr['objects']['occluder'], 'placed', ctx)
                    coverage = {cam.name: A.coverage(cam, target, occ) for cam in ctx.cams}
                    score = sum(p['middle_band'][0] < x < p['middle_band'][1] for x in coverage.values())
                    log['attempts'].append({'seed': seed, 'attempt': attempt, 'status': 'candidate', 'coverage': coverage, 'score': score})
                    candidates.append((score, seed, attempt, arr, poses))
                    if len(candidates) == p['valid_candidates_per_slot']:
                        break
                if len(candidates) == p['valid_candidates_per_slot']:
                    break
            if not candidates:
                (out / 'generation_log.json').write_text(json.dumps(logs, indent=2) + '\n')
                raise RuntimeError(f'no valid candidates for {base}')
            score, seed, attempt, arr, poses = sorted(candidates, key=lambda v: (-v[0], v[1], v[2]))[0]
            middle += score
            log['selected'] = {'seed': seed, 'attempt': attempt, 'score': score, 'candidate_count': len(candidates)}
            for level in p['levels']:
                rec = copy.deepcopy(arr)
                rec['base_id'], rec['level'] = base, level['id']
                rec['arrangement_id'] = f"SX{'S' if env == 'shelf' else 'T'}{i+1:02d}{level['id']}"
                rec['study_id'] = STUDY
                rec['stress_provenance'] = provenance
                stops, cameras = poses[level['id']]
                rec['revisit'] = {'schema': 'cmpe249.stress-revisit.v1', 'stops': stops, 'level': level}
                rec['scene']['revisit_cameras'] = cameras
                rec['arrangement_sha256'] = A.arrangement_hash(rec)
                name = rec['arrangement_id'] + '.json'
                (out / name).write_text(json.dumps(rec, indent=2) + '\n')
                manifest.append({'base_id': base, 'environment': env, 'level_id': level['id'],
                                 'arrangement_id': rec['arrangement_id'], 'arrangement_file': name,
                                 'seed': seed, 'render_directory': '../renders/' + rec['arrangement_id']})
            (out / 'generation_log.json').write_text(json.dumps(logs, indent=2) + '\n')
        (out / 'cohort_manifest.json').write_text(json.dumps({'study_id': STUDY, 'path_root': 'directory containing this manifest', 'records': manifest}, indent=2) + '\n')
        summaries[env] = {'middle_stops': middle, 'total_stops': len(slots) * len(p['stops']),
                          'fraction': middle / (len(slots) * len(p['stops'])),
                          'passes': middle / (len(slots) * len(p['stops'])) >= p['middle_fraction_target']}
        (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    if not all(x['passes'] for x in summaries.values()):
        raise RuntimeError('analytic middle-band target failed; amendment required before render')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    generate(args.out, json.loads(PROTOCOL.read_text()))
