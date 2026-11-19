"""Generate predeclared shared geometry without rendering or model outcomes."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from sim import arrange as A
from development.stress import build_cohort as B

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name('cohort_protocol.json')
PAIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def signs(p, split, env):
    count = p['splits'][split]['count_per_environment'] * len(p['stops'])
    env_index = p['environments'].index(env)
    seed = np.random.SeedSequence([p['sign_seed'], env_index, int(split == 'pilot')])
    order = np.random.Generator(np.random.PCG64(seed)).permutation(count)
    return [PAIRS[int(i) % 4] for i in order]


def layout_hash(arr):
    return hashlib.sha256(json.dumps(arr['objects'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def historical(p):
    seeds, layouts, sources = set(), set(), []
    exclude = ROOT / p['historical_exclude']
    roots = [ROOT / root for root in p['historical_search_roots']]
    mount = Path('/mnt/bench-data/datasets/cmpe249-active-verification/sim')
    if not mount.is_dir():
        raise RuntimeError('historical geometry mount unavailable')
    roots += sorted(d for d in mount.glob('*geometry') if 'test' not in d.name)
    for directory in roots:
        if not directory.exists():
            continue
        for file in directory.rglob('*.json'):
            if exclude in file.parents:
                continue
            try:
                arr = json.loads(file.read_text())
            except (ValueError, UnicodeError, OSError):
                continue
            if isinstance(arr, dict) and isinstance(arr.get('objects'), dict) and 'seed' in arr:
                if isinstance(arr['seed'], int):
                    seeds.add(arr['seed'])
                layouts.add(layout_hash(arr))
                sources.append(str(file))
    return seeds, layouts, sorted(sources)


def validate(p, old_seeds):
    if p['study_id'] != 'cross-sim-dev-v1' or p['status'] != 'development':
        raise ValueError('invalid study')
    declared = []
    for split in ('qualification', 'pilot'):
        for env in p['environments']:
            start = p['splits'][split]['seed_starts'][env]
            n = p['splits'][split]['count_per_environment'] * p['seeds_per_arrangement']
            for seed in range(start, start + n):
                A.check_dev_seed(seed)
                if seed in old_seeds:
                    raise ValueError(f'historical seed {seed}')
                declared.append(seed)
    if len(set(declared)) != len(declared):
        raise ValueError('overlapping seed reservations')
    # The complete earlier stress reserve is protected, including unselected seeds.
    stress = json.loads(B.PROTOCOL.read_text())
    for start in stress['seed_starts'].values():
        if set(declared) & set(range(start, start + 12 * stress['seeds_per_slot'])):
            raise ValueError('overlapping stress reservation')
    for config in B.CONFIGS.values():
        cfg = json.loads((ROOT / config).read_text())
        dev = cfg['slots']['dev']
        reserved = set(range(dev['primary_seed_start'], dev['primary_seed_start'] + len(A.slot_list(A.load_context(config)))))
        reserved.update(v for values in dev['reserve'].values() for v in values)
        if set(declared) & reserved:
            raise ValueError('overlapping configured reservation')


def generate(out_root, p):
    if platform.python_version() != '3.13.5' or np.__version__ != '2.5.3':
        raise RuntimeError('generation requires Python 3.13.5 and NumPy 2.5.3')
    old_seeds, old_layouts, sources = historical(p)
    validate(p, old_seeds)
    out_root = Path(out_root)
    if any((out_root / f'cohort-{s}').exists() for s in ('qualification', 'pilot')):
        raise FileExistsError('cohort output exists')
    provenance = {'protocol_sha256': A.sha256_file(PROTOCOL), 'builder_sha256': A.sha256_file(__file__),
                  'stress_builder_sha256': A.sha256_file(B.__file__), 'python': platform.python_version(),
                  'numpy': np.__version__, 'config_sha256': {e: A.sha256_file(ROOT / c) for e,c in B.CONFIGS.items()},
                  'historical_arrangement_files': sources, 'historical_seed_count': len(old_seeds),
                  'historical_layout_count': len(old_layouts), 'stress_protocol_sha256': A.sha256_file(B.PROTOCOL)}
    all_layouts = set(old_layouts)
    for split in ('qualification', 'pilot'):
        out = out_root / f'cohort-{split}'
        out.mkdir(parents=True)
        spec = p['splits'][split]
        levels = [x for x in p['levels'] if x['id'] in spec['levels']]
        manifest = {'study_id': p['study_id'], 'split': split, 'stops': p['stops'], 'states': p['states'],
                    'levels': spec['levels'], 'lighting': p['lighting'], 'reference': 'nominal_clear_present', 'arrangements': []}
        log = []
        for env in p['environments']:
            ctx = A.load_context(B.CONFIGS[env]); slots = A.slot_list(ctx); table = signs(p, split, env)
            for i in range(spec['count_per_environment']):
                # Pilot uses the five classes after the qualification class.
                cls = p['class_sequence'][i + (split == 'pilot')]
                slot = next(s for s in slots if A.class_key(s['family'], s['m16_side']) == cls)
                base = f"CS{'Q' if split == 'qualification' else 'P'}{'S' if env == 'shelf' else 'T'}{i+1:02d}"
                pair = dict(zip(p['stops'], table[i*len(p['stops']):(i+1)*len(p['stops'])]))
                selected = None
                start = spec['seed_starts'][env] + i * p['seeds_per_arrangement']
                for seed in range(start, start + p['seeds_per_arrangement']):
                    for attempt in range(p['attempts_per_seed']):
                        try:
                            arr = A.generate(seed, slot['family'], slot['m16_side'], attempt, ctx=ctx, target=slot['target'])
                            target = A.body_of(arr['objects']['target'], ctx=ctx)
                            poses = B.motions(ctx, target, pair, levels)
                            if any(rec['margin_min_unrounded_px'] < float(ctx.config['revisit']['min_margin_px']) for stops,_ in poses.values() for rec in stops.values()):
                                raise ValueError('frame')
                            digest = layout_hash(arr)
                            if digest in all_layouts:
                                raise ValueError('duplicate layout')
                        except (A.PlacementFailure, RuntimeError, ValueError) as exc:
                            log.append({'base_id':base, 'seed':seed, 'attempt':attempt, 'rejected':str(exc)})
                            continue
                        selected = (seed, attempt, arr, poses, digest); break
                    if selected: break
                if not selected: raise RuntimeError(f'no valid geometry for {base}')
                seed, attempt, arr, poses, digest = selected
                all_layouts.add(digest)
                log.append({'base_id':base, 'seed':seed, 'attempt':attempt, 'selected':True, 'layout_sha256':digest, 'slot':slot})
                item = {'base_id':base, 'environment':env, 'seed':seed, 'levels':{}}
                for level in levels:
                    rec = copy.deepcopy(arr); rec['base_id'] = base; rec['level'] = level['id']
                    rec['arrangement_id'] = base + level['id']; rec['study_id'] = p['study_id']
                    rec['stress_provenance'] = provenance
                    stops, cameras = poses[level['id']]
                    rec['revisit'] = {'schema':'cmpe249.stress-revisit.v1', 'stops':stops, 'level':level}
                    rec['scene']['revisit_cameras'] = cameras
                    rec['arrangement_sha256'] = A.arrangement_hash(rec)
                    path = out / (rec['arrangement_id'] + '.json')
                    path.write_text(json.dumps(rec, indent=2) + '\n')
                    item['levels'][level['id']] = {'path':path.name, 'sha256':A.sha256_file(path)}
                manifest['arrangements'].append(item)
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (out / 'generation_log.json').write_text(json.dumps(log, indent=2) + '\n')
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); args = ap.parse_args()
    generate(args.out, json.loads(PROTOCOL.read_text()))
