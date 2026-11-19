import copy
import json
from pathlib import Path
import unittest
from development.cross_sim import cohort as C
from sim import arrange as A


def test_sign_balance_and_determinism():
    p = json.loads(C.PROTOCOL.read_text())
    for split in p['splits']:
        for env in p['environments']:
            signs = C.signs(p, split, env)
            assert signs == C.signs(p, split, env)
            counts = [signs.count(pair) for pair in C.PAIRS]
            assert max(counts) - min(counts) <= 1


def test_reservations_refuse_overlap_and_test_seeds():
    p = json.loads(C.PROTOCOL.read_text())
    C.validate(p, set())
    with unittest.TestCase().assertRaises(ValueError):
        C.validate(p, {9000})
    bad = copy.deepcopy(p)
    bad['splits']['qualification']['seed_starts']['shelf'] = 100000
    with unittest.TestCase().assertRaises(A.SeedError):
        C.validate(bad, set())
    bad = copy.deepcopy(p)
    bad['splits']['pilot']['seed_starts']['shelf'] = 8000
    with unittest.TestCase().assertRaises(ValueError):
        C.validate(bad, set())


def test_materialized_manifests():
    root = C.ROOT / 'runs/2026-11-17-cross-sim'
    if not (root / 'cohort-pilot/manifest.json').exists():
        raise unittest.SkipTest('cohort inputs not generated')
    p = json.loads(C.PROTOCOL.read_text())
    seen_seeds, seen_layouts = set(), set()
    historical_seeds, historical_layouts, sources = C.historical(p)
    assert sources
    for split in ('qualification', 'pilot'):
        folder = root / f'cohort-{split}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['states'] == p['states'] and manifest['stops'] == p['stops']
        assert manifest['levels'] == p['splits'][split]['levels']
        assert len(manifest['arrangements']) == 2 * p['splits'][split]['count_per_environment']
        for item in manifest['arrangements']:
            assert item['seed'] not in historical_seeds | seen_seeds
            seen_seeds.add(item['seed'])
            expected_objects = None
            for level, ref in item['levels'].items():
                path = folder / ref['path']
                assert path.parent == folder and A.sha256_file(path) == ref['sha256']
                arr = json.loads(path.read_text())
                assert arr['seed'] == item['seed'] and arr['base_id'] == item['base_id']
                assert arr['arrangement_sha256'] == A.arrangement_hash(arr)
                assert arr['level'] == level
                assert all(x['margin_min_unrounded_px'] >= A.load_context(C.B.CONFIGS[item['environment']]).config['revisit']['min_margin_px'] for x in arr['revisit']['stops'].values())
                assert set(arr['objects']['occluder']['poses']) == {'placed', 'parked'}
                if expected_objects is None:
                    expected_objects = arr['objects']
                    digest = C.layout_hash(arr)
                    assert digest not in historical_layouts | seen_layouts
                    seen_layouts.add(digest)
                else:
                    assert arr['objects'] == expected_objects


if __name__ == '__main__':
    for test in (test_sign_balance_and_determinism, test_reservations_refuse_overlap_and_test_seeds, test_materialized_manifests):
        test()
        print('PASS', test.__name__)
