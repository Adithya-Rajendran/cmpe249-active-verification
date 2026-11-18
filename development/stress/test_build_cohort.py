import copy
import json
from pathlib import Path

import unittest

from development.stress import build_cohort as B
from sim import arrange as A


def protocol():
    return json.loads(B.PROTOCOL.read_text())


def test_signs_balanced_and_deterministic():
    p = protocol()
    table = B.signs(p)
    assert table == B.signs(p)
    for env in p['environments']:
        for stop in p['stops']:
            pairs = list(map(tuple, table[env][stop]))
            assert all(pairs.count(pair) == 3 for pair in ((1, 1), (1, -1), (-1, 1), (-1, -1)))


def test_rejects_invalid_study_and_reserved_seeds():
    p = protocol()
    p['study_id'] = 'test'
    with unittest.TestCase().assertRaises(ValueError):
        B.check_protocol(p)
    p = protocol()
    p['seed_starts']['shelf'] = 100000
    with unittest.TestCase().assertRaises(A.SeedError):
        B.check_protocol(p)


def test_motion_uses_camera_right_and_stays_fixed_across_levels():
    ctx = A.load_context(B.CONFIGS['shelf'])
    target = A.Body((0.05, 0.05, 0.05), (0.0, 0.15, 0.0))
    p = protocol()
    pairs = {stop: (1, -1) for stop in p['stops']}
    poses = B.motions(ctx, target, pairs, p['levels'])
    for cam in ctx.cams:
        d = poses['D6'][0][cam.name]['delta']
        assert abs((d['dx_m'] ** 2 + d['dy_m'] ** 2) ** 0.5 - 0.06) < 1e-8
        assert d['dyaw_deg'] == -3
        assert d['dz_m'] == d['dpitch_deg'] == d['droll_deg'] == 0
        assert poses['D0'][0][cam.name]['delta']['dx_m'] == 0


def test_revisit_metadata_serves_renderer_and_episode_consumers():
    from sim import episodes

    ctx = A.load_context(B.CONFIGS['shelf'])
    target = A.Body((0.05, 0.05, 0.05), (0.0, 0.15, 0.0))
    p = protocol()
    pairs = {stop: (1, -1) for stop in p['stops']}
    stops, _ = B.motions(ctx, target, pairs, p['levels'])['D6']
    summary = episodes.revisit_summary({'revisit': {'stops': stops}})
    for stop in p['stops']:
        rec = stops[stop]
        assert rec['attempt'] == rec['n_rejected'] == 0
        assert rec['nominal_margin_px']['min'] == A._margin_rec(
            A.frame_check(ctx, ctx.rig, ctx.rig.camera(stop), target.shape_points(), True))['min']
        assert summary[stop]['delta'] == rec['delta']
        assert isinstance(rec['margin_px']['min'], (int, float))


if __name__ == "__main__":
    for fn in (test_signs_balanced_and_deterministic, test_rejects_invalid_study_and_reserved_seeds, test_motion_uses_camera_right_and_stays_fixed_across_levels, test_revisit_metadata_serves_renderer_and_episode_consumers):
        fn()
        print(f"PASS {fn.__name__}")
