"""Synthetic contract checks. Fixtures are not simulator results."""
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from PIL import Image, PngImagePlugin
from sim.arrange import arrangement_hash
from development.stress.prepare_inputs import (PROTOCOL, STUDY, annotation_pack, clean_image,
                                                json_hash, load, sha, validate_render, prepare, case_id)


class InputTests(unittest.TestCase):
    def test_stable_case_ids_do_not_depend_on_cohort_order(self):
        case = {'environment': 'shelf', 'base_id': 'S1', 'level_id': 'D6'}
        ident = case_id(case, 'P1', 'removed')
        self.assertEqual(ident, case_id(dict(reversed(list(case.items()))), 'P1', 'removed'))
        self.assertNotEqual(ident, case_id(case, 'P1', 'visible'))
        self.assertNotIn('D6', ident); self.assertNotIn('removed', ident)

    def test_clean_pixels_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            source, dest = Path(d) / 'source.png', Path(d) / 'clean.png'
            info = PngImagePlugin.PngInfo(); info.add_text('physical_state', 'REMOVED')
            Image.new('RGB', (2304, 1296), 'red').save(source, pnginfo=info)
            clean_image(source, dest, (1200, 675))
            with Image.open(dest) as image:
                self.assertEqual(image.size, (1200, 675)); self.assertEqual(image.info, {})

    def test_review_pack_is_blind_and_blank(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); directory = root / 'verifier/primary'; directory.mkdir(parents=True)
            (root / 'preparation.json').write_text('{}')
            for name in ('I00001_r.png', 'I00001_q.png'):
                Image.new('RGB', (1200, 675), 'white').save(directory / name)
            annotation_pack(root, [{'id': 'I00001', 'reference': 'I00001_r.png',
                                    'query': 'I00001_q.png', 'box': [10, 20, 30, 40],
                                    'reference_sha256': sha(directory / 'I00001_r.png'), 'query_sha256': sha(directory / 'I00001_q.png')}])
            with (root / 'review/blank_labels.csv').open() as f:
                self.assertEqual(list(csv.DictReader(f))[0]['label'], '')
            with Image.open(root / 'review/images/I00001_q.png') as im:
                self.assertEqual(im.getpixel((10, 20)), (255, 255, 255))
            with Image.open(root / 'review/images/I00001_r.png') as im:
                self.assertEqual(im.getpixel((10, 20)), (255, 0, 0))
            html = (root / 'review/index.html').read_text()
            self.assertNotIn('base_id', html); self.assertNotIn('physical_state', html)
            self.assertIn('localStorage', html)

    def test_smoke_reuses_d0_reference_at_d6(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); records = []; lookup = {}
            for env, seed in [('shelf', 8000), ('tabletop', 8500)]:
                base = 'B-' + env
                for level in ('D0', 'D6'):
                    folder = root / (env + level); (folder / 'rgb').mkdir(parents=True)
                    for stop in load(PROTOCOL)['stops']:
                        for phase in ('reference', 'visible', 'occluded', 'removed'):
                            (folder / 'rgb' / f'{phase}_{stop}_L1.png').write_bytes(f'{env}-{level}-{stop}-{phase}'.encode())
                    (folder / 'labels.json').write_text('{}'); (folder / 'render.json').write_text('{}')
                    arr = {'revisit': {'stops': {s: {'sign_translation': 1, 'sign_yaw': 1} for s in load(PROTOCOL)['stops']}}}
                    lab = {'reference_box': {'L1': {s: [10, 20, 30, 40] for s in load(PROTOCOL)['stops']}},
                           'visible_fraction': {}, 'region_obstruction': {}}
                    record = {'environment': env, 'base_id': base, 'level_id': level,
                              'arrangement_id': env+level, 'seed': seed, 'arrangement_file': env+level+'.json',
                              'render_directory': folder.name}
                    (root / record['arrangement_file']).write_text(json.dumps(arr))
                    records.append(record); lookup[(base, level)] = (arr, lab, folder)
            manifest = root / 'cohort.json'; manifest.write_text(json.dumps({'study_id': STUDY, 'records': records}))
            def fake_clean(source, dest, size): Path(dest).write_bytes(Path(source).read_bytes())
            with patch('development.stress.prepare_inputs.validate_render', side_effect=lambda r, *args: lookup[(r['base_id'], r['level_id'])]), \
                 patch('development.stress.prepare_inputs.check_frozen', return_value={}), \
                 patch('development.stress.prepare_inputs.clean_image', side_effect=fake_clean), \
                 patch('development.stress.prepare_inputs.annotation_pack'):
                prepare(manifest, root / 'out', 'smoke')
            ledger = load(root / 'out/ledger.json')
            for env in ('shelf', 'tabletop'):
                for stop in load(PROTOCOL)['stops']:
                    rows = [r for r in ledger if r['environment'] == env and r['stop'] == stop]
                    self.assertEqual(len({r['source_reference_sha256'] for r in rows}), 1)
                    d0 = next(r for r in rows if r['level'] == 'D0')
                    self.assertEqual(d0['source_reference_sha256'], sha(root / (env+'D0') / 'rgb' / f'reference_{stop}_L1.png'))

    def test_render_validation_rejects_tampered_png_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); folder = root / 'render'; (folder / 'rgb').mkdir(parents=True)
            proto = load(PROTOCOL); stops = proto['stops']
            delta = {'dx_m': 0, 'dy_m': 0, 'dz_m': 0, 'dyaw_deg': 0, 'dpitch_deg': 0, 'droll_deg': 0}
            arr = {'study_id': STUDY, 'stress_provenance': {'protocol_sha256': sha(PROTOCOL)},
                   'arrangement_id': 'TEST', 'base_id': 'BASE', 'level': 'D0', 'seed': 8000,
                   'revisit': {'stops': {s: {'delta': delta} for s in stops}}}
            arr['arrangement_sha256'] = arrangement_hash(arr)
            raw = json.dumps(arr).encode(); (root / 'arr.json').write_bytes(raw); (folder / 'arrangement.json').write_bytes(raw)
            settings = {'renderer': 'RaytracedLighting', 'rt_subframes': 16, 'settle_updates': 200,
                        'capture_updates': 100, 'alone_updates': 10, 'resolution': [[2304,1296]]*5, 'lightings': ['L1']}
            frozen = {'sim/isaac_batch.py': 'batch', 'sim/isaac_rig.py': 'rig'}
            hashes = {'arrangement_file_sha256': sha(root / 'arr.json'), 'arrangement_sha256': arr['arrangement_sha256'],
                      'isaac_batch_sha256': 'batch', 'isaac_rig_sha256': 'rig', 'render_settings_sha256': json_hash(settings)}
            labels = {'arrangement_id': 'TEST', 'hashes': hashes, 'stops': stops, 'lightings': ['L1'], 'label_sanity': {'ok': True}}
            render = {'arrangement_id': 'TEST', 'settings': settings, 'settings_sha256': json_hash(settings),
                      'isaac_batch_sha256': 'batch', 'isaac_rig_sha256': 'rig'}
            (folder / 'labels.json').write_text(json.dumps(labels)); (folder / 'render.json').write_text(json.dumps(render))
            image = Image.new('RGB', (2304, 1296))
            for phase in ['reference'] + proto['query_phases']:
                for stop in stops:
                    camera = 'nominal' if phase == 'reference' else 'revisit'
                    meta = {'arrangement_id': 'TEST', 'seed': 8000, 'phase': phase, 'stop': stop, 'lighting': 'L1',
                            'camera': camera, 'revisit_delta': None if camera == 'nominal' else delta,
                            'arrangement_sha256': arr['arrangement_sha256']}
                    info = PngImagePlugin.PngInfo(); info.add_text('cmpe249', json.dumps(meta))
                    image.save(folder / 'rgb' / f'{phase}_{stop}_L1.png', pnginfo=info)
            record = {'arrangement_id': 'TEST', 'base_id': 'BASE', 'level_id': 'D0', 'seed': 8000,
                      'environment': 'shelf', 'arrangement_file': 'arr.json', 'render_directory': 'render'}
            validate_render(record, root, proto, frozen)
            image.save(folder / 'rgb/reference_P1_L1.png')
            with self.assertRaises((KeyError, ValueError)):
                validate_render(record, root, proto, frozen)

if __name__ == '__main__': unittest.main()
