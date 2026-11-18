"""Synthetic analysis fixtures. These are not research results."""
import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sim.arrange import arrangement_hash
from development.stress.analysis import analyze, development_decisions, human_labels, recoverable_abstentions, human_physical_conflicts
from development.stress.parity_check import compare
from development.stress.alignment_diagnostics import check_frozen
from development.stress.prepare_inputs import PROTOCOL, STUDY, sha, load, json_hash


class AnalysisTests(unittest.TestCase):
    def test_human_labels_bind_exact_images_and_box(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'labels.csv'
            job = {'id': 'opaque', 'reference_sha256': 'ref', 'query_sha256': 'qry', 'box': [1, 2, 3, 4]}
            row = {'id': 'opaque', 'label': 'PRESENT', 'cue': 'mark', 'reviewer': 'person', 'source': 'human',
                   'input_size': '1200x675', 'reference_sha256': 'ref', 'query_sha256': 'qry',
                   'box_sha256': json_hash(job['box'])}
            def write():
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
            write(); self.assertEqual(human_labels(path, [job], '1200x675')['opaque']['label'], 'PRESENT')
            with self.assertRaises(ValueError): human_labels(path, [job], '1200x675', 'geometry_proxy')
            row['source'] = 'ai'; write()
            self.assertEqual(human_labels(path, [job], '1200x675', 'ai')['opaque']['label'], 'PRESENT')
            with self.assertRaises(ValueError): human_labels(path, [job], '1200x675', 'human')
            row['source'] = 'human'; write()
            for key, value in [('query_sha256', 'changed'), ('reference_sha256', 'changed'),
                               ('box_sha256', 'changed'), ('input_size', '2304x1296'), ('id', 'other')]:
                before = row[key]; row[key] = value; write()
                with self.assertRaises(ValueError): human_labels(path, [job], '1200x675')
                row[key] = before

    def test_conflicting_label_is_not_recoverable_useful_answer(self):
        rows = [
            {'arrangement': 'A', 'answer': 'UNKNOWN', 'support': 'PRESENT', 'physical': 'REMOVED'},
            {'arrangement': 'B', 'answer': 'UNKNOWN', 'support': 'REMOVED', 'physical': 'REMOVED'},
        ]
        stats = recoverable_abstentions(rows)
        self.assertEqual(stats['numerator'], 1)
        self.assertEqual(stats['denominator'], 2)
        self.assertEqual(stats['arrangement_rates'], {'A': 0, 'B': 1})
        self.assertEqual(stats['arrangement_weighted'], .5)
        self.assertEqual(human_physical_conflicts(rows), {'definite_human_labels': 2,
            'human_present_physical_removed': 1, 'human_removed_physical_present': 0, 'total': 1})
        self.assertEqual(rows[0]['support'], 'PRESENT')

    def test_decision_scope_undefined_and_headroom(self):
        def level(unsupported=0.0, errors=0, useful=.5, headroom=.2, sensitivity_headroom=.2):
            metrics = {'unsupported': {'estimate': unsupported}, 'unsupported_error_arrangements': errors,
                       'useful': {'estimate': useful}}
            def h(value): return {'numerator': 2, 'denominator': 10, 'arrangement_weighted': value}
            return {'human_undisputed': copy.deepcopy(metrics), 'disputed_as_unknown': copy.deepcopy(metrics),
                    'recoverable_abstentions': h(headroom),
                    'recoverable_abstentions_disputed_as_unknown': h(sensitivity_headroom)}
        levels = {key: level() for key in ('D0', 'D1', 'D2', 'D4', 'D6')}
        results = {'shelf': levels}
        self.assertEqual(development_decisions(results, 'full', True, 'native')['status'], 'not eligible')
        self.assertEqual(development_decisions(results, 'full', True, 'relaxed_guard')['status'], 'not eligible')
        levels['D1'] = level(unsupported=None)
        self.assertEqual(development_decisions(results, 'full', True, 'primary')['shelf']['outcome'], 'INDETERMINATE')
        levels['D1'] = level(unsupported=.02, errors=3, sensitivity_headroom=.05)
        decision = development_decisions(results, 'full', True, 'primary')['shelf']
        self.assertEqual(decision['outcome'], 'A')
        self.assertEqual(decision['cascade_headroom']['D1']['status'], 'threshold not met')
        self.assertEqual(decision['cascade_headroom']['D1']['primary']['numerator'], 2)
        levels['D1'] = level()
        levels['D6'] = level(useful=.3)
        levels['D6']['disputed_as_unknown']['useful']['estimate'] = .5
        decision = development_decisions(results, 'full', True, 'primary')['shelf']
        self.assertEqual(decision['outcome'], 'B')
        self.assertEqual(decision['cascade_headroom']['D6']['status'], 'threshold met')
        ai_levels = {name: {**value, 'ai_reference': value['human_undisputed'], 'ai_reference_disputed_as_unknown': value['disputed_as_unknown']} for name, value in levels.items()}
        ai_decision = development_decisions({'shelf': ai_levels}, 'full', True, 'primary', 'ai')
        self.assertEqual(ai_decision['status'], 'not eligible')
        self.assertIn('qualification', ai_decision['reason'])
        self.assertEqual(development_decisions(results, 'full', True, 'primary', 'geometry_proxy')['status'], 'not eligible')
        self.assertIn('pending', decision['cascade_headroom']['D6']['next_step'])

    def test_parity_comparison_detects_changes(self):
        same = compare({'x': [1.0, True], 'method': 'image'}, {'x': [1.0, True], 'method': 'image'})
        self.assertEqual(same, {'max_abs_difference': 0.0, 'nonnumeric_mismatches': []})
        changed = compare({'x': 1.0, 'accepted': True}, {'x': 1.25, 'accepted': False})
        self.assertEqual(changed['max_abs_difference'], .25)
        self.assertEqual(changed['nonnumeric_mismatches'], ['$.accepted'])

    def test_engineering_and_human_sensitivity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); prepared = root/'prepared'; jobs_dir = prepared/'verifier/primary'
            jobs_dir.mkdir(parents=True); run = root/'run'; (run/'records').mkdir(parents=True)
            jobs, ledger, human = [], [], []
            for index, (base, level, support) in enumerate([('A','D0','PRESENT'), ('B','D0','DISPUTED'),
                                                           ('A','D6','UNKNOWN'), ('B','D6','PRESENT')]):
                ident = f'I{index:05d}'; image = jobs_dir / (ident+'.png'); image.write_bytes(ident.encode())
                arr = {'example': ident}; arr['arrangement_sha256'] = arrangement_hash(arr)
                ap = root/(ident+'.json'); ap.write_text(json.dumps(arr))
                jobs.append({'id': ident, 'reference': image.name, 'query': image.name,
                             'reference_sha256': sha(image), 'query_sha256': sha(image), 'box': [0,0,1,1]})
                ledger.append({'id': ident, 'base_id': base, 'environment': 'shelf', 'level': level,
                               'physical_state': 'PRESENT', 'sign_translation': 1, 'sign_yaw': 1,
                               'region_obstruction': .5, 'arrangement_file': str(ap), 'arrangement_file_sha256': sha(ap),
                               'stop': 'P1', 'reference_box_native': [0,0,1,1]})
                human.append({'id': ident, 'label': support, 'cue': 'synthetic cue', 'reviewer': 'fixture', 'source': 'human', 'input_size': '1200x675',
                              'reference_sha256': sha(image), 'query_sha256': sha(image), 'box_sha256': json_hash([0,0,1,1])})
            jp=jobs_dir/'jobs.json'; jp.write_text(json.dumps(jobs)); lp=prepared/'ledger.json'; lp.write_text(json.dumps(ledger))
            frozen = check_frozen()
            pre = {'study_id': STUDY, 'scope': 'smoke', 'count': len(jobs), 'ledger_sha256': sha(lp), 'frozen_hashes': frozen,
                   'jobs_sha256': {'primary': sha(jp)}}
            pp=prepared/'preparation.json'; pp.write_text(json.dumps(pre))
            manifest = {'study_id': STUDY, 'analysis': 'primary', 'protocol_sha256': sha(PROTOCOL), 'frozen_hashes': frozen,
                        'preparation_sha256': sha(pp), 'jobs_sha256': sha(jp)}
            mp=run/'manifest.json'; mp.write_text(json.dumps(manifest))
            for job in jobs:
                rec = {'id': job['id'], 'binding_sha256': sha(mp), 'failure': False,
                       'evidence': {'quality': .9, 'match_score': .9, 'obstruction': .1},
                       'diagnostics': {'align_ok': True}, 'refusal': 'accepted', 'raw_H_query_to_reference': None}
                (run/'records'/f"{job['id']}.json").write_text(json.dumps(rec))
            labels=root/'labels.csv'
            with labels.open('w', newline='') as f:
                writer=csv.DictWriter(f, fieldnames=['id','label','cue','reviewer','source','input_size','reference_sha256','query_sha256','box_sha256']); writer.writeheader(); writer.writerows(human)
            with patch('development.stress.analysis.plane_diagnostics', return_value={}):
                analyze(prepared, run, root/'no-label.json')
                analyze(prepared, run, root/'human.json', labels)
            for source in ('ai', 'geometry_proxy'):
                for row in human: row['source'] = source
                with labels.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(human[0])); writer.writeheader(); writer.writerows(human)
                with patch('development.stress.analysis.plane_diagnostics', return_value={}):
                    analyze(prepared, run, root/(source+'.json'), labels, source)
                reference = load(root/(source+'.json'))
                self.assertIsNone(reference['human_labels'])
                self.assertEqual(reference['support_labels']['source'], source)
                self.assertIn(source+'_reference', reference['results']['shelf']['D0'])
                self.assertNotIn('human_undisputed', reference['results']['shelf']['D0'])
                self.assertEqual(reference['development_decisions']['status'], 'not eligible')
            for row in human: row['source'] = 'human'
            with labels.open('w', newline='') as f:
                writer=csv.DictWriter(f, fieldnames=list(human[0])); writer.writeheader(); writer.writerows(human)
            plain=load(root/'no-label.json'); report=load(root/'human.json')
            self.assertEqual(report['provenance']['analysis_sha256'], sha(Path(__file__).with_name('analysis.py')))
            self.assertEqual(report['provenance']['metrics_sha256'], sha(Path(__file__).parents[1] / 'pilot/metrics.py'))
            self.assertEqual(report['provenance']['alignment_diagnostics_sha256'], sha(Path(__file__).with_name('alignment_diagnostics.py')))
            self.assertEqual(report['provenance']['labels_sha256'], sha(labels))
            self.assertEqual(report['provenance']['ledger_sha256'], sha(lp))
            self.assertEqual(report['provenance']['jobs_sha256'], sha(jp))
            self.assertEqual(report['provenance']['run_manifest_sha256'], sha(mp))
            self.assertEqual(report['provenance']['protocol_sha256'], sha(PROTOCOL))
            self.assertNotIn('human_undisputed', plain['results']['shelf']['D0'])
            d0=report['results']['shelf']['D0']
            self.assertEqual(d0['human_undisputed']['unsupported']['denominator'], 1)
            self.assertEqual(d0['disputed_as_unknown']['unsupported']['denominator'], 2)
            self.assertEqual(d0['disputed_as_unknown']['unsupported']['estimate'], .5)
            self.assertEqual(report['development_decisions']['status'], 'not eligible')
            self.assertEqual(report['paired_bootstrap']['shelf']['D6']['human_undisputed:coverage']['paired_arrangements'], 2)
            manifest['frozen_hashes'] = {}
            mp.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'frozen source binding mismatch'):
                analyze(prepared, run, root/'refused.json')

if __name__ == '__main__': unittest.main()
