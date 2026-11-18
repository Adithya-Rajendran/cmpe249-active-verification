"""Synthetic checks for mask proxy labels and their source binding."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from development.stress.alignment_diagnostics import check_frozen
from development.stress.label_proxy import classify, export
from development.stress.prepare_inputs import STUDY, PROTOCOL, sha

class ProxyTests(unittest.TestCase):
    def test_cutoffs_and_unknown_are_explicit(self):
        self.assertEqual(classify('PRESENT', .9, None)[0], 'PRESENT')
        self.assertEqual(classify('PRESENT', .899, None)[0], 'UNKNOWN')
        self.assertEqual(classify('REMOVED', None, .099)[0], 'REMOVED')
        label, cue = classify('REMOVED', None, .1)
        self.assertEqual(label, 'UNKNOWN'); self.assertIn('does not establish visual unanswerability', cue)

    def test_source_hash_and_prepared_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'prepared'; images=p/'verifier/primary'; images.mkdir(parents=True)
            render=root/'render'; render.mkdir(); labels=render/'labels.json'
            labels.write_text(json.dumps({'arrangement_id':'A', 'visible_fraction':{'visible':{'P1':.9}},
                                          'region_obstruction':{}}))
            cohort=root/'cohort.json'; cohort.write_text(json.dumps({'study_id':STUDY, 'records':[{'arrangement_id':'A',
                'base_id':'B', 'environment':'shelf', 'level_id':'D0', 'render_directory':'render'}]}))
            image=images/'x.png'; image.write_bytes(b'synthetic image')
            jobs=[{'id':'I','reference':'x.png','query':'x.png','reference_sha256':sha(image),'query_sha256':sha(image),'box':[1,2,3,4]}]
            jp=images/'jobs.json'; jp.write_text(json.dumps(jobs))
            ledger=[{'id':'I','arrangement_id':'A','base_id':'B','environment':'shelf','level':'D0','phase':'visible',
                     'stop':'P1','physical_state':'PRESENT','visible_fraction':.9,'region_obstruction':None,'labels_sha256':sha(labels)}]
            lp=p/'ledger.json';lp.write_text(json.dumps(ledger))
            (p/'preparation.json').write_text(json.dumps({'study_id':STUDY,'protocol_sha256':sha(PROTOCOL),'cohort_sha256':sha(cohort),
                'frozen_hashes':check_frozen(),'ledger_sha256':sha(lp),'jobs_sha256':{'primary':sha(jp)},'count':1}))
            out=root/'proxy.csv'; export(p,cohort,out,'primary')
            with out.open() as f: row=next(csv.DictReader(f))
            self.assertEqual(row['label'], 'PRESENT');self.assertEqual(row['source'], 'geometry_proxy')
            labels.write_text(labels.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'source labels hash'): export(p,cohort,root/'bad.csv','primary')
            self.assertFalse((root/'bad.csv').exists())

if __name__ == '__main__': unittest.main()
