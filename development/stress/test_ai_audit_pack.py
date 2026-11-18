"""Synthetic binding tests for the audit pack. No image transform is used."""
import json
import tempfile
import unittest
from pathlib import Path
from development.stress.alignment_diagnostics import check_frozen
from development.stress.build_ai_audit_pack import verify_full
from development.stress.prepare_inputs import STUDY, sha

class AuditPackTests(unittest.TestCase):
    def test_full_binding_accepts_and_refuses_changed_sample_box(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manager = root/'manager'; manager.mkdir()
            full = root/'full'; images = full/'verifier/primary'; images.mkdir(parents=True)
            (images/'r.png').write_bytes(b'reference'); (images/'q.png').write_bytes(b'query')
            jobs = [{'id': f'I{i:04}', 'reference':'r.png', 'query':'q.png', 'box':[1,2,3,4],
                     'reference_sha256':sha(images/'r.png'), 'query_sha256':sha(images/'q.png')} for i in range(1800)]
            records = [{k:job[k] for k in ('id','reference_sha256','query_sha256','box')} for job in jobs[:180]]
            jp = images/'jobs.json'; jp.write_text(json.dumps(jobs))
            (full/'preparation.json').write_text(json.dumps({'study_id':STUDY, 'scope':'full', 'protocol_sha256':'p',
                'cohort_sha256':'c', 'jobs_sha256':{'primary':sha(jp)}, 'frozen_hashes':check_frozen(), 'count':1800}))
            manifest={'protocol_sha256':'p', 'cohort_sha256':'c', 'case_count':180, 'records':records}
            mp=manager/'manifest.json'; mp.write_text(json.dumps(manifest))
            verify_full(manager, full, root/'pass.json')
            self.assertEqual(json.loads((root/'pass.json').read_text())['matched_inputs'], 180)
            records[0]['box']=[1,2,3,5]; mp.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'full input mismatch'):
                verify_full(manager, full, root/'fail.json')
            self.assertFalse((root/'fail.json').exists())

if __name__ == '__main__': unittest.main()
