"""Synthetic integrity tests for qualification selection and reviewer release."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from development.stress.qualification_pack import SEED, ranked_diverse, release
from development.stress.prepare_inputs import sha

class QualificationTests(unittest.TestCase):
    def test_diverse_ranking_is_deterministic_and_balances_base(self):
        rows = [{'id': f'I{x}', 'base_id': base} for x,base in enumerate(['A','A','A','B','B','C'])]
        ids = ranked_diverse(rows)
        by_id={r['id']: r['base_id'] for r in rows}
        self.assertEqual(len({by_id[i] for i in ids[:3]}), 3)
        self.assertEqual(ids, ranked_diverse(list(reversed(rows))))
        first = min(rows, key=lambda r: (hashlib.sha256((SEED+r['id']).encode()).hexdigest(),r['id']))
        self.assertEqual(ids[0], first['id'])

    def test_release_requires_locked_review_and_hides_control_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); preview=root/'preview'; (preview/'images').mkdir(parents=True)
            plan=root/'plan.json'; plan.write_text(json.dumps({'arms':{'original':'Task instructions'}}))
            ids=[f'I{i:03}' for i in range(48)]
            selection=root/'selection.json'; selection.write_text(json.dumps({'qualification_plan_sha256':sha(plan),
                'cells':[{'candidate_ids':ids,'quota':48,'provisional_class':'PRESENT'}]}))
            records=[]
            for ident in ids:
                ref=preview/'images'/(ident+'_r.png'); qry=preview/'images'/(ident+'_q.png')
                ref.write_bytes(b'synthetic reference'); qry.write_bytes(b'synthetic query')
                records.append({'id':ident,'reference':ref.name,'query':qry.name,
                                'presented_reference_sha256':sha(ref),'presented_query_sha256':sha(qry)})
            pp=preview/'preview.json';pp.write_text(json.dumps({'selection_sha256':sha(selection),'records':records}))
            approval=root/'approval.json'
            data={'selection_sha256':sha(selection),'preview_sha256':sha(pp),'reviewer':'main reviewer',
                  'locked':False,'accepted_ids':ids,'excluded':[],
                  'reference_labels':{i:'PRESENT' for i in ids},'visual_cues':{i:'visible' for i in ids}}
            approval.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'locked'): release(selection,preview,approval,root/'out',plan,'original')
            self.assertFalse((root/'out').exists())
            data['locked']=True; data['reference_labels'][ids[0]]='UNKNOWN'; approval.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'reference labels'): release(selection,preview,approval,root/'out',plan,'original')
            data['reference_labels'][ids[0]]='PRESENT';approval.write_text(json.dumps(data))
            release(selection,preview,approval,root/'out',plan,'original')
            public=json.loads((root/'out/pairs.json').read_text())
            self.assertEqual(len(public['pairs']),48)
            self.assertEqual(set(public),{'instructions','pairs'})
            self.assertTrue(all(set(item)=={'id','reference','query','presented_reference_sha256','presented_query_sha256'} for item in public['pairs']))

if __name__ == '__main__': unittest.main()
