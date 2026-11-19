import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PIL import Image
from development.cross_sim import scoring_prepare as P, scoring_analysis as A, scoring_run as R
from development.stress.prepare_inputs import sha


class ScoringTests(unittest.TestCase):
    def test_protocol_matches_paired_study_and_controls(self):
        p=P.check_fixed_protocol()
        self.assertEqual(p['design']['arrangements_per_environment'],5)
        self.assertEqual([x['id'] for x in p['design']['levels']],['D0','D4','D6'])
        self.assertEqual(p['design']['states'],P.STATES)
        self.assertEqual(p['endpoint']['physical_state_source'],'simulator_intervention')
        self.assertEqual(p['decision']['stop_min_clear_correct_resolution_each_class'],.9)
        self.assertNotIn('support_label_source',p)

    def test_unguarded_runner_fails(self):
        with TemporaryDirectory() as tmp:
            record=Path(tmp)/'fake.json'
            record.write_text('{}')
            with self.assertRaises(ValueError): R.verify_offline_guard(record)

    def test_mask_and_gate(self):
        with TemporaryDirectory() as tmp:
            mask=np.zeros((1296,2304),dtype=np.uint8); mask[10:20,30:40]=255
            path=Path(tmp)/'mask.png'; Image.fromarray(mask).save(path)
            self.assertEqual(P.box_from_mask(path),[30,10,40,20])
        thresholds=json.loads(P.PROTOCOL.read_text())['thresholds']
        rec={'failure':False,'diagnostics':{'align_ok':True},'refusal':'accepted','evidence':{'quality':.3,'match_score':.7,'obstruction':.2}}
        self.assertEqual(A.answer(rec,thresholds),'PRESENT')
        rec['evidence']['match_score']=.6; rec['evidence']['obstruction']=.4
        self.assertEqual(A.answer(rec,thresholds),'REMOVED')
        rec['diagnostics']['align_ok']=False
        self.assertEqual(A.answer(rec,thresholds),'UNKNOWN')
        rec['failure']=True
        with self.assertRaises(ValueError): A.answer(rec,thresholds)

    def test_release_checks_every_accepted_file(self):
        with TemporaryDirectory() as tmp:
            tmp=Path(tmp); manifest=tmp/'manifest.json'; manifest.write_text('{}')
            engines={}
            for engine in P.ENGINES:
                root=tmp/engine; root.mkdir(); (root/'image.png').write_bytes(b'pixels')
                audit=tmp/(engine+'.json'); audit.write_text(json.dumps({'files':{'image.png':sha(root/'image.png')}}))
                engines[engine]={'root':str(root),'audit_path':str(audit),'audit_sha256':sha(audit)}
            release=tmp/'release.json'; release.write_text(json.dumps({'study_id':P.STUDY,'stage':'pilot-inputs-accepted','status':'PASS','manifest_sha256':sha(manifest),'protocol_sha256':sha(P.QUALIFICATION_PROTOCOL),'engines':engines}))
            P.verify_release(release,'pilot-inputs-accepted',manifest)
            (tmp/'isaac/image.png').write_bytes(b'changed')
            with self.assertRaises(ValueError): P.verify_release(release,'pilot-inputs-accepted',manifest)

    def test_summary_unknown_and_pair(self):
        rows=[{'state':'clear_present','answer':'UNKNOWN','physical_state':'PRESENT','pair_id':'c'},
              {'state':'clear_removed','answer':'REMOVED','physical_state':'REMOVED','pair_id':'c'},
              {'state':'placed_present','answer':'UNKNOWN','physical_state':'PRESENT','pair_id':'p'},
              {'state':'placed_removed','answer':'UNKNOWN','physical_state':'REMOVED','pair_id':'p'}]
        s=A.summarize(rows)
        self.assertEqual(s['coverage']['numerator'],1); self.assertEqual(s['correct_yield']['numerator'],1)
        self.assertEqual(s['clear_resolution']['clear_present']['numerator'],0)
        self.assertEqual(s['placed_both_unknown']['numerator'],1)
        self.assertEqual(s['placed_unknown']['placed_present']['numerator'],1)
        self.assertEqual(s['placed_unknown']['placed_removed']['numerator'],1)
        with self.assertRaises(ValueError): A.summarize(rows[:-1])

    def test_stop_and_joint_mechanism(self):
        rows=[]
        for engine in ('isaac','mujoco'):
            for env in ('shelf','tabletop'):
                for level in ('D0','D4','D6'):
                    for i in range(5):
                        for stop in range(5):
                            for state in ('clear_present','clear_removed','placed_present','placed_removed'):
                                physical='PRESENT' if state.endswith('present') else 'REMOVED'
                                rows.append({'engine':engine,'environment':env,'level':level,'base_id':f'{env}{i}', 'stop':f'P{stop+1}', 'state':state,'physical_state':physical,'answer':physical,'refusal':'accepted','alignment_valid':True,'quality':1.0,'target_match':.8 if physical=='PRESENT' else .2,'obstruction':0.0,'pair_id':f'{engine}{env}{level}{i}{stop}{state.split("_")[0]}'})
        controls={c:[dict(r) for r in rows] for c in A.CONTROLS}
        self.assertEqual(A.decide(controls)['outcome'],'stop_added_veto_cascade_claim')
        malformed={c:[dict(r) for r in group] for c,group in controls.items()}
        malformed['primary'][0]['stop'] = 'P2'
        with self.assertRaises(ValueError): A.decide(malformed)
        for group in controls.values():
            for r in group:
                if r['engine']=='isaac' and r['environment']=='shelf' and r['level']=='D6' and r['state']=='clear_present' and r['base_id'] in ('shelf0','shelf1','shelf2'):
                    r['answer']='UNKNOWN'; r['refusal']='shift'; r['alignment_valid']=False
        self.assertEqual(A.decide(controls)['outcome'],'continuing_mechanism')
        for r in controls['primary']:
            if r['base_id']=='shelf0' and r['engine']=='isaac' and r['level']=='D6' and r['state']=='clear_present': r['refusal']='scale'
        self.assertEqual(A.decide(controls)['outcome'],'inconclusive')

    def test_clear_abstention_reasons_are_disjoint(self):
        t=json.loads(P.PROTOCOL.read_text())['thresholds']
        base={'alignment_valid':True,'refusal':'accepted','quality':1.0,'target_match':.65,'obstruction':0.0}
        cases=[({'alignment_valid':False,'refusal':'shift'},'clear_unresolved:alignment:shift'),
               ({'quality':.2},'clear_unresolved:quality_below_threshold'),
               ({},'clear_unresolved:match_intermediate'),
               ({'target_match':.4,'obstruction':.5},'clear_unresolved:low_match_high_obstruction')]
        for changes,expected in cases:
            self.assertEqual(A.clear_unknown_reason({**base,**changes},t),expected)
        with self.assertRaises(ValueError): A.clear_unknown_reason({**base,'target_match':.9},t)

    def test_placed_abstentions_do_not_create_a_mechanism(self):
        rows=[]
        for engine in ('isaac','mujoco'):
            for env in ('shelf','tabletop'):
                for level in ('D0','D4','D6'):
                    for i in range(5):
                        for stop in range(5):
                            for state in P.STATES:
                                physical='PRESENT' if state.endswith('present') else 'REMOVED'
                                unknown=state.startswith('placed_')
                                # One clear shortfall prevents the global stop but does not recur.
                                if engine=='isaac' and env=='shelf' and level=='D6' and i==0 and state=='clear_present': unknown=True
                                rows.append({'engine':engine,'environment':env,'level':level,'base_id':f'{env}{i}',
                                    'stop':f'P{stop+1}','state':state,'physical_state':physical,
                                    'answer':'UNKNOWN' if unknown else physical,'refusal':'accepted',
                                    'alignment_valid':True,'quality':1.0,'target_match':.65 if unknown else (.8 if physical=='PRESENT' else .2),
                                    'obstruction':0.0,'pair_id':f'{engine}{env}{level}{i}{stop}{state.split("_")[0]}'})
        self.assertEqual(A.decide({c:[dict(r) for r in rows] for c in A.CONTROLS})['outcome'],'inconclusive')


if __name__=='__main__': unittest.main()
