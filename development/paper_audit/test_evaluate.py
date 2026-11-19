import unittest
import json
import tempfile
from pathlib import Path
from PIL import Image
from development.paper_audit.inputs_packets import prepare
from development.paper_audit.evaluate_certificate import certificate
import numpy as np
from development.paper_audit.evaluate_witness import choose_patch, contrast, witness_pass, exposure
from development.paper_audit.evaluate_pairs import evaluate
from development.paper_audit.evaluate_results import bootstrap


class TestAudit(unittest.TestCase):
    def test_reference_patch_and_controls(self):
        mask = np.zeros((12,12),bool); mask[2:9,3:10] = True
        patch = choose_patch(mask,mask,[3,2,10,9])
        self.assertEqual(patch['xywh'], [3,2,7,7])
        a = np.zeros((12,12,3),np.uint8); b = a.copy(); b[2:9,3:10] = 10
        self.assertEqual(contrast(a,b,patch,mask,mask),10)
        self.assertTrue(witness_pass(8,4,3))
        self.assertFalse(witness_pass(8,5,3))
        self.assertEqual(exposure(mask,mask)['visible_fraction_of_alone'],1)
    def test_ties_and_ineligible_support(self):
        a=np.zeros((12,12),bool); a[2:9,2:10]=True
        self.assertEqual(choose_patch(a,a,[0,0,12,10])['center_xy'],[6,5])
        a[:]=False; a[5:7,5:7]=True
        self.assertIsNone(choose_patch(a,a,[0,0,12,12]))
        self.assertTrue(witness_pass(12,8,2))
        self.assertFalse(witness_pass(7.999,0,0))

    def test_packet_and_complete_input_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'source.png'
            Image.new('RGB',(10,10),(12,34,56)).save(image)
            instructions = root / 'instructions.txt'
            instructions.write_text('Compare these images.')
            packet = root / 'packet'
            mapping = prepare([{'private_id':'secret-retained','reference':str(image),
                                'query':str(image),'box_xyxy':[1,1,8,8]}],instructions,packet)
            public = json.loads((packet/'cases.json').read_text())
            self.assertNotIn('secret', (packet/'cases.json').read_text())
            self.assertEqual(mapping[0]['id'],public[0]['id'])
            case = {'reference':packet/public[0]['reference'], 'query':packet/public[0]['query'],
                    'instructions':packet/'instructions.txt', 'fields':{'box_xyxy':[1,1,8,8]}}
            self.assertTrue(certificate(case,case)['equal_complete_input'])
            other = dict(case,fields={'box_xyxy':[1,1,9,8]})
            self.assertFalse(certificate(case,other)['equal_complete_input'])

    def test_cluster_resampling_weights(self):
        rows=[{'environment':'shelf','arrangement':'A','truth':'REMOVED','answer':'REMOVED','correct':1},
              {'environment':'shelf','arrangement':'B','truth':'REMOVED','answer':'UNKNOWN','correct':0}]
        weights=np.array([[2,0],[0,2],[1,1]])
        ci=bootstrap(rows,weights,[('shelf','A'),('shelf','B')])
        self.assertAlmostEqual(ci['accuracy_all'][0],.025)
        self.assertAlmostEqual(ci['accuracy_all'][1],.975)
        self.assertIsNone(ci['false_removal_rate'])

    def test_pair_bootstrap_and_unknown(self):
        rows = [{'engine':'a','arrangement':'one','pair':'1','retained':'present','removed':'unknown'},
                {'engine':'a','arrangement':'two','pair':'2','retained':'absent','removed':'absent'}]
        m=evaluate(rows,100,1)['engines']['a']['metrics']
        self.assertEqual(m['coverage']['estimate'],.75)
        self.assertEqual(m['wrong_definite']['estimate'],.25)
        self.assertEqual(m['both_correct']['estimate'],0)
        with self.assertRaises(ValueError): evaluate(rows+rows[:1],10)

if __name__ == '__main__': unittest.main()
