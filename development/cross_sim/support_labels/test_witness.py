import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from development.cross_sim.support_labels import witness as W


class WitnessTests(unittest.TestCase):
    def test_patch_thresholds_repeat_and_edges(self):
        mask=np.ones((12,12),bool);present=np.zeros((12,12,3),np.uint8);removed=present.copy();removed[3:10,3:10]=12
        clear=present.copy();repeat=present.copy();repeat[3:10,3:10]=4
        x=W._evidence_for_point([6,6],mask,present,removed,clear,repeat)
        self.assertEqual(x['mask_pixels'],49);self.assertEqual(x['present_removed_mean_abs_channel_delta'],12)
        self.assertTrue(x['mask_majority'] and x['intervention_change_ge_8'] and x['intervention_minus_repeat_ge_4'])
        mask[3:10,3:10]=False;mask[3:8,3:8]=True
        self.assertTrue(W._evidence_for_point([6,6],mask,present,removed,clear,repeat)['mask_majority'])
        mask[3,3]=False;self.assertFalse(W._evidence_for_point([6,6],mask,present,removed,clear,repeat)['mask_majority'])
        self.assertEqual(W._evidence_for_point([2,2],mask,present,removed,clear,repeat)['status'],'patch_outside_image')
        self.assertEqual(W._evidence_for_point([3.5,4],mask,present,removed,clear,repeat)['status'],'invalid_point')

    def test_case_binds_prepared_and_opposite_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(W,'NATIVE',(24,16)),patch.object(W,'PRIMARY',(12,8)):
            root=Path(tmp); base=root/'CAP'/'LQS01';d0=base/'D0';folder=base/'D6'
            for f in (d0/'rgb',d0/'masks',folder/'rgb',folder/'masks'):f.mkdir(parents=True)
            rgb=np.full((16,24,3),90,np.uint8); present=rgb.copy();removed=rgb.copy()
            mask=np.zeros((16,24),np.uint8);mask[2:14,6:18]=255; zero=np.zeros_like(mask)
            def save(path,data):Image.fromarray(data).save(path)
            save(d0/'rgb/reference_P1.png',rgb);save(d0/'masks/target_reference_P1.png',mask)
            for state in ('placed_present','placed_removed','clear_present','repeat_clear_present'):save(folder/'rgb'/f'{state}_P1.png',present)
            for state in ('placed_present','clear_present','repeat_clear_present'):save(folder/'masks'/f'target_{state}_P1.png',mask)
            save(folder/'masks/target_placed_removed_P1.png',zero);save(folder/'masks/alone_P1.png',mask)
            for state in ('placed_present','placed_removed'):save(folder/'masks'/f'blocker_{state}_P1.png',zero)
            ref=root/'r.png';qry=root/'q.png';Image.fromarray(rgb).resize((12,8),Image.Resampling.LANCZOS).save(ref);Image.fromarray(present).resize((12,8),Image.Resampling.LANCZOS).save(qry)
            row={'id':'I1','engine':'mujoco','base_id':'LQS01','level':'D6','stop':'P1','state':'placed_present','condition':'partial',
                 'prepared_reference':str(ref),'prepared_query':str(qry),'reference_sha256':W.sha(ref),'query_sha256':W.sha(qry),'box_xyxy':[3,1,9,7]}
            vote={'answer':'UNKNOWN','point':None,'target':'can','anchor':'','cue':''};result=W.case_evidence(row,{'terra':vote,'main':vote},{'mujoco':root/'CAP'})
            self.assertTrue(result['exact_opposite_input_certificate']);self.assertEqual(result['mask_counts_native']['removed_target'],0)
            row['query_sha256']='0'*64
            with self.assertRaises(ValueError):W.case_evidence(row,{'terra':vote,'main':vote},{'mujoco':root/'CAP'})


if __name__=='__main__':unittest.main()
