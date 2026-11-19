import unittest
import numpy as np
from development.cross_sim import landmark_residuals as L


class LandmarkTests(unittest.TestCase):
    def test_sample_centers_identity_and_translation(self):
        native=np.array([[100.5,200.5],[300.5,400.5]])
        np.testing.assert_allclose(L.sample_index(native,(2304,1296)),native-.5)
        np.testing.assert_allclose(L.sample_index(native,(1200,675)),native*(1200/2304)-.5)
        query=np.array([[0.,0.],[10.,10.],[3.,4.]])
        target=query+[7,-5]; h=np.array([[1,0,7],[0,1,-5],[0,0,1]])
        self.assertEqual(L.residuals(h,query,target,[0,0,30,40])['p90_px'],0)
        self.assertGreater(L.residuals(np.linalg.inv(h),query,target,[0,0,30,40])['median_px'],10)
        self.assertEqual(L.residuals(np.eye(3),query,query,[0,0,30,40])['median_px'],0)
        self.assertEqual(L.residuals(None,query,target,[0,0,30,40])['status'],'missing_matrix')
        self.assertEqual(L.residuals(np.zeros((3,3)),query,target,[0,0,30,40])['status'],'invalid_matrix')

    def test_frozen_extractor_direction_on_translation(self):
        import cv2
        from development.stress.alignment_diagnostics import load_alignment_only, observe_fit
        module=load_alignment_only(); rng=np.random.default_rng(177)
        ref=rng.integers(0,256,(256,256,3),dtype=np.uint8)
        # Moving reference image right 5 and down 3 creates query; expected query→reference is (-5,-3).
        query=cv2.warpAffine(ref,np.float32([[1,0,5],[0,1,3]]),(256,256))
        p=module.AlignParams(levels=2,stop_level=0,border=12,min_cc=.2,max_shift_px=50)
        accepted,info,raw=observe_fit(module,ref,query,(95,95,135,135),p)
        self.assertIsNotNone(accepted,info); self.assertIsNotNone(raw)
        true_query=np.array([[80.,80.],[160.,80.],[160.,160.],[80.,160.]])+[5,3]
        fit=L.residuals(raw,true_query,true_query-[5,3],[90,90,140,140])
        self.assertLess(fit['p90_px'],.8)
        wrong=L.residuals(np.linalg.inv(raw),true_query,true_query-[5,3],[90,90,140,140])
        self.assertGreater(wrong['median_px'],5)


if __name__=='__main__': unittest.main()
