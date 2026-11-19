import dataclasses
import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
from development.cross_sim import geometry_run as G


class Encoder:
    patch=14
    def __call__(self,rgb):
        h,w=rgb.shape[:2]; means=rgb.reshape(h//14,14,w//14,14,3).mean(axis=(1,3)).astype(float)+1
        return means/np.linalg.norm(means,axis=2,keepdims=True)


class Detector:
    def __call__(self,image):
        return [G.ev.Detection((3.,4.,23.,24.),'object',.75)]


class FixedProvider:
    def __init__(self,h,provided_ok=True): self.h=h;self.ok=provided_ok
    def register(self,ref,query,box):
        linear,valid,_=G.ev.register(ref,query,lambda r,q:self.h)
        nearest,_,_=G.ev.register(ref,query,lambda r,q:self.h,cv2.INTER_NEAREST)
        return linear,valid,{'warped_nearest_bgr':nearest,'registration_ok':self.ok,'reason':'accepted' if self.ok else 'refused'}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(93); self.ref=rng.integers(1,240,(180,240,3),dtype=np.uint8)
        self.query=np.roll(self.ref,5,axis=1);self.box=[50,50,95,100]
        self.p=G.ev.Params(); self.v=G.v03.V03Params();self.encoder=Encoder();self.detector=Detector()
        self.thresholds={'quality':.3,'present':.7,'absent':.6,'obstruction':.4}

    def compare_original(self,h):
        old,diag=G.v03.evidence_for(self.ref,self.box,self.query,0,self.encoder,self.detector,self.p,self.v,align=lambda r,q:h)
        provider=FixedProvider(h,provided_ok=h is not None)
        got=G.process(self.ref,self.query,self.box,provider,{'method':'baseline_200'},self.encoder,self.detector,self.p,self.v,self.thresholds)
        for k,v in dataclasses.asdict(old).items(): self.assertAlmostEqual(got['evidence'][k],v,places=12)
        for k in ('align_ok','quality_parts','obstruction_parts','detections','obstruction_v021','v03_parts'): self.assertEqual(got['diagnostics'][k],diag[k])
        return got

    def test_identity_shift_and_refusal_match_frozen(self):
        self.compare_original(np.eye(3)); self.compare_original(np.array([[1,0,-5.3],[0,1,2.2],[0,0,1.]]))
        refused=self.compare_original(None); self.assertEqual(refused['evidence']['quality'],0);self.assertEqual(refused['answer'],'UNKNOWN')

    def test_detector_offsets_after_warp(self):
        got=self.compare_original(np.eye(3))
        x0,y0,_,_=G.ev.grow(self.box,self.p.det_context,self.ref.shape[1],self.ref.shape[0])
        self.assertEqual(got['diagnostics']['detections'][0]['box'],[x0+3.,y0+4.,x0+23.,y0+24.])

    def test_partial_valid_mask_and_refusal(self):
        class Partial:
            def register(s,ref,q,box):
                mask=np.ones(ref.shape[:2],bool); mask[50:100,50:73]=False
                return q.copy(),mask,{'warped_nearest_bgr':q.copy(),'registration_ok':True,'reason':'ok'}
        cfg={'method':'geometry','min_target_valid_fraction':.8}
        out=G.process(self.ref,self.query,self.box,Partial(),cfg,self.encoder,self.detector,self.p,self.v,self.thresholds)
        self.assertEqual(out['refusal'],'target_valid_fraction_below_threshold');self.assertEqual(out['answer'],'UNKNOWN')
        self.assertEqual(out['evidence']['quality'],0);self.assertLess(out['diagnostics']['quality_parts']['valid'],1)
        self.assertAlmostEqual(out['diagnostics']['registration']['target_valid_fraction'],22/45)

    def test_rejects_missing_nearest_and_invalid_config(self):
        class Bad:
            def register(s,r,q,b): return q,np.ones(r.shape[:2],bool),{'registration_ok':True,'reason':'ok'}
        with self.assertRaises(ValueError): G.process(self.ref,self.query,self.box,Bad(),{'method':'geometry','min_target_valid_fraction':.8},self.encoder,self.detector,self.p,self.v,self.thresholds)
        with self.assertRaises(ValueError): G.validate_config({'method':'geometry','provider':'x:Y','provider_config':{},'min_target_valid_fraction':1.01})

    def test_artifact_bounds_integrity_and_corruption(self):
        valid=np.ones((675,1200),bool); valid[0,0]=False; box=[0,0,3,2]
        class Artifact:
            def geometry_artifact(s):
                return {'bounds_xyxy':[0,0,4,3],'query_xy':np.ones((3,4,2),dtype=np.float64),'valid':valid[:3,:4].copy()}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); G.save_artifact(Artifact(),'I123',box,valid,folder,'binding')
            meta=G.verify_artifact(folder,'I123',box,'binding');self.assertEqual(meta['query_xy_dtype'],'float32')
            with np.load(folder/'I123.npz',allow_pickle=False) as a:
                self.assertEqual(a['query_xy'].dtype,np.float32);self.assertEqual(a['valid'].dtype,bool)
            data=folder/'I123.npz'; data.write_bytes(data.read_bytes()+b'corruption')
            with self.assertRaises(ValueError): G.verify_artifact(folder,'I123',box,'binding')
            data.unlink()
            with self.assertRaises(FileNotFoundError): G.verify_artifact(folder,'I123',box,'binding')

    def test_fixed_candidate_threshold(self):
        config={'method':'geometry','provider':'a:B','provider_config':{},'min_target_valid_fraction':.60}
        G.validate_config(config)
        config['min_target_valid_fraction']=.59
        with self.assertRaises(ValueError): G.validate_config(config)


if __name__=='__main__': unittest.main()
