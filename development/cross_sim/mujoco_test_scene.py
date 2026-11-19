"""Synthetic scene compilation and camera-binding checks."""
import os
import unittest
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
from development.cross_sim.mujoco_scene import build_xml, apply_state
from development.cross_sim.mujoco_geometry import camera_axes


def fixture():
    cam = dict(name='P1', width=64, height=48, position=[1, -2, 3], forward=[-.2, 1, -.4], up=[0, 0, 1], f_px=50)
    mesh = dict(name='target_body', role='target', vertices=[[0,0,0],[.2,0,0],[0,.2,0],[0,0,.2]],
                faces=[[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    quad = dict(name='top', role='static', vertices=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
                faces=[[0,1,2],[0,2,3]], uv=[[0,0],[1,0],[1,1],[0,1]], normals=[[0,0,1]]*4)
    return {'scene':{'cameras':[cam], 'revisit_cameras':[cam], 'meshes':[mesh,quad], 'tags_enabled':False},
            'objects':{'target':{'kind':'mesh','meshes':['target_body']},
                       'occluder':{'kind':'box','size_m':[.1,.2,.3],
                                   'poses':{'parked':{'base':[1,2,3],'yaw_deg':30}, 'placed':{'base':[.5,.6,.7],'yaw_deg':-20}}}, 'distractors':[]}}


class SceneTest(unittest.TestCase):
    def test_compiled_camera_and_blocker(self):
        a = fixture()
        xml, _ = build_xml(a, '/tmp', '/tmp', '/tmp')
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        camid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'nominal_P1')
        bodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'blocker')
        c = a['scene']['cameras'][0]
        np.testing.assert_allclose(d.cam_xpos[camid], c['position'], atol=1e-12)
        np.testing.assert_allclose(d.cam_xmat[camid].reshape(3,3), np.array(camera_axes(c['forward'], c['up'])).T, atol=1e-12)
        np.testing.assert_allclose(d.xpos[bodyid], [1,2,3], atol=1e-12)
        self.assertEqual(m.nmesh, 2)

    def test_compiled_interventions_and_bounds(self):
        a = fixture()
        xml, _ = build_xml(a, '/tmp', '/tmp', '/tmp')
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        removed = apply_state(m, d, a, 'placed_removed')
        self.assertFalse(removed['state']['target_enabled'])
        np.testing.assert_allclose(removed['state']['blocker']['position'], [.5,.6,.7])
        np.testing.assert_allclose(removed['objects']['target']['bounds'], [[0,0,0],[.2,.2,.2]], atol=1e-7)
        np.testing.assert_allclose(removed['static_bounds']['top'], [[-1,-1,0],[1,1,0]], atol=1e-7)
        present = apply_state(m, d, a, 'clear_present')
        self.assertTrue(present['state']['target_enabled'])
        np.testing.assert_allclose(present['state']['blocker']['position'], [1,2,3])

    def test_invalid_texture_fails_closed(self):
        a = fixture()
        a['scene']['materials'] = {'bad':{'texture':{'file':'missing.png','sha256':'invalid'}}}
        with self.assertRaises(Exception):
            build_xml(a, '/tmp', '/tmp', '/tmp')

if __name__ == '__main__':
    unittest.main()
