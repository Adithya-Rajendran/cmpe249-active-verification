import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from development.cross_sim import qualification_geometry as G


class OracleTests(unittest.TestCase):
    def camera(self):
        return {'position':[0,0,0],'forward':[0,0,1],'up':[0,1,0],'f_px':10,'width':20,'height':20,'name':'P1'}

    def test_projection_and_pixel_centers(self):
        camera=self.camera()
        p,z=G.project([[0,0,10],[1,1,10]],camera)
        np.testing.assert_allclose(p,[[10,10],[9,9]])
        self.assertEqual(z.tolist(),[10,10])
        camera.update(principal_point_px=[8,7],fy_px=20)
        p,_=G.project([[0,0,10],[1,1,10]],camera)
        np.testing.assert_allclose(p,[[8,7],[7,5]])
        camera=self.camera()
        mesh=G._mesh([[.5,.5,10],[-2.5,.5,10],[-2.5,-2.5,10],[.5,-2.5,10]],[[0,1,2],[0,2,3]])
        image=G.rasterize(mesh,camera)
        self.assertEqual(int(image.sum()),16)
        self.assertTrue(image[9,9]); self.assertFalse(image[8,9])
        with self.assertRaises(ValueError): G.project([[0,0,-1]],camera)

    def test_band_and_excess_only(self):
        expected=np.zeros((25,25),bool); expected[6:19,6:19]=True
        shifted=np.roll(expected,2,axis=1)
        self.assertTrue(G.compare_mask(shifted,expected)['pass'])
        shifted=np.roll(expected,3,axis=1)
        self.assertFalse(G.compare_mask(shifted,expected)['pass'])
        partial=expected.copy(); partial[8:17,8:17]=False
        self.assertGreater(G.compare_mask(partial,expected)['missing_outside_band'],0)
        self.assertTrue(G.compare_mask(partial,expected,excess_only=True)['pass'])
        self.assertFalse(G.compare_mask(expected,np.zeros_like(expected))['pass'])
        self.assertTrue(G.compare_mask(np.zeros_like(expected),np.zeros_like(expected))['pass'])

    def test_ply_and_world_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.ply'
            path.write_text('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1000 0 0\n1000 2000 0\n0 2000 0\n4 0 1 2 3\n')
            v,f=G.read_ply(path); self.assertEqual(f.tolist(),[[0,1,2],[0,2,3]])
            obj={'kind':'ycbv','ply':'model.ply','rest_matrix':np.eye(3).tolist(),'pose':{'base':[5,6,7],'yaw_deg':90}}
            mesh=G.object_mesh(obj,tmp)
            np.testing.assert_allclose(mesh['bounds'],[[4,5.5,7],[6,6.5,7]],atol=1e-12)
            binary=Path(tmp)/'binary.ply'; header=b'ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'
            binary.write_bytes(header+b''.join(struct.pack('<fff',*x) for x in [[0,0,0],[1,0,0],[0,1,0]])+struct.pack('<Biii',3,0,1,2))
            self.assertEqual(G.read_ply(binary)[1].tolist(),[[0,1,2]])

    def test_box_size_and_camera_selection(self):
        obj={'kind':'box','size_m':[2,4,6],'pose':{'base':[10,20,30],'yaw_deg':90}}
        np.testing.assert_allclose(G.object_mesh(obj,'.')['bounds'],[[8,19,30],[12,21,36]],atol=1e-12)
        a={'scene':{'cameras':[self.camera()],'revisit_cameras':[self.camera()]}}
        self.assertEqual(G.camera_record(a,'P1')['name'],'P1')
        with self.assertRaises(ValueError): G.camera_record(a,'missing')

    def test_source_geometry_static_names_and_surrogates(self):
        target={'kind':'mesh','meshes':['target_mesh']}
        box={'kind':'box','size_m':[1,1,1],'poses':{'placed':{'base':[0,0,1],'yaw_deg':0},'parked':{'base':[2,0,1],'yaw_deg':0}}}
        tri={'vertices':[[0,0,1],[1,0,1],[0,1,1]],'faces':[[0,1,2]]}
        arr={'objects':{'target':target,'occluder':box,'distractors':[]},'scene':{'boxes':[{'name':'room','role':'room','center':[0,0,0],'half':[1,1,1]}],
             'meshes':[dict(tri,name='target_mesh',role='target'),dict(tri,name='legacy_blocker',role='occluder'),dict(tri,name='floor_mesh',role='room')]}}
        meshes=G.source_geometry(arr,'.')
        self.assertEqual(set(meshes),{'target','blocker_placed','blocker_parked','room','floor_mesh'})


if __name__=='__main__': unittest.main()
