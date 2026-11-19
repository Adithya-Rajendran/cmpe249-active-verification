import unittest
import numpy as np
from development.cross_sim import qualification_visibility as V


def quad(z, x0=-1, x1=1, y0=-1, y1=1):
    return {'vertices':np.array([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]],float),'faces':np.array([[0,1,2],[0,2,3]])}


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.camera={'position':[0,0,0],'forward':[0,0,1],'up':[0,1,0],'width':40,'height':40,'f_px':20}

    def test_front_back_and_ties(self):
        subject=quad(4); base=V.visible_mask(subject,[],self.camera)
        behind=V.visible_mask(subject,[quad(6)],self.camera)
        front=V.visible_mask(subject,[quad(2)],self.camera)
        tied=V.visible_mask(subject,[quad(4)],self.camera)
        self.assertGreater(base['visible_pixels'],0)
        self.assertEqual(base['visible_pixels'],behind['visible_pixels'])
        self.assertEqual(base['visible_pixels'],tied['visible_pixels'])
        self.assertTrue(front['fully_occluded']); self.assertEqual(front['visible_pixels'],0)
        self.assertEqual(front['occluded_pixels'],base['isolated_pixels'])

    def test_partial_foreground_and_offscreen(self):
        subject=quad(4); result=V.visible_mask(subject,[quad(2,0,1)],self.camera)
        self.assertGreater(result['visible_pixels'],0); self.assertGreater(result['occluded_pixels'],0)
        self.assertEqual(result['visible_pixels']+result['occluded_pixels'],result['isolated_pixels'])
        outside=V.visible_mask(quad(4,100,101),[],self.camera)
        self.assertFalse(outside['fully_occluded']); self.assertEqual(outside['isolated_pixels'],0)

    def test_near_clipping_and_perspective_depth(self):
        # A triangle crossing the camera plane remains rasterizable after clipping.
        mesh={'vertices':np.array([[-.5,-.5,-1],[.5,-.5,2],[0,.5,2]]),'faces':np.array([[0,1,2]])}
        result=V.visible_mask(mesh,[],self.camera,near_m=.1)
        self.assertGreater(result['isolated_pixels'],0)
        self.assertEqual(result['visible_pixels'],result['isolated_pixels'])
        all_behind={'vertices':mesh['vertices']-np.array([0,0,4]),'faces':mesh['faces']}
        self.assertEqual(V.visible_mask(all_behind,[],self.camera)['isolated_pixels'],0)
        with self.assertRaises(ValueError): V.visible_mask(mesh,[],self.camera,near_m=0)


if __name__=='__main__': unittest.main()
