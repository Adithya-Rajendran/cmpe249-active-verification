"""Synthetic capture integrity checks; no study images."""
import tempfile
import unittest
from pathlib import Path
from development.cross_sim.mujoco_capture import capture
from development.cross_sim.mujoco_scene import build_xml
from development.cross_sim.mujoco_test_scene import fixture


class CaptureTest(unittest.TestCase):
    def test_resume_and_tamper_refusal(self):
        a = fixture()
        xml, inv = build_xml(a, '/tmp', '/tmp', '/tmp')
        row = ('SYNTH', 'D0', a, xml, inv, {}, 'input')
        manifest = {'stops':['P1'], 'states':['clear_present','clear_removed','placed_present','placed_removed'], 'split':'qualification'}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(capture(row, manifest, {'hash':'one'}, out), 'captured')
            self.assertEqual(capture(row, manifest, {'hash':'one'}, out), 'resumed')
            with self.assertRaises(RuntimeError):
                capture(row, manifest, {'hash':'changed'}, out)
            (out/'SYNTH'/'D0'/'rgb'/'reference_P1.png').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                capture(row, manifest, {'hash':'one'}, out)

    def test_partial_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'SYNTH'/'D0.partial').mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                capture(('SYNTH','D0',None,None,None,None,None), {}, {}, out)

if __name__ == '__main__':
    unittest.main()
