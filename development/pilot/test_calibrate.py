import unittest
from calibrate import select, candidate, GRID

class CalibrationTests(unittest.TestCase):
    def row(self, arrangement, physical, support, base, match, obstruction):
        return {'arrangement':arrangement,'partition':'calibration','physical':physical,'support':support,'base':base,
                'confidence':0.8,'valid':True,'quality':1.0,'match':match,
                'obstruction':obstruction,'failure':False}
    def test_grid_and_same_cache(self):
        self.assertTrue(all(a < p for q,p,a,o in GRID))
        rows=[self.row('a','PRESENT','PRESENT','PRESENT',.95,.1),
              self.row('b','REMOVED','REMOVED','REMOVED',.1,.1)]
        out=candidate(rows,'full',GRID[0])
        self.assertEqual([r['answer'] for r in out], ['PRESENT','REMOVED'])
        self.assertEqual([r['base'] for r in out], [r['base'] for r in rows])
        self.assertIsNotNone(select(rows,'full')['selected_index'])
    def test_missing_class_and_all_disputed_cannot_calibrate(self):
        r=self.row('a','PRESENT','DISPUTED','PRESENT',.95,.1)
        result=select([r],'confidence')
        self.assertIsNone(result['selected_index'])
        self.assertFalse(any(c['eligible'] for c in result['candidates']))
    def test_calibration_rejects_evaluation(self):
        r=self.row('a','PRESENT','PRESENT','PRESENT',.95,.1);r['partition']='pilot_evaluation'
        with self.assertRaises(ValueError): select([r],'full')
    def test_image_arm_independent_of_model_failure(self):
        r=self.row('a','PRESENT','PRESENT','PRESENT',.95,.1);r['failure']=True
        out=candidate([r],'image',GRID[0])[0]
        self.assertFalse(out['failure']);self.assertEqual(out['answer'],'PRESENT')
    def test_failure_not_correct_abstention(self):
        r=self.row('a','PRESENT','PRESENT','PRESENT',.95,.1);r['failure']=True
        out=candidate([r],'full',GRID[0])[0]
        self.assertEqual(out['answer'],'UNKNOWN');self.assertTrue(out['failure'])

if __name__ == '__main__': unittest.main()
