import copy
import json
import unittest
from development.cross_sim import support_label_cohort as S


class ProtocolTests(unittest.TestCase):
    def test_fixed_disjoint_reservations_and_rejections(self):
        p=json.loads(S.PROTOCOL.read_text());S.validate(p,set())
        with self.assertRaises(ValueError):S.validate(p,{9500})
        bad=copy.deepcopy(p);bad['arrangements'][0]['seed_start']=100000
        with self.assertRaises(ValueError):S.validate(bad,set())
        bad=copy.deepcopy(p);bad['study_id']='test'
        with self.assertRaises(ValueError):S.validate(bad,set())


if __name__=='__main__':unittest.main()
