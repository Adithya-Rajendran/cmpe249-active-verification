import unittest
from metrics import summarize, validate_split, veto

class MetricsTests(unittest.TestCase):
    def row(self, **kw):
        return dict(arrangement='a', partition='calibration', physical='PRESENT', support='PRESENT', answer='UNKNOWN', failure=False, **kw)
    def test_all_disputed_and_empty_denominators(self):
        r=self.row(); r['support']='DISPUTED'
        m=summarize([r])
        self.assertIsNone(m['unsupported']['estimate'])
        self.assertIsNone(m['removal_resolution']['estimate'])
        self.assertEqual(summarize([r],True)['unsupported']['denominator'],1)
    def test_all_unknown_and_failures(self):
        a=self.row(); b=self.row(); b['failure']=True
        m=summarize([a,b])
        self.assertEqual(m['coverage']['estimate'],0)
        self.assertEqual(m['unknown']['estimate'],1)
        self.assertEqual(m['failure']['estimate'],0.5)
        self.assertIsNone(m['answered_error']['estimate'])
    def test_veto_cannot_create_or_swap(self):
        for base in ('PRESENT','REMOVED','UNKNOWN'):
            for p in (False,True):
                for r in (False,True):
                    self.assertIn(veto(base,p,r), {base,'UNKNOWN'})
    def test_split(self):
        a=self.row(); b=self.row(); b['partition']='pilot_evaluation'
        with self.assertRaises(ValueError): validate_split([a,b])

if __name__ == '__main__': unittest.main()
