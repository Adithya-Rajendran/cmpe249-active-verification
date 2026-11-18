import unittest
from analyze import report
from metrics import summarize

def row(arr,physical='PRESENT',support='PRESENT',answer='UNKNOWN',failure=False,phase='visible',stop='P1'):
    return dict(arrangement=arr,physical=physical,support=support,answer=answer,failure=failure,phase=phase,stop=stop)

class AnalysisTests(unittest.TestCase):
    def test_unequal_arrangement_weight(self):
        rows=[row('a',answer='PRESENT'),row('b',answer='UNKNOWN'),row('b',answer='UNKNOWN'),row('b',answer='UNKNOWN')]
        x=summarize(rows)['useful']
        self.assertEqual(x['estimate'],.5);self.assertEqual((x['numerator'],x['denominator']),(1,4))
    def test_failed_pair_is_not_both_unknown(self):
        a=row('a',phase='occluded'); b=row('a',physical='REMOVED',support='UNKNOWN',phase='removed',failure=True)
        p=report([a,b])['pairs']['all']
        self.assertEqual(p['both_unknown_numerator'],0);self.assertEqual(p['both_unknown_denominator'],1)

if __name__=='__main__': unittest.main()
