import unittest
from unittest.mock import patch
from cohort import IDS, validate

class CohortTest(unittest.TestCase):
    def test_synthetic_cohort_and_tampering(self):
        rows=[]
        for env,ids in IDS.items():
            for i,name in enumerate(ids):
                for stop in ('P1','P3','P5'):
                    for phase in ('visible','occluded','removed'):
                        rows.append({'id':len(rows),'environment':env,'arrangement':name,
                          'partition':'calibration' if i%2==0 else 'pilot_evaluation',
                          'stop':stop,'phase':phase,'support':'UNKNOWN','reviewer_type':'AI',
                          'reference':{},'query':{}})
        data={'study_id':'support-dev-pilot-v1','evidence_status':'SANITIZED_UNMARKED_V1','rows':rows}
        with patch('cohort.load_image'):
            validate(data,'unused')
            data['rows'][0]['id']=data['rows'][1]['id']
            with self.assertRaises(ValueError): validate(data,'unused')
if __name__=='__main__':unittest.main()
