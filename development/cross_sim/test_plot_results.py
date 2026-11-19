import copy
import json
import tempfile
import unittest
from pathlib import Path
from development.cross_sim import plot_results as P


def synthetic():
    data={'study_id':'cross-sim-dev-v1','synthetic':True,'rows':{}}
    for control in P.CONTROLS:
        rows=[]
        for engine in P.ENGINES:
            for env in P.ENVIRONMENTS:
                for level in P.LEVELS:
                    for n in range(5):
                        base=f'{env}{n}'
                        for stop in P.STOPS:
                            for state in P.STATES:
                                physical='PRESENT' if state.endswith('present') else 'REMOVED'
                                answer=('UNKNOWN' if state.startswith('placed') else physical)
                                if env=='shelf' and state=='clear_present' and control=='primary': answer='UNKNOWN'
                                ident='|'.join((engine,env,level,base,stop,state))
                                pair='|'.join((engine,env,level,base,stop,state.split('_')[0]))
                                rows.append({'id':ident,'engine':engine,'environment':env,'level':level,'base_id':base,'stop':stop,'state':state,'physical_state':physical,'answer':answer,'pair_id':pair})
        data['rows'][control]=rows
    return data


class PlotTests(unittest.TestCase):
    def test_counts_and_complete_cohort(self):
        data=synthetic(); rows=P.build_rows(data); self.assertEqual(len(rows),216)
        cell=next(r for r in rows if r['scope']=='aggregate' and r['control']=='primary' and r['engine']=='isaac' and r['environment']=='shelf' and r['level']=='D0')
        self.assertEqual((cell['denominator'],cell['definite'],cell['unknown'],cell['correct'],cell['physical_errors']),(100,25,75,25,0))
        self.assertEqual((cell['clear_present_correct'],cell['clear_present_denominator']),(0,25))
        self.assertEqual((cell['placed_both_unknown'],cell['placed_pair_denominator']),(25,25))
        broken=copy.deepcopy(data); broken['rows']['primary'].pop()
        with self.assertRaises(ValueError): P.build_rows(broken)

    def test_synthetic_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'synthetic.json'; source.write_text(json.dumps(synthetic()))
            with self.assertRaises(ValueError): P.export(source,Path(tmp)/'missing-flag')
            result=P.export(source,Path(tmp)/'synthetic-output',synthetic=True)
            self.assertEqual(result['csv_rows'],216); self.assertTrue(result['synthetic'])
            self.assertEqual(set(result['outputs_sha256']),{'arrangement_metrics.csv','clear_correct_counts.png'})
            self.assertEqual(result['analysis_sha256'],P.digest(source))


if __name__=='__main__': unittest.main()
