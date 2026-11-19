import copy
import unittest
import numpy as np
from development.cross_sim import paired_intervals as P


def fixture():
    result={'study_id':'cross-sim-dev-v1','metrics':{}}
    for control in P.CONTROLS:
        records=[]
        for engine in P.ENGINES:
            for env in P.ENVIRONMENTS:
                for level in P.LEVELS:
                    for i in range(5):
                        fraction=lambda n,d:{'numerator':n,'denominator':d,'rate':n/d}
                        inc=1 if engine=='mujoco' else 0
                        guard=1 if control=='relaxed_guard' else 0
                        records.append({'engine':engine,'environment':env,'level':level,'arrangement':f'{env}{i}',
                           'coverage':fraction(10+inc,20),'correct_yield':fraction(8+guard+(i if engine=='mujoco' else 0),20),
                           'physical_errors':inc,'clear_resolution':{'clear_present':fraction(3,5),'clear_removed':fraction(4,5)},
                           'placed_both_unknown':fraction(1,5)})
        result['metrics'][control]=records
    return result


class BootstrapTests(unittest.TestCase):
    def test_stream_and_paired_estimates(self):
        data=fixture(); result=P.compute(data); self.assertEqual(result,P.compute(data))
        case=next(x for x in result['engine_differences'] if x['control']=='primary' and x['environment']=='shelf' and x['level']=='D0' and x['metric']=='coverage')
        self.assertAlmostEqual(case['estimate'],.05); np.testing.assert_allclose(case['interval_95_percentile'],[.05,.05])
        case=next(x for x in result['engine_differences'] if x['control']=='primary' and x['environment']=='shelf' and x['level']=='D0' and x['metric']=='correct_yield')
        rng=np.random.default_rng(20261117); shelf=rng.integers(0,5,size=(2000,5)); tabletop=rng.integers(0,5,size=(2000,5))
        expected=np.percentile(np.mean((np.arange(5)/20)[shelf],axis=1),[2.5,97.5],method='linear')
        self.assertAlmostEqual(case['estimate'],.1); np.testing.assert_allclose(case['interval_95_percentile'],expected)
        strata=P.bootstrap_indices({'shelf':list(range(5)),'tabletop':list(range(5))})
        np.testing.assert_array_equal(strata['shelf'],shelf); np.testing.assert_array_equal(strata['tabletop'],tabletop)
        guard=next(x for x in result['guard_correct_yield_differences'] if x['engine']=='combined' and x['environment']=='balanced_combined' and x['level']=='D0')
        self.assertEqual(guard['arrangements'],10); self.assertAlmostEqual(guard['estimate'],.05); np.testing.assert_allclose(guard['interval_95_percentile'],[.05,.05])
        for overall in result['guard_correct_yield_all_levels']:
            self.assertAlmostEqual(overall['estimate'],.05); np.testing.assert_allclose(overall['interval_95_percentile'],[.05,.05])
        physical=next(x for x in result['engine_differences'] if x['control']=='primary' and x['environment']=='shelf' and x['level']=='D0' and x['metric']=='physical_error_rate')
        self.assertAlmostEqual(physical['estimate'],.05)

    def test_incomplete_or_invalid_refused(self):
        data=fixture(); data['metrics']['primary'].pop()
        with self.assertRaises(ValueError): P.compute(data)
        data=fixture(); data['metrics']['primary'][0]['coverage']['denominator']=19
        with self.assertRaises(ValueError): P.compute(data)
        with self.assertRaises(ValueError): P.compute(fixture(),seed=5)


if __name__=='__main__': unittest.main()
