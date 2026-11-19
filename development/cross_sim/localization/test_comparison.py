import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from development.cross_sim.localization.evidence import overlap
from development.cross_sim.localization.prepare import BASES, balanced_pairs, select
from development.cross_sim.localization.results import compare
from development.cross_sim.localization.reviews import broker, decision_instructions, validate_locations
from development.cross_sim.support_labels.prepare import save, sha


class ComparisonTests(unittest.TestCase):
    def test_two_arrangements_with_one_level_each_still_balance_all_cells(self):
        candid = {BASES[0]:{"partial":[("D0","P1"),("D0","P2")]},
                  BASES[1]:{"partial":[("D6","P1"),("D6","P2")]}}
        pairs = balanced_pairs("shelf",BASES[:2],candid)
        self.assertEqual(Counter((pos[0],i) for pair in pairs for i,pos in enumerate(pair)),
                         Counter({("D0",0):1,("D6",0):1,("D0",1):1,("D6",1):1}))

    def test_selection_preserves_shortages_and_keeps_unique_positions(self):
        def values(visible,blocked):
            m = {"clear":100,"alone":100,"placed":int(visible*100),"fraction_visible":visible,"fraction_blocker":blocked}
            return {"isaac":m,"mujoco":dict(m)}
        pool={("D6","P1"):values(.10,.90),("D0","P2"):values(0,1),("D6","P3"):values(1,0)}
        choices,_=select({b:pool for b in BASES})
        for records in choices.values():
            self.assertEqual(len(records),3)
            self.assertEqual({k for k,_,_ in records},set(pool))
            self.assertEqual({g for _,g,_ in records},{"partial","clear","hidden"})
        self.assertEqual(choices,select({b:pool for b in BASES})[0])

    def test_box_convention_and_missing_estimate(self):
        self.assertEqual(overlap([10,10,20,20],[20,10,30,20]),0.)
        self.assertAlmostEqual(overlap([1,2,5,6],[3,2,7,6]),1/3)
        self.assertIsNone(overlap(None,[1,1,10,10]))
        row=dict(target="cup",landmarks="table corner",uncertainty="width",box=None)
        validate_locations([row])
        with self.assertRaises(ValueError):validate_locations([{**row,"box":[1,1,1,2]}])

    def test_gains_losses_and_new_errors_count_paired_cases(self):
        a={"id":"first","arm":"original","condition":"partial","environment":"shelf","level":"D6",
           "physical_state":"REMOVED","corroborated_definite":False,"terra_wrong_definite":False,"terra_definite_on_exact":False}
        b={**a,"arm":"two_step","corroborated_definite":True}
        rows=[a,b,{**a,"id":"lost","corroborated_definite":True},{**b,"id":"lost","corroborated_definite":False},
              {**a,"id":"error","condition":"hidden","physical_state":"PRESENT"},
              {**b,"id":"error","condition":"hidden","physical_state":"PRESENT","corroborated_definite":False,"terra_wrong_definite":True,"terra_definite_on_exact":True}]
        result=compare(rows)
        self.assertEqual(result["net_corroborated_partial_removals"],0)
        self.assertEqual(result["new_corroborated_partial_removals"],["first"])
        self.assertEqual(result["lost_corroborated_partial_removals"],["lost"])
        self.assertEqual(result["new_wrong_definite"],["error"])
        self.assertEqual(result["new_definite_on_exact"],["error"])
        self.assertFalse(result["complete_balance"])

    def test_broker_copies_estimates_before_releasing_decision_instructions(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            prep=root/"prepared";prep.mkdir()
            packet=root/"p-fixture";packet.mkdir()
            save(packet/"cases.json",[{"id":"q"}])
            save(prep/"release.json",{"packets":[{"arm":"two_step","path":str(packet),"sha256":sha(packet/"cases.json")}]})
            save(prep/"blind-audit.json",{"release_sha256":sha(prep/"release.json")})
            record={"id":"q","target":"tin","landmarks":"shelf and corner","box":[10,20,30,40],"uncertainty":"edge"}
            save(packet/"localization.json",[record])
            broker(root)
            locked=root/"sealed"/"locations"/"p-fixture.json"
            self.assertEqual(sha(locked),sha(packet/"localization.json"))
            self.assertEqual((packet/"Decision.md").read_text(),decision_instructions())
            save(packet/"localization.json",[{**record,"box":None}])
            with self.assertRaises(ValueError):broker(root)


if __name__=="__main__":
    unittest.main()
