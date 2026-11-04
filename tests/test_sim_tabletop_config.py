"""The tabletop study config's provenance (data/raw/2026-11-04_tabletop_scan/make_tabletop_config.py final) and, once written,
configs/sim/arrange_tabletop_v1.json itself.

Pins: the board-height scan's committed outcome stays "no height satisfies (a) and (b)"; `final` refuses a decision without a
height unless the post-hoc development amendment (amendment_2026-11-04.json) names that decision by sha256, repeats its
outcome verbatim and gives a height and width; the amendment records H_T 0.18 m, IND28 0.36 x 0.18 x 0.005 m, parking >= 1
stop and the four kept ycb_box choices. numpy only (the module imports sim/arrange.py lazily).
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN = REPO / "data" / "raw" / "2026-11-04_tabletop_scan"
DECISION = SCAN / "decision.json"
AMENDMENT = SCAN / "amendment_2026-11-04.json"
MATRIX = SCAN / "matrix.json"
CONFIG = REPO / "configs" / "sim" / "arrange_tabletop_v1.json"
SLOW = os.environ.get("CMPE249_GOLDEN_SLOW") == "1"


def _tool():
    if not hasattr(_tool, "m"):
        spec = importlib.util.spec_from_file_location("make_tabletop_config", SCAN / "make_tabletop_config.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _tool.m = m
    return _tool.m


class TestAmendmentRecord(unittest.TestCase):
    def test_the_original_outcome_is_kept(self):
        dec = json.loads(DECISION.read_text())
        self.assertEqual(dec["outcome"], "no height satisfies (a) and (b)")
        self.assertIsNone(dec["H_T_m"])
        self.assertIsNone(dec["ind28_width_m"])
        am = json.loads(AMENDMENT.read_text())
        self.assertEqual(am["original"]["outcome"], dec["outcome"])
        self.assertEqual(am["original"]["decision"]["sha256"], _tool()._sha256(DECISION))
        for k in ("plan", "results", "cells", "base_config"):
            self.assertEqual(am["original"][k]["sha256"], _tool()._sha256(REPO / am["original"][k]["file"]), k)

    def test_the_amendment_says_what_it_is(self):
        am = json.loads(AMENDMENT.read_text())
        self.assertEqual(am["kind"], "POST-HOC DEVELOPMENT AMENDMENT")
        self.assertEqual(am["made_by"], "the researcher")
        new = am["amendment"]
        self.assertEqual((new["H_T_m"], new["ind28_width_m"], new["parking_min_stops_in_frame"]), (0.18, 0.36, 1))
        self.assertEqual(new["ycb_box_choices"], [[2, "upright"], [3, "upright"], [16, "upright"], [7, "edge"]])
        self.assertEqual(len(am["statistics_disclosure"]), 4)
        self.assertEqual(set(am["evidence"]["files"]), {"audit.json", "ind_design_margin.json", "ind_design_revisit.json"})


class TestHeightSource(unittest.TestCase):
    def test_no_height_and_no_amendment_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            _tool().height_source(DECISION)
        self.assertIn("refused", str(cm.exception))

    def test_the_amendment_is_accepted(self):
        src = _tool().height_source(DECISION, AMENDMENT)
        self.assertEqual((src["H"], src["W"]), (0.18, 0.36))
        self.assertEqual(src["kept"], [[2, "upright"], [3, "upright"], [16, "upright"], [7, "edge"]])
        self.assertEqual(src["amendment"]["sha256"], _tool()._sha256(AMENDMENT))
        self.assertIn("POST-HOC DEVELOPMENT AMENDMENT", src["height_note"])
        self.assertIn("no height satisfies (a) and (b)", src["height_note"])

    def _refused(self, edit_amendment=None, edit_decision=None):
        am = json.loads(AMENDMENT.read_text())
        dec = json.loads(DECISION.read_text())
        if edit_decision:
            edit_decision(dec)
        with tempfile.TemporaryDirectory() as t:
            dp = Path(t) / "decision.json"
            dp.write_text(DECISION.read_text() if edit_decision is None else json.dumps(dec))
            if edit_amendment:
                edit_amendment(am)
            ap = Path(t) / "amendment.json"
            ap.write_text(json.dumps(am))
            with self.assertRaises(SystemExit) as cm:
                _tool().height_source(dp, ap)
        self.assertIn("refused", str(cm.exception))

    def test_an_amendment_without_a_height_is_refused(self):
        self._refused(lambda am: am["amendment"].update(H_T_m=None))
        self._refused(lambda am: am["amendment"].pop("ind28_width_m"))

    def test_an_amendment_of_another_decision_is_refused(self):
        self._refused(lambda am: am["original"]["decision"].update(sha256="0" * 64))
        self._refused(edit_decision=lambda d: d.update(results_n_cells=174))         # the file's sha256 changes

    def test_an_amendment_with_another_outcome_is_refused(self):
        self._refused(lambda am: am["original"].update(outcome="H_T = 0.18 m"))

    def test_other_files_are_refused(self):
        self._refused(lambda am: am.update(schema="something.else"))

    def test_an_amendment_of_a_decision_that_picked_a_height_is_refused(self):
        self._refused(edit_decision=lambda d: d.update(H_T_m=0.2, ind28_width_m=0.28))

    def test_the_kept_ycb_choices_and_parking_rule_must_match(self):
        self._refused(lambda am: am["amendment"].update(ycb_box_choices=[[2, "upright"]]))
        self._refused(lambda am: am["amendment"].update(parking_min_stops_in_frame=5))


class TestStudyConfig(unittest.TestCase):
    """configs/sim/arrange_tabletop_v1.json as written by make_tabletop_config.py final --amendment."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(CONFIG.read_text())

    def test_status_is_dev_and_cites_the_amendment(self):
        st = self.cfg["status"]
        self.assertTrue(st.startswith("DEV config"))
        self.assertIn("NOT FROZEN", st)
        self.assertNotIn("frozen config", st.lower())
        self.assertIn("POST-HOC DEVELOPMENT AMENDMENT", st)
        self.assertIn("amendment_2026-11-04.json", st)
        self.assertIn("no height satisfies (a) and (b)", st)
        self.assertEqual(self.cfg["height_amendment"]["sha256"], _tool()._sha256(AMENDMENT))

    def test_boards(self):
        fam = self.cfg["occluder"]["families"]
        self.assertEqual([fam[f]["height_m"] for f in ("S8", "M16", "IND28")], [0.18] * 3)
        self.assertEqual((fam["IND28"]["width_m"], fam["IND28"]["thickness_m"]), (0.36, 0.005))
        self.assertIn("TABLETOP VARIANT", fam["IND28"]["variant"])
        self.assertEqual(fam["ycb_box"]["choices"], [[2, "upright"], [3, "upright"], [16, "upright"], [7, "edge"]])
        self.assertEqual(self.cfg["occluder"]["parked"]["min_stops_in_frame"], 1)

    def test_target_lists_are_the_class_matrix_rule(self):
        mx = json.loads(MATRIX.read_text())
        self.assertEqual((mx["tries_per_cell"], mx["probe_seeds"]), (100, [7200, 7299]))
        ta = self.cfg["slots"]["target_assignment"]
        want, excl = {}, {}
        for r in mx["rows"]:
            (want if r["ok"] >= 1 else excl).setdefault(r["class"], set()).add(int(r["target"]))
        self.assertEqual({k: set(v) for k, v in ta["classes"].items()}, want)
        self.assertEqual({k: {int(t) for t in v} for k, v in ta["excluded"].items()}, excl)

    def test_everything_else_is_the_base_config(self):
        base = json.loads((SCAN / "base_config.json").read_text())
        c = json.loads(CONFIG.read_text())
        for k in ("seeds", "ids", "objects", "rest_poses", "clearances_m", "target", "distractors", "revisit", "materials", "scene"):
            self.assertEqual(c[k], base[k], k)
        self.assertEqual({k: v for k, v in c["slots"].items() if k != "target_assignment"},
                         {k: v for k, v in base["slots"].items() if k != "target_assignment"})

    def test_it_loads_and_the_design_cases_build(self):
        from sim import arrange as A
        ctx = A.load_context("configs/sim/arrange_tabletop_v1.json")
        ids = [A.design_arrangement(i, ctx=ctx)["arrangement_id"] for i in range(4)]
        self.assertEqual(ids, ["TDES0", "TDES1", "TDES2", "TDES3"])
        tdes3 = self.cfg["design_cases"]["cases"][3]
        self.assertEqual((tdes3["gap_m"], tdes3["shift_m"], tdes3["margin_px"]), (0.005, -0.02, 3.435))

    @unittest.skipUnless(SLOW, "the IND28 margin scan takes a few seconds: set CMPE249_GOLDEN_SLOW=1")
    def test_the_ind28_design_case_is_the_margin_scan_result(self):
        from sim import arrange as A
        ctx = A.load_context("configs/sim/arrange_tabletop_v1.json")
        case = self.cfg["design_cases"]["cases"][3]
        best = A.design_margin_scan(ctx, "IND28", "HHHHH")
        self.assertEqual((best["gap_m"], best["shift_m"], best["margin_px"], best["stop_margins_px"]),
                         (case["gap_m"], case["shift_m"], case["margin_px"], case["stop_margins_px"]))


if __name__ == "__main__":
    unittest.main()
