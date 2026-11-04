"""The tabletop study config's provenance (data/raw/2026-11-04_tabletop_scan/make_tabletop_config.py final) and, once written,
configs/sim/arrange_tabletop_v1.json itself.

Pins: the board-height scan's committed outcome stays "no height satisfies (a) and (b)"; `final` refuses a decision without a
height unless the post-hoc development amendment (amendment_2026-11-04.json) names that decision by sha256, repeats its
outcome verbatim and gives a height and width; the amendment records H_T 0.18 m, IND28 0.36 x 0.18 x 0.005 m, parking >= 1
stop and the four kept ycb_box choices. numpy only (the module imports sim/arrange.py lazily).
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN = REPO / "data" / "raw" / "2026-11-04_tabletop_scan"
DECISION = SCAN / "decision.json"
AMENDMENT = SCAN / "amendment_2026-11-04.json"


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


if __name__ == "__main__":
    unittest.main()
