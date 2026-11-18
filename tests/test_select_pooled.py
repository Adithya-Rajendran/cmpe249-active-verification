"""Tabletop wave, step 9: verify/select.py's pooled calibration over both environments' dev sets. The sets are
tests/test_select_sim.py's hand-built sim sets (invented scores; no number here is a threshold); the tabletop one carries
"environment": "tabletop" in its manifest rows and the tabletop positions file.

Pinned: one set still runs the one-environment path (its byte-identity with the code before the tabletop was checked
against a 1782efa export, not here); pooled data keep every record's environment, calibrate one threshold set on all
views, replay each episode on its environment's positions with the cap in that environment's arc steps (the same
decisions as the set run alone), and report a per-environment breakdown whose counts add up to the pooled rows; bad
pools are refused. DECISIONS 2026-11-04 (tabletop pre-render choices 2 and 3): rows pooled over environments carry the cap
in arc steps only (no travel_cap_cm, no mean_travel_cm, no dev_check.travel_caps_cm), the per-environment rows and the
per-episode records carry their own environment's cm; FRR stays a reported policy check with the pooled all_within_limit
as the headline, and frr_within_limit_by_environment / all_environments_within_limit sit next to it, with report text
saying a pooled pass does not imply each environment passed. The single-set byte-identity with 4fd65d7 was re-checked
by running the CLI from both trees (feasible, infeasible exit 3, --synthetic exit 4), not here.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_select_sim import CTX, FAST, QUICK, SLOTS, write_sim_set
from verify import select as sel
from verify.schema import REPO_ROOT, Budget, Lighting, Positions, cap_cm_from_steps

TAB_POS = REPO_ROOT / "configs" / "sim" / "positions_tabletop_A.json"
SHELF_POS = REPO_ROOT / "configs" / "sim" / "positions_shelf_A.json"
TSLOTS = (("TD5000", "S8", None), ("TD5001", "M16", "L"), ("TD5002", "M16", "R"), ("TD5003", "IND28", None),
          ("TD5004", "white_box", None), ("TD5005", "S8", None))


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def write_tabletop_set(d: Path, slots=TSLOTS, environment: str | None = "tabletop", **kw) -> list[str]:
    args = write_sim_set(d, slots, **kw)
    man = json.loads((d / "manifest.json").read_text())
    for row in man["arrangements"]:
        if environment is not None:
            row["environment"] = environment
    (d / "manifest.json").write_text(json.dumps(man, indent=2))
    (d / "positions.json").write_bytes(TAB_POS.read_bytes())
    return args


def set_files(d: Path) -> tuple[Path, Path, Path, Path]:
    return d / "manifest.json", d / "evidence.jsonl", d / "view_labels.json", d / "positions.json"


class PooledInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        write_sim_set(cls.d / "shelf")
        write_tabletop_set(cls.d / "tab")
        cls.dev = sel.sim_dev_data_pooled([set_files(cls.d / "shelf"), set_files(cls.d / "tab")])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_records_keep_their_environment_and_positions(self) -> None:
        dev = self.dev
        self.assertTrue(dev.pooled)
        self.assertEqual(dev.environments, ("shelf", "tabletop"))
        self.assertEqual(dev.reference_environment, "shelf")
        self.assertEqual(dev.positions, Positions.load(SHELF_POS))
        self.assertEqual(dev.environment_arrangements("tabletop"), tuple(a for a, _, _ in TSLOTS))
        self.assertEqual(dev.positions_of("TD5001"), Positions.load(TAB_POS))
        self.assertEqual(dev.positions_of("SD0001"), Positions.load(SHELF_POS))
        self.assertEqual(len(dev.specs), 6 * (len(SLOTS) + len(TSLOTS)))
        self.assertEqual((dev.strata["SD0001"], dev.strata["TD5001"], dev.strata["TD5000"]), ("M16-L", "tabletop:M16-L", "tabletop:S8"))
        self.assertEqual({k.split("/")[0] for k in dev.inputs}, {"shelf", "tabletop"})
        self.assertIn("tabletop/manifest", sel.data_provenance(dev)["inputs"])

    def test_one_set_is_not_pooled(self) -> None:
        one = sel.sim_dev_data(*set_files(self.d / "shelf"))
        self.assertFalse(one.pooled)
        self.assertEqual((one.environments, one.environment_of, one.reference_environment), ((), None, None))
        self.assertIs(one.positions_of("SD0001"), one.positions)

    def test_subset_keeps_the_environments(self) -> None:
        sub = self.dev.subset(["TD5000", "SD0000"])
        self.assertEqual(sub.environment_of, {"SD0000": "shelf", "TD5000": "tabletop"})
        self.assertEqual(sub.positions_of("TD5000"), Positions.load(TAB_POS))
        held = self.dev.subset(["TD5003"])
        self.assertEqual((held.environments, held.positions), (("tabletop",), self.dev.positions))

    def test_cap_steps_convert_per_environment(self) -> None:
        shelf, tab = Positions.load(SHELF_POS), Positions.load(TAB_POS)
        for steps in (1.0, 2.0, 1.5, 6.0):
            b = Budget(3, cap_cm_from_steps(steps, shelf))
            self.assertEqual(sel.environment_budget(b, shelf, tab), Budget(3, cap_cm_from_steps(steps, tab)))
            self.assertIs(sel.environment_budget(b, shelf, shelf), b)
        self.assertIs(sel.environment_budget(Budget(3), shelf, tab).travel_cap_cm, None)

    def test_pooled_runs_equal_each_set_alone(self) -> None:
        """An episode replays on its environment's positions with the cap in its arc steps: the same results as its set
        run alone, for every policy and cap."""
        th = sel.frozen_thresholds(0.5, 0.5, 0.3)
        pooled = sel.PolicyRunner(self.dev, FAST, CTX)
        for name in ("shelf", "tab"):
            alone_dev = sel.sim_dev_data(*set_files(self.d / name))
            alone = sel.PolicyRunner(alone_dev, FAST, CTX)
            for p in ("F", "G", "R"):
                for cap in FAST.cap_settings:
                    got = {r.episode_id: r for r in pooled.results(p, th, sel.run_budget(p, cap, FAST, self.dev.positions))}
                    for r in alone.results(p, th, sel.run_budget(p, cap, FAST, alone_dev.positions)):
                        with self.subTest(set=name, policy=p, cap=cap, episode=r.episode_id):
                            self.assertEqual((got[r.episode_id].mean_travel_cm, got[r.episode_id].decision_frac,
                                              got[r.episode_id].mean_views),
                                             (r.mean_travel_cm, r.decision_frac, r.mean_views))
        # and the environments differ: the first F move is 45 deg x 126.8 cm on the shelf, 60 deg x 45 cm on the table
        self.assertNotEqual(Positions.load(SHELF_POS).travel_cm(2, 0), Positions.load(TAB_POS).travel_cm(2, 0))

    def test_bad_pools_are_refused(self) -> None:
        d = self.d
        with self.assertRaisesRegex(ValueError, "at least two"):
            sel.sim_dev_data_pooled([set_files(d / "shelf")])
        write_sim_set(d / "shelf2", (("SD0100", "S8", None), ("SD0101", "S8", None)))
        with self.assertRaisesRegex(ValueError, "same environment"):
            sel.sim_dev_data_pooled([set_files(d / "shelf"), set_files(d / "shelf2")])
        write_tabletop_set(d / "tab_dup", (("SD0000", "S8", None), ("TD5100", "S8", None)))
        with self.assertRaisesRegex(ValueError, "SD0000 is in the shelf and the tabletop"):
            sel.sim_dev_data_pooled([set_files(d / "shelf"), set_files(d / "tab_dup")])
        write_tabletop_set(d / "mixed")
        man = json.loads((d / "mixed" / "manifest.json").read_text())
        del man["arrangements"][0]["environment"]
        (d / "mixed" / "manifest.json").write_text(json.dumps(man))
        with self.assertRaisesRegex(ValueError, "one environment"):
            sel.sim_dev_data_pooled([set_files(d / "shelf"), set_files(d / "mixed")])
        write_tabletop_set(d / "notsim", simulation=False)
        with self.assertRaisesRegex(ValueError, "simulation flag"):
            sel.sim_dev_data_pooled([set_files(d / "shelf"), set_files(d / "notsim")])


class PooledSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        a = write_sim_set(d / "shelf")
        b = write_tabletop_set(d / "tab")
        cls.args = [*a, *b]
        cls.dev = sel.sim_dev_data_pooled([set_files(d / "shelf"), set_files(d / "tab")])
        with quiet():
            cls.res = sel.run_selection(cls.dev, FAST, CTX)
            cls.rc = [sel.main([*cls.args, *QUICK, "--out", str(d / name)]) for name in ("a", "b")]
        cls.out = d / "a"
        with quiet():                                    # FAST has a cap (2 steps): the cm labels are exercised
            sel.write_outputs(d / "fast", cls.dev, FAST, cls.res)
        cls.fast_doc = json.loads((d / "fast" / "frozen_candidate.json").read_text())
        cls.fast_rep = (d / "fast" / sel.REPORT_FILE).read_text()
        cls.fast_details = json.loads((d / "fast" / "selection_details.json").read_text())
        cls.doc = json.loads((cls.out / "frozen_candidate.json").read_text())
        cls.rep = (cls.out / sel.REPORT_FILE).read_text()
        cls.details = json.loads((cls.out / "selection_details.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_one_threshold_set_from_all_views(self) -> None:
        rows = sel.view_rows(self.dev)
        self.assertEqual(len(rows), 30 * (len(SLOTS) + len(TSLOTS)))
        self.assertEqual(self.res["calibration"].arrangements, self.dev.arrangements)
        self.assertEqual(self.res["thresholds"], sel.calibrate(rows, FAST).thresholds)
        self.assertEqual(len(self.res["folds"]), len(SLOTS) + len(TSLOTS))       # leave one out across the pool

    def test_breakdown_adds_up_to_the_pooled_numbers(self) -> None:
        by = self.res["by_environment"]
        self.assertEqual(set(by), {"shelf", "tabletop"})
        for k in ("view_rates", "cv_held_out"):
            for light in (*sel.REPORT_LIGHTINGS, sel.POOLED):
                self.assertEqual(by["shelf"][k][light] + by["tabletop"][k][light], self.res[k][light])
        pooled = {(r.policy, r.cap_steps, r.lighting): r.counts for r in self.res["check"]}
        for key, c in pooled.items():
            parts = [next(r.counts for r in by[e]["check"] if (r.policy, r.cap_steps, r.lighting) == key) for e in by]
            self.assertEqual(sum(p.n_episodes for p in parts), c.n_episodes)
            self.assertAlmostEqual(sum(p.n_false_removal for p in parts), c.n_false_removal)
            self.assertAlmostEqual(sum(p.total_travel_cm for p in parts), c.total_travel_cm)
        ho = self.res["held_out_check"]
        for key in {(r.policy, r.cap_steps, r.lighting) for r in ho["rows"]}:
            c = next(r.counts for r in ho["rows"] if (r.policy, r.cap_steps, r.lighting) == key)
            parts = [next(r.counts for r in ho["by_environment"][e]["rows"] if (r.policy, r.cap_steps, r.lighting) == key)
                     for e in by]
            self.assertEqual(sum(p.n_episodes for p in parts), c.n_episodes)
        # a capped row is labelled with its environment's cm
        tab_row = next(r for r in by["tabletop"]["check"] if r.policy == "F" and r.cap_steps == 2.0)
        self.assertAlmostEqual(tab_row.travel_cap_cm, cap_cm_from_steps(2.0, Positions.load(TAB_POS)))
        self.assertEqual(by["tabletop"]["stratification"]["strata"], {"tabletop:IND28": 1, "tabletop:M16-L": 1,
                                                                      "tabletop:M16-R": 1, "tabletop:S8": 2,
                                                                      "tabletop:white_box": 1})

    def test_candidate_report_and_details(self) -> None:
        self.assertEqual(self.rc[0], self.rc[1])
        for name in ("frozen_candidate.json", sel.REPORT_FILE, "selection_details.json"):
            self.assertEqual((self.out / name).read_bytes(), (self.out.parent / "b" / name).read_bytes(), name)
        env = self.doc["environments"]
        self.assertEqual((env["pooled"], env["reference_environment"]), (["shelf", "tabletop"], "shelf"))
        self.assertEqual(env["environment_of"]["TD5004"], "tabletop")
        self.assertEqual(set(env["breakdown"]), {"shelf", "tabletop"})
        b = env["breakdown"]["tabletop"]
        self.assertEqual(b["positions"]["radius_cm"], 45.0)
        for k in ("per_view_rates", "held_out_per_view_rates", "gate_pass_by_lighting", "no_diagnostic_view_share",
                  "dev_check", "held_out_check", "lam_grid_check", "bootstrap_stratification", "travel_caps_cm"):
            self.assertIn(k, b)
        self.assertIn("all_within_limit", b["dev_check"])
        self.assertIn("## Per environment (pooled calibration)", self.rep)
        self.assertIn("### tabletop: 6 arrangements", self.rep)
        self.assertEqual({e.get("environment") for e in self.details["check_episodes"]}, {"shelf", "tabletop"})
        self.assertEqual({e.get("environment") for e in self.details["held_out_episodes"]}, {"shelf", "tabletop"})
        self.assertIn("tabletop/evidence", self.doc["inputs"])

    def test_pooled_rows_carry_cap_steps_only(self) -> None:
        doc, tab = self.fast_doc, Positions.load(TAB_POS)
        self.assertNotIn("travel_caps_cm", doc["dev_check"])
        pooled_rows = [*doc["dev_check"]["rows"], *doc["held_out_check"]["rows"]]
        self.assertTrue(any(r["cap_steps"] == 2.0 for r in pooled_rows))
        for r in pooled_rows:
            self.assertIn("cap_steps", r)
            self.assertNotIn("travel_cap_cm", r)
            self.assertNotIn("mean_travel_cm", r)
        self.assertTrue(all(not r.travel_in_cm for r in (*self.res["check"], *self.res["held_out_check"]["rows"])))
        # per environment: travel and each cap's cm of that environment's geometry
        for env, pos in (("shelf", Positions.load(SHELF_POS)), ("tabletop", tab)):
            b = doc["environments"]["breakdown"][env]
            self.assertEqual(b["travel_caps_cm"], [None, cap_cm_from_steps(2.0, pos)])
            for r in (*b["dev_check"]["rows"], *b["held_out_check"]["rows"]):
                self.assertIn("mean_travel_cm", r)
                if r["cap_steps"] == 2.0:
                    self.assertAlmostEqual(r["travel_cap_cm"], cap_cm_from_steps(2.0, pos))
        # per-episode records: the episode's own environment's cm
        for e in (*self.fast_details["check_episodes"], *self.fast_details["held_out_episodes"]):
            if e["cap_steps"] == 2.0:
                pos = tab if e["environment"] == "tabletop" else Positions.load(SHELF_POS)
                self.assertAlmostEqual(e["travel_cap_cm"], cap_cm_from_steps(2.0, pos))
        # the report: pooled rows "2 steps", per-environment rows with cm
        self.assertIn("| F | 2 steps | ", self.fast_rep)
        self.assertIn(f"| F | 2 steps ({cap_cm_from_steps(2.0, tab):.1f} cm) | ", self.fast_rep)
        self.assertIn("| per environment |", self.fast_rep)

    def test_frr_per_environment_flags_next_to_the_pooled_ones(self) -> None:
        for k, res in (("dev_check", self.res), ("held_out_check", self.res["held_out_check"])):
            d = self.fast_doc[k]
            by = self.fast_doc["environments"]["breakdown"]
            self.assertEqual(d["frr_within_limit_by_environment"],
                             {e: by[e][k]["all_within_limit"] for e in ("shelf", "tabletop")})
            self.assertEqual(d["frr_within_limit_by_environment"], res["frr_within_limit_by_environment"])
            self.assertEqual(d["all_environments_within_limit"], res["all_environments_within_limit"])
            self.assertIn("does NOT imply that each environment passed", d["pooled_note"])
            self.assertIn("never gates freeze eligibility", d["pooled_note"])
        # the pooled headline and freeze eligibility are what they were: the pooled rows' flags, CV feasibility only
        self.assertEqual(self.fast_doc["dev_check"]["all_within_limit"], self.res["all_within_limit"])
        self.assertEqual(self.res["all_within_limit"], sel.all_within_limit(self.res["flags"], self.res["unchecked"]))
        self.assertEqual(sel.freeze_eligibility(self.res["folds"], self.dev.arrangements)[0], self.res["eligible_for_freeze"])
        self.assertIn("a pooled pass does NOT imply that each environment passed", self.fast_rep)
        self.assertIn("`all_environments_within_limit`", self.fast_rep)

    def test_environment_verdicts(self) -> None:
        v = lambda *xs: sel.environment_verdicts({f"e{i}": {"all_within_limit": x} for i, x in enumerate(xs)})
        self.assertEqual(v(True, True), {"frr_within_limit_by_environment": {"e0": True, "e1": True},
                                         "all_environments_within_limit": True})
        self.assertIs(v(True, False)["all_environments_within_limit"], False)   # a pooled pass can hide this
        self.assertIs(v(None, False)["all_environments_within_limit"], False)
        self.assertIsNone(v(True, None)["all_environments_within_limit"])

    def test_cli_needs_one_of_each_flag_per_set(self) -> None:
        with quiet():
            rc = sel.main([*self.args[:8], *self.args[8:14], *QUICK, "--out", str(self.out.parent / "c")])
        self.assertEqual(rc, 2)
        self.assertFalse((self.out.parent / "c").exists())


if __name__ == "__main__":
    unittest.main()
