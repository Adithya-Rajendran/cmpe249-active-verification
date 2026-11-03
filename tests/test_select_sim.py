"""Checkpoint 2 tests for verify/select.py: select-v4 on sim dev set files, the held-out policy check (with the
per-fold training hook), freeze eligibility, the stratified bootstrap and the SIMULATION label. The sim sets here are
hand-built in temp dirs; their scores are invented and no number here is a threshold."""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import shutil
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np

import verify.policies as pol
from analysis.metrics import compute
from verify import select as sel
from verify.manifest import SIM_VIEW_LABELS_SCHEMA, SIMULATION_LABEL
from verify.schema import (
    REPO_ROOT,
    Condition,
    EvidenceRecord,
    EvidenceStore,
    Lighting,
    Positions,
    RunContext,
    Thresholds,
    episode_id,
)

SIM_POS = REPO_ROOT / "configs" / "sim" / "positions_shelf_A.json"
CTX = RunContext(synthetic=False)
QUICK = ["--cap-steps", "none", "--r-seeds", "1", "--k-sensitivity", "none", "--lam-sensitivity", "none",
         "--bootstrap-B", "20", "--tau-q-grid", "0.5,0.6", "--tau-occ-grid", "0.3", "--tau-match-grid", "0.5,0.6"]
FAST = sel.SelectionSettings(cap_steps=(2,), r_seeds=2, bootstrap_B=40, k_sensitivity=(), lam_sensitivity=(),
                             tau_q_grid=(0.5, 0.6), tau_occ_grid=(0.3,), tau_match_grid=(0.5, 0.6))
# (arrangement, family, M16 side): one of every quota slot class, two S8
SLOTS = (("SD0000", "S8", None), ("SD0001", "M16", "L"), ("SD0002", "M16", "R"), ("SD0003", "IND28", None),
         ("SD0004", "white_box", None), ("SD0005", "ycb_box", None), ("SD0006", "S8", None))


def blocked_of(i: int) -> set[int]:
    return [{1, 2}, {0, 1}, {3, 4}, {0, 1, 2, 3, 4}, {2}, {2, 3}, {1, 2}][i % 7]


def write_sim_set(d: Path, slots=SLOTS, *, split: str = "dev", simulation: bool = True, labels_simulation=None,
                  extra: str | None = None, family: bool = True, l2_quality: float = 0.85,
                  camouflaged: bool = False) -> list[str]:
    """A tiny sim set as sim/episodes.py lays it out: manifest.json (+ quota keys and the simulation flag),
    evidence.jsonl (EvidenceStore format), view_labels.json (cmpe249.sim-view-labels-set.v1) and the positions.
    The occluder blocks blocked_of(i) in the occluded and removed episodes; those views are non-diagnostic. extra adds
    one more arrangement to the evidence and labels files only (as a combined jobs file would). camouflaged: the
    occluder hides the present target WITHOUT an obstruction cue (obstruction 0.05), so those occluded views are clear
    misses and multi-view policies make false removals."""
    d.mkdir(parents=True, exist_ok=True)
    arrs = []
    for i, (a, fam, side) in enumerate(slots):
        rec = {"arrangement_id": a, "target_id": "mug", "split": split, "lightings": ["L1", "L2"],
               "indistinguishable": fam == "IND28"}
        if family:
            rec.update(occluder_family=fam, m16_side=side, design_case=False)
        arrs.append(rec)
    man = {"schema": "cmpe249.arrangements.v1", **({"simulation": True} if simulation else {}), "arrangements": arrs}
    (d / "manifest.json").write_text(json.dumps(man, indent=2))
    store = EvidenceStore(5, synthetic=False)
    eps = {}
    ids = [a for a, _, _ in slots] + ([extra] if extra else [])
    for i, a in enumerate(ids):
        blocked = blocked_of(i)
        for cond in Condition:
            for light in Lighting:
                eid = episode_id(a, cond, light)
                q = 0.9 if light is Lighting.L1 else l2_quality
                stops = []
                for st in range(5):
                    hidden = cond is not Condition.VISIBLE and st in blocked
                    m = 0.05 if cond is Condition.REMOVED else (0.1 if hidden else 0.9)
                    o = 0.05 if hidden and camouflaged and cond is Condition.OCCLUDED else (0.9 if hidden else 0.05)
                    store.put(eid, EvidenceRecord(st, m, q, o))
                    stops.append({"stop": st, "name": f"P{st + 1}", "visible_fraction": None,
                                  "diagnostic": not hidden})
                eps[eid] = {"arrangement_id": a, "condition": cond.value, "lighting": light.value,
                            "truth": "removed" if cond is Condition.REMOVED else "present", "stops": stops}
    store.to_jsonl(d / "evidence.jsonl")
    lab = {"schema": SIM_VIEW_LABELS_SCHEMA,
           "simulation": simulation if labels_simulation is None else labels_simulation,
           "rules": {"present": "visible fraction >= 0.9"}, "episodes": eps}
    (d / "view_labels.json").write_text(json.dumps(lab, indent=1))
    shutil.copyfile(SIM_POS, d / "positions.json")
    return ["--manifest", str(d / "manifest.json"), "--evidence", str(d / "evidence.jsonl"),
            "--view-labels", str(d / "view_labels.json"), "--positions", str(d / "positions.json")]


def load(d: Path) -> sel.DevData:
    return sel.sim_dev_data(d / "manifest.json", d / "evidence.jsonl", d / "view_labels.json", d / "positions.json")


def quiet() -> contextlib.ExitStack:
    st = contextlib.ExitStack()
    st.enter_context(contextlib.redirect_stdout(io.StringIO()))
    st.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return st


class SimInputTest(unittest.TestCase):
    def test_builds_the_same_devdata_as_the_library_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_sim_set(d)
            dev = load(d)
            store = EvidenceStore.from_jsonl(d / "evidence.jsonl", 5)
        self.assertEqual(dev.arrangements, tuple(a for a, _, _ in SLOTS))
        self.assertEqual(len(dev.specs), 6 * len(SLOTS))
        self.assertTrue(dev.simulation)
        self.assertFalse(dev.synthetic)
        self.assertEqual(dev.label, SIMULATION_LABEL)
        self.assertEqual(dev.positions, Positions.load(SIM_POS))
        # the same DevData the library path builds: evidence, labels and truth
        ref = sel.RecordStoreFactory.from_store(store, [s.episode_id for s in dev.specs])
        self.assertEqual(dev.evidence_sha256, ref.evidence_sha256())
        lab = {(l.episode_id, l.stop): l.diagnostic for l in dev.view_labels}
        for i, (a, _, _) in enumerate(SLOTS):
            for st in range(5):
                self.assertTrue(lab[(episode_id(a, "visible", "L2"), st)])
                self.assertEqual(lab[(episode_id(a, "removed", "L1"), st)], st not in blocked_of(i))
        self.assertEqual(dev.strata, {"SD0000": "S8", "SD0001": "M16-L", "SD0002": "M16-R", "SD0003": "IND28",
                                      "SD0004": "white_box", "SD0005": "ycb_box", "SD0006": "S8"})
        self.assertEqual(set(dev.inputs), {"manifest", "evidence", "view_labels", "positions", "ignored_episodes"})

    def test_refuses_any_non_dev_arrangement(self) -> None:
        for split in ("test", "pilot"):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as d:
                args = write_sim_set(Path(d), split=split)
                with self.assertRaises(sel.SplitRefused):
                    load(Path(d))
                with quiet():
                    rc = sel.main([*args, *QUICK, "--out", str(Path(d) / "out")])
                self.assertEqual(rc, 2)
                self.assertFalse((Path(d) / "out").exists())

    def test_one_test_arrangement_among_dev_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_sim_set(d)
            man = json.loads((d / "manifest.json").read_text())
            man["arrangements"][3]["split"] = "test"
            (d / "manifest.json").write_text(json.dumps(man))
            with self.assertRaises(sel.SplitRefused) as cm:
                load(d)
        self.assertIn("SD0003", str(cm.exception))

    def test_labels_come_only_from_view_labels(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_sim_set(d)
            base_dev = load(d)
            base = sel.run_selection(base_dev, FAST, CTX)["thresholds"]
            man = json.loads((d / "manifest.json").read_text())
            for a in man["arrangements"]:
                a["indistinguishable"] = not a["indistinguishable"]   # the manifest flag is not a label
            (d / "manifest.json").write_text(json.dumps(man))
            same = sel.run_selection(load(d), FAST, CTX)["thresholds"]
            lab = json.loads((d / "view_labels.json").read_text())
            for e in lab["episodes"].values():   # every removed view diagnostic: the labels move the counts
                if e["condition"] == "removed":
                    for st in e["stops"]:
                        st["diagnostic"] = True
            (d / "view_labels.json").write_text(json.dumps(lab))
            changed = load(d)
        self.assertEqual(base, same)
        c0 = sel.calibrate(sel.view_rows(base_dev), FAST, log_warnings=False).choice.chosen
        c1 = sel.calibrate(sel.view_rows(changed), FAST, log_warnings=False).choice.chosen
        self.assertEqual(c1.clear_miss[1], 5 * 2 * len(SLOTS))   # every removed view now counts
        self.assertLess(c0.clear_miss[1], c1.clear_miss[1])
        self.assertEqual(changed.evidence_sha256, base_dev.evidence_sha256)
        self.assertNotEqual(changed.view_labels_sha256, base_dev.view_labels_sha256)

    def test_extra_episodes_in_evidence_and_labels_are_ignored_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_sim_set(d, extra="DES0")
            with self.assertLogs("verify.select", level="WARNING"):
                dev = load(d)
        self.assertNotIn("DES0", dev.arrangements)
        ign = dev.inputs["ignored_episodes"]
        self.assertEqual((ign["evidence"]["n"], ign["evidence"]["arrangements"]), (6, ["DES0"]))
        self.assertEqual((ign["view_labels"]["n"], ign["view_labels"]["arrangements"]), (6, ["DES0"]))

    def test_missing_evidence_or_labels_and_flag_mismatch_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_sim_set(d)
            lab = json.loads((d / "view_labels.json").read_text())
            del lab["episodes"][episode_id("SD0002", "occluded", "L2")]
            (d / "view_labels.json").write_text(json.dumps(lab))
            with self.assertRaises(ValueError):
                load(d)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_sim_set(d, labels_simulation=False)
            with self.assertRaises(ValueError):
                load(d)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_sim_set(d)
            lines = (d / "evidence.jsonl").read_text().splitlines()
            (d / "evidence.jsonl").write_text("\n".join(lines[1:]) + "\n")   # one stop of one episode missing
            with self.assertRaises(KeyError):
                load(d)

    def test_cli_needs_all_four_files(self) -> None:
        with tempfile.TemporaryDirectory() as d, quiet():
            args = write_sim_set(Path(d))
            self.assertEqual(sel.main(args[:6] + QUICK), 2)        # no --positions
            self.assertEqual(sel.main(["--synthetic", *args]), 2)  # not both
            self.assertEqual(sel.main([]), 2)


class SimCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.args = write_sim_set(d / "set")
        with quiet():
            cls.rc = [sel.main([*cls.args, *QUICK, "--out", str(d / name)]) for name in ("a", "b")]
        cls.out = d / "a"
        cls.doc = json.loads((cls.out / "frozen_candidate.json").read_text())
        cls.rep = (cls.out / sel.REPORT_FILE).read_text()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_exit_0_eligible_and_byte_identical(self) -> None:
        self.assertEqual(self.rc, [0, 0])
        self.assertIs(self.doc["eligible_for_freeze"], True)
        for name in ("frozen_candidate.json", sel.REPORT_FILE, "selection_details.json"):
            self.assertEqual((self.out / name).read_bytes(), (self.out.parent / "b" / name).read_bytes(), name)

    def test_simulation_label_everywhere(self) -> None:
        self.assertEqual((self.doc["simulation"], self.doc["synthetic"], self.doc["label"]),
                         (True, False, SIMULATION_LABEL))
        self.assertTrue(self.rep.startswith(f"# {SIMULATION_LABEL}"))
        self.assertTrue(self.rep.rstrip().endswith(f"_{SIMULATION_LABEL}._"))
        self.assertNotIn("REAL PILOT", self.rep)
        # the hashed settings record the flag: the same files without it hash differently
        dev = load(self.out.parent / "set")
        s = sel.SelectionSettings(**{**dataclasses.asdict(FAST), "cap_steps": (), "r_seeds": 1, "bootstrap_B": 20,
                                     "tau_match_grid": (0.5, 0.6)})
        prov = sel.data_provenance(dev)
        self.assertNotEqual(sel.settings_sha256(s, None, prov),
                            sel.settings_sha256(s, None, {**prov, "simulation": False}))

    def test_input_file_hashes_recorded(self) -> None:
        d = self.out.parent / "set"
        for name, f in (("manifest", "manifest.json"), ("evidence", "evidence.jsonl"),
                        ("view_labels", "view_labels.json"), ("positions", "positions.json")):
            from verify.schema import file_sha256
            self.assertEqual(self.doc["inputs"][name]["sha256"], file_sha256(d / f), name)
            self.assertIn(self.doc["inputs"][name]["sha256"], self.rep)

    def test_stratified_bootstrap_in_the_dev_check(self) -> None:
        st = self.doc["bootstrap_stratification"]
        self.assertTrue(st["stratified"])
        self.assertEqual(st["strata"], {"IND28": 1, "M16-L": 1, "M16-R": 1, "S8": 2, "white_box": 1, "ycb_box": 1})
        boots = [r["bootstrap"] for r in self.doc["dev_check"]["rows"] + self.doc["held_out_check"]["rows"]
                 if r["bootstrap"]]
        self.assertTrue(boots and all(b["stratified"] for b in boots))
        self.assertIn("stratified by quota slot class", self.rep)

    def test_freeze_gate_accepts_this_candidate_and_refuses_tampering(self) -> None:
        doc = sel.require_freeze_eligible(self.out / "frozen_candidate.json")
        self.assertIs(doc["eligible_for_freeze"], True)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for name in ("frozen_candidate.json", "frozen_candidate.json.sha256"):
                shutil.copyfile(self.out / name, t / name)
            p = t / "frozen_candidate.json"
            p.write_text(p.read_text().replace('"tau_match": 0.5', '"tau_match": 0.55', 1))
            with self.assertRaises(sel.NotEligibleForFreeze):
                sel.require_freeze_eligible(p)
            (t / "frozen_candidate.json.sha256").unlink()
            with self.assertRaises(sel.NotEligibleForFreeze):
                sel.require_freeze_eligible(p)
        for i, bad in enumerate(({**self.doc, "eligible_for_freeze": False}, {k: v for k, v in self.doc.items()
                                                                 if k != "eligible_for_freeze"},
                    {**self.doc, "eligible_for_freeze": "true"}, {**self.doc, "synthetic": True},
                    {**self.doc, "procedure_revision": "select-v4"}, {**self.doc, "schema": "x"})):
            with self.subTest(case=i), \
                    self.assertRaises(sel.NotEligibleForFreeze):
                sel.require_freeze_eligible(bad)


class EligibilityExitTest(unittest.TestCase):
    def test_infeasible_on_all_dev_exits_3_with_no_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as d, quiet():
            d = Path(d)
            args = write_sim_set(d / "set")
            out = d / "out"
            self.assertEqual(sel.main([*args, *QUICK, "--out", str(out)]), 0)
            rc = sel.main([*args, *QUICK[:-6], "--tau-q-grid", "0", "--tau-occ-grid", "1", "--tau-match-grid", "0.95",
                           "--out", str(out)])
            self.assertEqual(rc, sel.EXIT_INFEASIBLE)
            self.assertFalse((out / "frozen_candidate.json").exists())
            doc = json.loads((out / sel.INFEASIBLE_FILE).read_text())
            rep = (out / sel.REPORT_FILE).read_text()
        self.assertEqual((doc["simulation"], doc["label"]), (True, SIMULATION_LABEL))
        self.assertTrue(rep.startswith(f"# {SIMULATION_LABEL}"))
        self.assertEqual(doc["failure"]["binding"], "both")
        self.assertEqual(set(doc["inputs"]) - {"ignored_episodes"}, {"manifest", "evidence", "view_labels", "positions"})

    def test_an_infeasible_fold_writes_an_ineligible_candidate_and_exits_4(self) -> None:
        # The fold that holds out SD0002 is made infeasible by a stand-in calibrate (every gate fails the false-clear
        # constraint there); every other calibration is the real one. The candidate is written but ineligible.
        real = sel.calibrate

        def cal(rows, settings, arrangements=None, **kw):
            if arrangements is not None and "SD0002" not in set(arrangements):
                pts = sel.ViewTable([r for r in rows if r.arrangement_id in set(arrangements)], settings).points()
                bad = [dataclasses.replace(p, false_clear=(p.false_clear[1], p.false_clear[1] or 1)) for p in pts]
                raise sel.CalibrationInfeasible(sel.Infeasibility.of(bad, settings.alpha_clear, {}, arrangements))
            return real(rows, settings, arrangements, **kw)
        with tempfile.TemporaryDirectory() as d, quiet(), mock.patch.object(sel, "calibrate", cal):
            d = Path(d)
            args = write_sim_set(d / "set")
            rc = sel.main([*args, *QUICK, "--out", str(d / "out")])
            doc = json.loads((d / "out" / "frozen_candidate.json").read_text())
            rep = (d / "out" / sel.REPORT_FILE).read_text()
        self.assertEqual(rc, sel.EXIT_INELIGIBLE)
        self.assertIs(doc["eligible_for_freeze"], False)
        self.assertEqual([(r["fold"], r["constraint"]) for r in doc["ineligible_reasons"]], [("SD0002", "false-clear")])
        self.assertFalse(doc["calibration"]["grouped_cv"]["complete"])
        self.assertFalse(doc["held_out_check"]["complete"])
        self.assertNotIn("SD0002", doc["held_out_check"]["arrangements_run"])
        self.assertIn("eligible_for_freeze: FALSE", rep)
        with self.assertRaises(sel.NotEligibleForFreeze):
            sel.require_freeze_eligible(doc)

    def test_one_dev_arrangement_is_not_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as d, quiet():
            d = Path(d)
            args = write_sim_set(d / "set", slots=SLOTS[:1])
            rc = sel.main([*args, *QUICK, "--out", str(d / "out")])
            doc = json.loads((d / "out" / "frozen_candidate.json").read_text())
        self.assertEqual(rc, sel.EXIT_INELIGIBLE)
        self.assertIn("fewer than 2 dev arrangements", doc["ineligible_reasons"][0]["reason"])

    def test_synthetic_candidate_is_never_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as d, quiet():
            out = Path(d) / "out"
            self.assertEqual(sel.main(["--synthetic", "--n-arr", "9", *QUICK[:8], "--out", str(out)]), 0)
            with self.assertRaises(sel.NotEligibleForFreeze) as cm:
                sel.require_freeze_eligible(out / "frozen_candidate.json")
        self.assertIn("synthetic", str(cm.exception))


class DummyTrainable:
    """A fold-trained policy (the hook ranker A will use) that records everything it was given."""
    name = "DUMMY"

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def train(self, train: sel.DevData, thresholds: Thresholds) -> Callable[[], object]:
        store = train.store_factory()
        self.seen.append({"arrangements": set(train.arrangements), "store": set(store.episode_ids()),
                          "truth": set(train.truth.states), "labels": {l.episode_id for l in train.view_labels},
                          "strata": set(train.strata or {}), "thresholds": thresholds, "store_obj": store})
        return pol.F   # a fresh F per episode: the policy itself is not the point here


class HeldOutCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        write_sim_set(Path(cls.tmp.name), l2_quality=0.45)
        cls.dev = load(Path(cls.tmp.name))
        cls.dummy = DummyTrainable()
        cls.res = sel.run_selection(cls.dev, FAST, CTX, trainables=(cls.dummy,))
        cls.ho = cls.res["held_out_check"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_trainable_never_sees_the_held_out_arrangement(self) -> None:
        folds = [f for f in self.res["folds"] if f.thresholds is not None]
        self.assertEqual(len(self.dummy.seen), len(folds))
        for f, seen in zip(folds, self.dummy.seen):
            with self.subTest(fold=f.held_out):
                self.assertEqual(seen["arrangements"], set(self.dev.arrangements) - {f.held_out})
                self.assertEqual(seen["strata"], seen["arrangements"])
                for key in ("store", "truth", "labels"):
                    self.assertFalse({e for e in seen[key] if e.startswith(f.held_out + "-")}, key)
                self.assertEqual(seen["thresholds"], f.thresholds)   # the fold's own thresholds
                with self.assertRaises(KeyError):   # the evidence is not reachable through what it was given
                    seen["store_obj"].get(episode_id(f.held_out, "occluded", "L1"), 0)

    def test_every_policy_runs_with_its_folds_thresholds(self) -> None:
        events: list[tuple[str, Thresholds, int]] = []

        def record(real: Callable) -> Callable:
            def f(policy, env, rule, **kw):
                run = real(policy, env, rule, **kw)
                events.append((run.policy, rule.thresholds, rule.k))
                return run
            return f
        fold_th = {f.held_out: f.thresholds for f in self.res["folds"]}
        with mock.patch.object(sel, "run_episode", record(pol.run_episode)), \
                mock.patch.object(pol, "run_episode", record(pol.run_episode)):
            ho = sel.held_out_check(self.dev, FAST, CTX, self.res["folds"], (DummyTrainable(),))
        self.assertEqual(len(events), ho["n_runs"])
        self.assertEqual({p for p, _, _ in events}, set(sel.HELDOUT_POLICIES) | {"DUMMY", "DUMMY-fcm"})
        # every held-out episode's run used its own fold's thresholds (episodes are logged in run order)
        eps = ho["episodes"]
        by_fold = [e["fold"] for e in eps]
        self.assertEqual(sorted(set(by_fold)), sorted(a for a, t in fold_th.items() if t is not None))
        for e in eps:
            self.assertTrue(e["episode_id"].startswith(e["fold"] + "-"))
        ths = {(round(t.tau_q, 9), t.tau_occ, t.tau_match) for _, t, _ in events}
        self.assertLessEqual(ths, {(t.tau_q, t.tau_occ, t.tau_match) for t in fold_th.values() if t})
        self.assertEqual({k for p, _, k in events if p.endswith("-fcm") or p == "S"}, {1})
        self.assertEqual({k for p, _, k in events if not p.endswith("-fcm") and p != "S"}, {2})

    def test_rows_recompute_from_the_episodes(self) -> None:
        # G uncapped, L2 queries: pool the per-fold results by hand with analysis.metrics
        runner_rows = [r for r in self.ho["rows"] if (r.policy, r.cap_steps, r.lighting) == ("G", None, "L2")]
        self.assertEqual(len(runner_rows), 1)
        c = runner_rows[0].counts
        res = []
        for f in self.res["folds"]:
            held = self.dev.subset([f.held_out])
            r = sel.PolicyRunner(held, FAST, CTX).results("G", f.thresholds, sel.run_budget("G", None, FAST,
                                                                                           self.dev.positions))
            res += [x for x in r if x.lighting is Lighting.L2]
        m = compute(res, self.dev.truth)
        self.assertAlmostEqual(c.frr, m["false_removal_rate"])
        self.assertAlmostEqual(c.recall, m["removal_recall"])
        self.assertAlmostEqual(c.coverage, m["coverage"])
        self.assertAlmostEqual(c.mean_views, m["mean_views"])
        self.assertAlmostEqual(c.mean_travel_cm, m["mean_travel_cm"])
        self.assertEqual(c.n_arr_with_false_removal, m["arrangement_false_removal"])
        self.assertEqual(c.n_episodes, 3 * len(self.dev.arrangements))

    def test_every_policy_lighting_and_cap_has_a_row(self) -> None:
        keys = {(r.policy, r.cap_steps, r.lighting) for r in self.ho["rows"]}
        for p in (*sel.HELDOUT_POLICIES, "DUMMY", "DUMMY-fcm"):
            for cap in sel.policy_caps(p, FAST):
                for light in ("L1", "L2", "pooled"):
                    self.assertIn((p, cap, light), keys)
        full = next(r for r in self.ho["rows"] if (r.policy, r.lighting) == ("FULL", "L1"))
        self.assertEqual((full.budget, full.cap_steps, full.counts.mean_views), (4, None, 5.0))

    def test_l2_gate_rejection_is_flagged(self) -> None:
        # L2 quality 0.45 < every tau_q on the grid: the gate rejects every L2 view, which the report must make obvious
        flags = self.res["gate_pass_flags"]
        self.assertIn("Step 1 (chosen setting, all dev views): the gate rejects ALL L2 removed diagnostic views "
                      f"(0/{self.res['gate_pass']['L2']['removed_diagnostic'][1]} pass tau_q / tau_occ)", flags)
        self.assertTrue(any(f.startswith("Step 3 fold without SD0000 (training views): the gate rejects ALL L2")
                            for f in flags))
        for f in self.res["folds"]:
            self.assertEqual(f.gate_pass_train["L2"]["removed_diagnostic"][0], 0)
            self.assertGreater(f.gate_pass_train["L1"]["removed_diagnostic"][0], 0)
        doc = sel.candidate_doc(self.dev, FAST, self.res, "x")
        self.assertEqual(doc["calibration"]["gate_pass_by_lighting"]["L2"]["removed_diagnostic"]["x"], 0)
        rep = sel.render_report(self.dev, FAST, self.res, doc, "y")
        self.assertIn("**GATE FLAG** Step 1 (chosen setting, all dev views): the gate rejects ALL L2", rep)
        # the held-out per-lighting evidence shows it too: no L2 removal by any k-rule policy
        g2 = next(r for r in self.ho["rows"] if (r.policy, r.cap_steps, r.lighting) == ("G", None, "L2"))
        self.assertEqual(g2.counts.n_true_removal, 0)

    def test_infeasible_fold_runs_nothing(self) -> None:
        folds = list(self.res["folds"])
        folds[0] = sel.ViewFold(folds[0].held_out, None, None, None)
        ho = sel.held_out_check(self.dev, FAST, CTX, folds)
        self.assertFalse(ho["complete"])
        self.assertNotIn(folds[0].held_out, ho["arrangements_run"])
        self.assertFalse([e for e in ho["episodes"] if e["fold"] == folds[0].held_out])

    def test_bad_trainable_names_are_refused(self) -> None:
        for name in ("G", "FULL", "A-fcm", ""):
            t = DummyTrainable()
            t.name = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                sel.run_selection(self.dev, FAST, CTX, trainables=(t,))


class SubsetAndBootstrapTest(unittest.TestCase):
    def test_subset_holds_nothing_else(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            write_sim_set(Path(d))
            dev = load(Path(d))
        sub = dev.subset(["SD0001", "SD0004"])
        self.assertEqual(sub.arrangements, ("SD0001", "SD0004"))
        store = sub.store_factory()
        self.assertEqual({e.split("-")[0] for e in store.episode_ids()}, {"SD0001", "SD0004"})
        self.assertFalse(store.sealed)
        self.assertEqual(set(sub.truth.states), {s.episode_id for s in sub.specs})
        self.assertEqual(sub.strata, {"SD0001": "M16-L", "SD0004": "white_box"})
        with self.assertRaises(ValueError):
            dev.subset(["NOPE"])

    def test_select_bootstrap_keeps_strata(self) -> None:
        # each arrangement its own stratum: every replicate is the sample itself, so the CI is the point estimate
        outs = [sel.Outcome(f"{a}-{i}", a, Lighting.L1, i < 2, float(a == "A" and i == 0), 1.0, 0.0, 1.0)
                for a in ("A", "B", "C") for i in range(3)]
        bt = sel.cluster_bootstrap(outs, 300, 0, {"A": "x", "B": "y", "C": "z"})
        self.assertEqual(bt["frr"]["ci95"], [1 / 6, 1 / 6])
        self.assertTrue(bt["stratified"])
        un = sel.cluster_bootstrap(outs, 300, 0)
        self.assertFalse(un["stratified"])
        self.assertLess(un["frr"]["ci95"][0], un["frr"]["ci95"][1])
        counts = sel._draw_counts(3, 300, 0, ("x", "x", "y"))
        self.assertTrue(np.all(counts[:, :2].sum(axis=1) == 2) and np.all(counts[:, 2] == 1))


if __name__ == "__main__":
    unittest.main()
