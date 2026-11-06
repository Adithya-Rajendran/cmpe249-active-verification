"""Side descriptors on the replay path (verify/side.py), the direction-rule control Dir and ranker A / A-nocue
(verify/policies.py, verify/ranker.py): sealing, requested-view-only access, fold isolation, the A-nocue exclusion,
the logistic fit and model selection, and the held-out check end to end on a hand-built sim set. Scores are invented;
no number here is a result."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from verify import policies as pol
from verify import ranker as rk
from verify import select as sel
from verify.replay import ReplayEnv
from verify.schema import (
    Budget,
    Condition,
    EvidenceRecord,
    EvidenceStore,
    LeakError,
    Lighting,
    Positions,
    RunContext,
    Thresholds,
    episode_id,
)
from verify.side import SIDE_SCHEMA, SideData, SideEnv, SideObservation, SideStore
from verify.stopping import StoppingRule
from tests.test_select_sim import CTX, FAST, SLOTS, blocked_of, load, quiet, reachable_strings, write_sim_set

POS = Positions.default()          # five stops, start P3 (index 2), radius 45 cm
TH = Thresholds(tau_match=0.6, tau_q=0.5, tau_occ=0.3, k=2, lam=0.0)


def side_value(i: int, cond: Condition, stop: int) -> float | None:
    """Invented descriptors: the occluder's side from the blocked stops (left = low indices), 0 when visible, one
    missing value per arrangement (stop 4 of the removed L2 episode)."""
    if cond is Condition.VISIBLE:
        return 0.0
    b = blocked_of(i)
    return 0.4 if max(b) <= 2 and min(b) < 2 else (-0.4 if min(b) >= 2 and max(b) > 2 else 0.1)


def write_side(d: Path, slots=SLOTS) -> Path:
    rows = []
    for i, (a, _, _) in enumerate(slots):
        for cond in Condition:
            for light in Lighting:
                for st in range(5):
                    v = None if (cond is Condition.REMOVED and light is Lighting.L2 and st == 4) else side_value(i, cond, st)
                    rows.append({"schema": SIDE_SCHEMA, "descriptor_version": "side-0.1-dev",
                                 "episode_id": episode_id(a, cond, light), "stop": st, "d": v,
                                 "missing": None if v is not None else "support"})
    p = d / "side.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def store_of(records: dict[int, tuple[float, float, float]], eid: str = "X0001-occluded-L1") -> tuple[EvidenceStore, str]:
    st = EvidenceStore(5)
    for s, (m, q, o) in records.items():
        st.put(eid, EvidenceRecord(s, m, q, o))
    return st, eid


def side_env(records, sides, budget=Budget(3), eid="X0001-occluded-L1", log=None) -> SideEnv:
    st, eid = store_of(records, eid)
    key = st.replay_key(eid)
    ss = SideStore(5)
    for s, v in sides.items():
        ss.put(eid, s, v)
    reveal = ss.bind(eid)

    def spy(stop: int):
        if log is not None:
            log.append(stop)
        return reveal(stop)
    return SideEnv(ReplayEnv(key, st, POS, budget), spy)


OCC = {s: (0.1, 0.9, 0.9) for s in range(5)}               # every view occluded: no decision, the policy keeps going


class SideStoreTest(unittest.TestCase):
    def test_bind_seals_and_reveals_one_episode(self) -> None:
        ss = SideStore(5)
        for s in range(5):
            ss.put("E1", s, 0.1 * s)
            ss.put("E2", s, None)
        reveal = ss.bind("E1")
        self.assertTrue(ss.sealed)
        self.assertAlmostEqual(reveal(3), 0.3)
        for call in (lambda: ss.get("E2", 0), lambda: ss.put("E3", 0, 0.0), ss.episode_ids):
            with self.assertRaises(LeakError):
                call()
        self.assertIsNone(ss.bind("E2")(1))   # binding stays possible after sealing (the runner binds per episode)
        with self.assertRaises(LeakError):
            ss.bind("nope")

    def test_values_are_checked(self) -> None:
        ss = SideStore(5)
        for bad in (1.5, float("nan"), -2.0):
            with self.assertRaises(ValueError):
                ss.put("E", 0, bad)
        ss.put("E", 0, 0.5)
        with self.assertRaises(ValueError):
            ss.put("E", 0, 0.4)

    def test_jsonl_round_trip_and_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = write_side(Path(d))
            ids = {episode_id(a, c, l) for a, _, _ in SLOTS for c in Condition for l in Lighting}
            st = SideStore.from_jsonl(p, episode_ids=ids)
            self.assertEqual(set(st.episode_ids()), ids)
            self.assertIsNone(st.get(episode_id("SD0000", "removed", "L2"), 4))
            with self.assertRaises(ValueError):          # a manifest episode without descriptors
                SideStore.from_jsonl(p, episode_ids=ids | {"SD9999-visible-L1"})
            bad = Path(d) / "bad.jsonl"
            bad.write_text(json.dumps({"schema": "other", "episode_id": "E", "stop": 0, "d": 0.0}) + "\n")
            with self.assertRaises(ValueError):
                SideStore.from_jsonl(bad)

    def test_subset_holds_only_its_episodes(self) -> None:
        data = SideData({"A-visible-L1": [0.0] * 5, "B-visible-L1": [0.2] * 5}, 5)
        sub = data.subset(["A-visible-L1"])
        self.assertEqual(sub.episode_ids, ("A-visible-L1",))
        self.assertFalse([s for s in reachable_strings(sub) if "B-" in s])
        with self.assertRaises(ValueError):
            data.subset(["C-visible-L1"])


class SideEnvTest(unittest.TestCase):
    def test_only_requested_stops_are_revealed(self) -> None:
        log: list[int] = []
        env = side_env(OCC, {s: 0.1 * s for s in range(5)}, log=log)
        run = pol.run_episode(pol.G.from_thresholds(TH), env, StoppingRule(TH))
        self.assertEqual(log, list(run.requested_stops))
        self.assertEqual(len(log), 4)                    # P3 and three more: unknown at b = 3

    def test_observations_carry_their_own_stop_descriptor(self) -> None:
        env = side_env(OCC, {0: -0.2, 1: None, 2: 0.3, 3: 0.4, 4: 0.5})
        o = env.reset()
        self.assertIsInstance(o, SideObservation)
        self.assertEqual((o.stop, o.side), (2, 0.3))
        self.assertEqual(env.request(1).side, None)
        self.assertEqual(env.request(0).side, -0.2)

    def test_policies_that_do_not_read_it_are_unchanged(self) -> None:
        for make in (pol.F, lambda: pol.G.from_thresholds(TH), lambda: pol.G.from_thresholds(TH, lookahead=True)):
            st, eid = store_of(OCC)
            plain = pol.run_episode(make(), ReplayEnv(st.replay_key(eid), st, POS, Budget(3, 60.0)), StoppingRule(TH))
            wrapped = pol.run_episode(make(), side_env(OCC, {s: 0.5 for s in range(5)}, Budget(3, 60.0)),
                                      StoppingRule(TH))
            self.assertEqual(plain, wrapped)


class DirTest(unittest.TestCase):
    def run_dir(self, d3, budget=Budget(3)):
        sides = {s: 0.0 for s in range(5)}
        sides[2] = d3
        return pol.run_episode(pol.Dir.from_thresholds(TH), side_env(OCC, sides, budget), StoppingRule(TH)).requested_stops

    def test_positive_goes_to_larger_azimuth_nearest_first(self) -> None:
        self.assertEqual(self.run_dir(0.3)[:3], (2, 3, 4))

    def test_negative_goes_to_smaller_azimuth_nearest_first(self) -> None:
        self.assertEqual(self.run_dir(-0.3)[:3], (2, 1, 0))

    def test_missing_or_zero_is_primary_g(self) -> None:
        g = pol.run_episode(pol.G.from_thresholds(TH), side_env(OCC, {s: 0.0 for s in range(5)}), StoppingRule(TH))
        for d3 in (None, 0.0):
            self.assertEqual(self.run_dir(d3), g.requested_stops)

    def test_two_step_cap_keeps_one_side(self) -> None:
        cap = 2 * POS.travel_cm(2, 3)
        self.assertEqual(self.run_dir(0.3, Budget(3, cap)), (2, 3, 4))
        self.assertEqual(self.run_dir(-0.3, Budget(3, cap)), (2, 1, 0))

    def test_side_exhausted_falls_back_to_g(self) -> None:
        stops = self.run_dir(0.3)
        self.assertEqual(len(stops), 4)
        self.assertIn(stops[3], (0, 1))
        self.assertEqual(pol.DIR_NAME, "Dir")


def obs(stops_vals) -> tuple[SideObservation, ...]:
    return tuple(SideObservation(step=i, stop=s, azimuth_deg=POS.azimuth(s), match_score=m, quality=q, obstruction=o,
                                 side=d) for i, (s, m, q, o, d) in enumerate(stops_vals))


class FeatureTest(unittest.TestCase):
    def test_values(self) -> None:
        h = obs([(2, 0.2, 0.8, 0.7, 0.4), (3, 0.5, 0.6, 0.1, None)])
        f = pol.ranker_features(h, 0, POS, 0.3)
        self.assertEqual(set(f), set(pol.RANKER_FEATURES))
        self.assertAlmostEqual(f["az_from_start"], 1.0)              # P1 at -45 deg; half-span 45 deg
        self.assertAlmostEqual(f["sep_visited"], 1.0)                # nearest visited is P3
        self.assertAlmostEqual(f["sep_occluded"], 1.0)               # only P3 is flagged occluded (0.7 > 0.3)
        self.assertAlmostEqual(f["obstruction_start"], 0.7)
        self.assertAlmostEqual(f["side_toward"], -0.4)               # P3 favours larger azimuth; P1 is smaller
        self.assertEqual(f["side_missing"], 0.0)
        self.assertAlmostEqual(f["match_max"], 0.5)
        self.assertAlmostEqual(f["quality_min"], 0.6)
        f4 = pol.ranker_features(h, 4, POS, 0.3)
        self.assertAlmostEqual(f4["side_toward"], 0.4)
        self.assertAlmostEqual(f4["sep_visited"], 0.5)               # P5 is 22.5 deg from P4

    def test_missing_side_and_visited_candidate(self) -> None:
        f = pol.ranker_features(obs([(2, 0.2, 0.8, 0.1, None)]), 1, POS, 0.3)
        self.assertEqual((f["side_toward"], f["side_missing"]), (0.0, 1.0))
        self.assertAlmostEqual(f["sep_occluded"], f["sep_visited"])  # nothing flagged: G's reference set is all visited
        with self.assertRaises(ValueError):
            pol.ranker_features(obs([(2, 0.2, 0.8, 0.1, None)]), 2, POS, 0.3)

    def test_nocue_excludes_every_obstruction_derived_feature(self) -> None:
        self.assertEqual(set(pol.OBSTRUCTION_DERIVED), {"sep_occluded", "obstruction_start", "side_toward", "side_missing"})
        self.assertFalse(set(pol.NOCUE_FEATURES) & set(pol.OBSTRUCTION_DERIVED))
        rng = np.random.default_rng(0)
        model = dict(mean=rng.normal(size=8), scale=rng.uniform(0.5, 2, 8), coef=rng.normal(size=8), intercept=0.3)
        a = pol.Ranker("A", pol.RANKER_FEATURES, tau_occ=0.3, **model)
        idx = [pol.RANKER_FEATURES.index(f) for f in pol.NOCUE_FEATURES]
        nocue = pol.Ranker("A-nocue", pol.NOCUE_FEATURES, model["mean"][idx], model["scale"][idx],
                           model["coef"][idx], 0.3, 0.3)
        base = [(2, 0.2, 0.8, 0.1, 0.4), (4, 0.3, 0.7, 0.9, -0.2)]
        changed = [(s, m, q, 1.0 - o, None if d is not None else 0.5) for s, m, q, o, d in base]
        for c in (0, 1, 3):
            self.assertEqual(nocue.probability(obs(base), c, POS), nocue.probability(obs(changed), c, POS))
        self.assertNotEqual(a.probability(obs(base), 0, POS), a.probability(obs(changed), 0, POS))


class RequestedViewsOnlyTest(unittest.TestCase):
    def test_unrequested_views_never_change_the_run(self) -> None:
        rng = np.random.default_rng(1)
        a = pol.Ranker("A", pol.RANKER_FEATURES, rng.normal(size=8), rng.uniform(0.5, 2, 8), rng.normal(size=8), 0.1,
                       0.3)
        recs = {s: (0.1, 0.9, 0.9 if s in (1, 2) else 0.05) for s in range(5)}
        sides = {s: (0.3 if s == 2 else -0.1) for s in range(5)}
        for budget in (Budget(1), Budget(2), Budget(3), Budget(3, 2 * POS.travel_cm(2, 3))):
            first = pol.run_episode(a, side_env(recs, sides, budget), StoppingRule(TH))
            untouched = set(first.requested_stops)
            recs2 = {s: (v if s in untouched else (0.99, 0.01, 0.0)) for s, v in recs.items()}
            sides2 = {s: (v if s in untouched else 1.0) for s, v in sides.items()}
            again = pol.run_episode(a, side_env(recs2, sides2, budget), StoppingRule(TH))
            self.assertEqual(first, again)

    def test_training_and_decision_features_agree(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            write_sim_set(Path(d))
            dev = dataclasses.replace(load(Path(d)), side=SideData.from_store(
                SideStore.from_jsonl(write_side(Path(d))), load(Path(d)).store_factory().episode_ids()))
            eid = episode_id("SD0001", "occluded", "L1")
            st = dev.store_factory()
            ss = dev.side()
            recs = {s: st.get(eid, s) for s in range(5)}
            env = side_env({s: (r.match_score, r.quality, r.obstruction) for s, r in recs.items()},
                           {s: ss.get(eid, s) for s in range(5)})
            seen = [env.reset(), env.request(1)]
            by_env = pol.ranker_features(tuple(seen), 0, POS, TH.tau_occ)
            built = tuple(SideObservation(step=i, stop=s, azimuth_deg=POS.azimuth(s), match_score=recs[s].match_score,
                                          quality=recs[s].quality, obstruction=recs[s].obstruction,
                                          side=ss.get(eid, s)) for i, s in enumerate((2, 1)))
            self.assertEqual(by_env, pol.ranker_features(built, 0, POS, TH.tau_occ))


class FitTest(unittest.TestCase):
    def problem(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 3))
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 + X @ np.array([1.0, -2.0, 0.0]))))).astype(float)
        w = np.full(n, 3.0 / n)
        return X, y, w

    def test_optimum_has_zero_gradient(self) -> None:
        X, y, w = self.problem()
        for C in rk.C_GRID:
            b0, b = rk.fit_logistic(X, y, w, C)
            z = b0 + X @ b
            mu = 1 / (1 + np.exp(-z))
            grad = np.r_[(w * (mu - y)).sum(), X.T @ (w * (mu - y)) + b / C]
            self.assertLess(np.max(np.abs(grad)), 1e-8, C)

    def test_more_regularisation_shrinks(self) -> None:
        X, y, w = self.problem()
        norms = [np.linalg.norm(rk.fit_logistic(X, y, w, C)[1]) for C in rk.C_GRID]
        self.assertEqual(norms, sorted(norms))

    def test_one_label_class_does_not_converge(self) -> None:
        X, _, w = self.problem(50)
        with self.assertRaises(RuntimeError):
            rk.fit_logistic(X, np.ones(50), w, 1.0)

    def test_constant_feature_is_only_centred(self) -> None:
        X = np.c_[np.ones(10), np.arange(10.0)]
        m, s = rk.standardise(X, np.ones(10))
        self.assertEqual((m[0], s[0]), (1.0, 1.0))

    def test_ties_go_to_the_smaller_c(self) -> None:
        X, y, w = self.problem()
        rows = rk.Rows(X, y, w, tuple(f"A{i % 10}" for i in range(len(y))), ("a", "b", "c"))
        with mock.patch.object(rk, "weighted_logloss", return_value=0.5):
            self.assertEqual(rk.fit_with_selection(rows).C, min(rk.C_GRID))

    def test_inner_folds_partition_the_arrangements(self) -> None:
        folds = rk.inner_folds([f"A{i}" for i in range(12)])
        self.assertEqual(len(folds), rk.INNER_FOLDS)
        self.assertEqual(sorted(a for f in folds for a in f), sorted(f"A{i}" for i in range(12)))
        with self.assertRaises(ValueError):
            rk.inner_folds(["A", "B"])


def side_dev(d: Path) -> sel.DevData:
    write_sim_set(d)
    return sel.sim_dev_data(d / "manifest.json", d / "evidence.jsonl", d / "view_labels.json", d / "positions.json",
                            write_side(d))


class TrainingRowsTest(unittest.TestCase):
    def test_rows_weights_and_nonterminal_histories(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            dev = side_dev(Path(d))
            rows = rk.training_rows(dev, TH, pol.RANKER_FEATURES)
            totals = {}
            for a, w in zip(rows.arrangement, rows.w):
                totals[a] = totals.get(a, 0.0) + w
            for a, t in totals.items():
                self.assertAlmostEqual(t, 1.0, msg=a)
            self.assertEqual(rows.X.shape, (len(rows.y), 8))
            # visible episodes decide PRESENT at P3 (match 0.9): no rows from them; every row is a nonterminal state
            self.assertTrue(set(np.unique(rows.y)) <= {0.0, 1.0})
            self.assertEqual(len(rk.histories(5, 2)), 1 + 4 + 12)

    def test_rankers_need_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            write_sim_set(Path(d))
            with self.assertRaises(ValueError):
                rk.training_rows(load(Path(d)), TH, pol.NOCUE_FEATURES)


class HeldOutRankerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dev = side_dev(Path(cls.tmp.name))
        with quiet():
            cls.res = sel.run_selection(cls.dev, FAST, CTX, trainables=(rk.RankerA, rk.RankerANoCue))
        cls.ho = cls.res["held_out_check"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_every_policy_ran_held_out(self) -> None:
        for p in ("A", "A-fcm", "A-nocue", "A-nocue-fcm", "Dir", "Dir-fcm", "G", "G-lookahead"):
            self.assertIn(p, self.ho["policies"])
            self.assertTrue([r for r in self.ho["rows"] if r.policy == p], p)
        self.assertIn("Dir", [r.policy for r in self.res["check"]])

    def test_each_fold_trained_on_the_others_only(self) -> None:
        ran = [f for f in self.ho["folds"] if f["ran"]]
        self.assertEqual(len(ran), len(self.dev.arrangements))
        for f in ran:
            self.assertNotIn(f["held_out"], f["training_arrangements"])
            for name in ("A", "A-nocue"):
                rep = f["trained_reports"][name]
                self.assertEqual(rep["n_training_arrangements"], len(self.dev.arrangements) - 1)
                self.assertIn(rep["C"], rk.C_GRID)
                self.assertEqual(set(rep["inner_logloss"]), {f"{c:g}" for c in rk.C_GRID})
            self.assertEqual(set(f["trained_reports"]["A-nocue"]["coef"]), set(pol.NOCUE_FEATURES))

    def test_training_subset_holds_no_held_out_descriptor(self) -> None:
        for held in self.dev.arrangements:
            train = self.dev.subset([a for a in self.dev.arrangements if a != held])
            self.assertFalse([e for e in train.side.episode_ids if e.startswith(held)])
            self.assertFalse([s for s in reachable_strings(train) if held in s])

    def test_side_validation(self) -> None:
        extra = SideData({**{e: [0.0] * 5 for e in self.dev.side.episode_ids}, "SD9999-visible-L1": [0.0] * 5}, 5)
        with self.assertRaises(sel.SplitRefused):
            dataclasses.replace(self.dev, side=extra)
        with self.assertRaises(ValueError):
            dataclasses.replace(self.dev, side=self.dev.side.subset(self.dev.side.episode_ids[:-1]))


class CliTest(unittest.TestCase):
    def test_side_and_ranker_options(self) -> None:
        with tempfile.TemporaryDirectory() as d, quiet():
            d = Path(d)
            args = write_sim_set(d)
            side = write_side(d)
            quick = ["--cap-steps", "2", "--r-seeds", "1", "--k-sensitivity", "none", "--lam-sensitivity", "none",
                     "--bootstrap-B", "20", "--tau-q-grid", "0.5,0.6", "--tau-occ-grid", "0.3", "--tau-match-grid",
                     "0.5,0.6"]
            self.assertEqual(sel.main([*args, "--ranker", "A", *quick, "--out", str(d / "x")]), 2)   # needs --side
            rc = sel.main([*args, "--side", str(side), "--ranker", "A", "--ranker", "A-nocue", *quick,
                           "--out", str(d / "out")])
            self.assertIn(rc, (0, 4))
            cand = json.loads((d / "out" / "frozen_candidate.json").read_text())
            ho = cand["held_out_check"]
            self.assertEqual(ho["trainable_policies"], ["A", "A-nocue"])
            self.assertEqual(set(ho["trainable_modules_sha256"]), {"ranker.py", "policies.py", "side.py"})
            self.assertIn("side", cand["inputs"])


class ModelFileTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        f = rk.Fitted(pol.NOCUE_FEATURES, (0.1, 0.2, 0.3, 0.4), (1.0, 2.0, 1.0, 0.5), (0.5, -0.5, 1.0, 0.0), -0.2, 1.0,
                      {"1": 0.6}, 10, 3, 0.4)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            rk.save_model(p, "A-nocue", f, TH, {"note": "test"})
            r = rk.load_model(p)
        self.assertEqual((r.name, r.features, r.coef, r.intercept, r.tau_occ),
                         ("A-nocue", pol.NOCUE_FEATURES, f.coef, f.intercept, TH.tau_occ))


if __name__ == "__main__":
    unittest.main()


class PooledSideTest(unittest.TestCase):
    def test_pooled_sets_carry_both_environments_descriptors(self) -> None:
        from tests.test_select_pooled import TSLOTS, set_files, write_tabletop_set
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_sim_set(d / "shelf")
            write_tabletop_set(d / "tab")
            sides = (write_side(d / "shelf"), write_side(d / "tab", TSLOTS))
            dev = sel.sim_dev_data_pooled([(*set_files(d / "shelf"), sides[0]), (*set_files(d / "tab"), sides[1])])
            self.assertEqual(set(dev.side.episode_ids), {s.episode_id for s in dev.specs})
            self.assertEqual({k for k in dev.inputs if k.endswith("/side")}, {"shelf/side", "tabletop/side"})
            with quiet():
                res = sel.run_selection(dev, FAST, CTX, trainables=(rk.RankerA,))
            self.assertTrue([r for r in res["held_out_check"]["rows"] if r.policy == "A"])
            with self.assertRaises(ValueError):          # all sets with descriptors, or none
                sel.sim_dev_data_pooled([(*set_files(d / "shelf"), sides[0]), set_files(d / "tab")])
