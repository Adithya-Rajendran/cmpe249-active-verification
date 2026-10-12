"""Hand-checked tests for verify/policies.py (README section 3).

Uses the real verify.replay / verify.stopping when they exist. Otherwise it falls back to
minimal stand-ins defined here that follow the README interface, so this file passes on its own.
"""
from __future__ import annotations

import importlib.util
import unittest

import itertools

from verify.policies import (
    DEFAULT_R_SEEDS,
    GMYOPIC_LAM_DEG,
    EpisodeLimits,
    F,
    FullSweep,
    G,
    GMyopic,
    R,
    S,
    can_visit_more,
    default_name,
    r_seed_salt,
    run_episode,
    run_r_seeds,
)
from verify.schema import (
    CONTINUE,
    Budget,
    Condition,
    Decision,
    EpisodeLog,
    EpisodeRun,
    EpisodeSpec,
    EvidenceRecord,
    EvidenceStore,
    LeakError,
    Lighting,
    Observation,
    Positions,
    ReplayError,
    TRAVEL_TOL_CM,
    RunContext,
    Split,
    Stop,
    Thresholds,
    cap_cm_from_steps,
    derive_seed,
)

HAVE_REAL = all(importlib.util.find_spec(m) is not None for m in ("verify.replay", "verify.stopping"))

if HAVE_REAL:
    from verify.replay import ReplayEnv
    from verify.stopping import StoppingRule
else:
    class StoppingRule:  # minimal stand-in, README section 1
        def __init__(self, thresholds: Thresholds, k: int | None = None) -> None:
            self._th, self._k = thresholds, thresholds.k if k is None else k

        @property
        def k(self) -> int:
            return self._k

        @classmethod
        def first_clear_miss(cls, th: Thresholds) -> "StoppingRule":
            return cls(th, k=1)

        def decide(self, observations, can_continue):
            th = self._th
            if not observations:
                raise ValueError("empty")
            if any(o.match_score >= th.tau_match for o in observations):
                return Decision.PRESENT
            miss = {o.stop for o in observations if o.quality >= th.tau_q
                    and o.obstruction <= th.tau_occ and o.match_score < th.tau_match}
            if len(miss) >= self._k:
                return Decision.REMOVED
            return CONTINUE if can_continue else Decision.UNKNOWN

    class ReplayEnv:  # minimal stand-in, README section 2
        def __init__(self, key, evidence_store, positions, budget, context=RunContext()):
            self._reveal = evidence_store.bind(key)
            self.positions, self.budget, self._ctx = positions, budget, context
            self.visited: tuple[int, ...] = ()
            self.travel_cm, self._obs, self.done = 0.0, [], False

        @property
        def current_stop(self) -> int:
            return self.visited[-1]

        @property
        def n_additional_views(self) -> int:
            return len(self.visited) - 1

        def reset(self) -> Observation:
            self.visited = (self.positions.start_index,)
            self._obs = [Observation.from_record(self._reveal(self.visited[0]), 0, self.positions)]
            return self._obs[0]

        def candidates(self) -> tuple[int, ...]:
            if not self.visited or self.done or self.n_additional_views >= self.budget.additional_views:
                return ()
            cap = self.budget.travel_cap_cm
            return tuple(s for s in range(self.positions.n_stops) if s not in self.visited and
                         (cap is None or self.positions.travel_cm(self.current_stop, s) <= cap - self.travel_cm + 1e-9))

        def request(self, stop: int) -> Observation:
            if stop not in self.candidates():
                raise ReplayError(f"bad stop {stop}")
            self.travel_cm += self.positions.travel_cm(self.current_stop, stop)
            self.visited += (stop,)
            o = Observation.from_record(self._reveal(stop), len(self.visited) - 1, self.positions)
            self._obs.append(o)
            return o

        def finish(self, decision, *, policy, seed=None) -> EpisodeRun:
            if not isinstance(decision, Decision) or self.done:
                raise ReplayError("bad finish")
            self.done = True
            c = self._ctx
            return EpisodeRun(c.model_id, c.config_version, c.config_sha256, policy,
                              self.budget.additional_views, self.budget.travel_cap_cm, seed,
                              self.visited, tuple(self._obs), decision, self.travel_cm,
                              len(self.visited), self.n_additional_views, c.synthetic)


TH = Thresholds()
POS = Positions.default()
STEP = POS.nominal_step_cm()                       # 23.5619 cm
CAP1, CAP2, CAP3, CAP4 = (cap_cm_from_steps(n, POS) for n in (1, 2, 3, 4))
SPEC = EpisodeSpec("SYN01", Condition.OCCLUDED, Lighting.L1, "synthetic-target", Split.DEV)
EID = SPEC.episode_id

# per-stop evidence presets (match, quality, obstruction)
AMBIG = (0.1, 0.9, 0.9)      # flagged occluded, never a clear miss: no decision possible
CLEAR = (0.1, 0.9, 0.05)     # clear miss
UNFLAG = (0.1, 0.2, 0.05)    # not flagged, but bad quality: not a clear miss either
MATCH = (0.9, 0.9, 0.05)


def make_store(per_stop) -> EvidenceStore:
    store = EvidenceStore(5, synthetic=True, source="tests.test_policies")
    for s, (m, q, o) in enumerate(per_stop):
        store.put(EID, EvidenceRecord(s, m, q, o))
    return store


def make_env(per_stop, budget: Budget = Budget(4)) -> ReplayEnv:
    store = make_store(per_stop)
    return ReplayEnv(store.replay_key(EID), store, POS, budget, RunContext())


def obs(stop: int, obstruction: float, step: int = 0) -> Observation:
    return Observation(step, stop, POS.azimuth(stop), 0.1, 0.9, obstruction)


class SpyRule(StoppingRule):
    """Records the identity of every rule object whose decide() is called."""
    calls: list[int] = []

    def decide(self, observations, can_continue):
        SpyRule.calls.append(id(self))
        return super().decide(observations, can_continue)


class TestGMyopicSelect(unittest.TestCase):
    """The pre-D-G G, now the G-myopic ablation: every old hand-checked value must still hold."""

    def test_nothing_flagged_from_p3_ties_to_p1(self):
        g = GMyopic.from_thresholds(TH)
        o = [obs(2, 0.05)]
        self.assertAlmostEqual(g.score(0, (2,), POS, 2), 45.0, places=9)
        self.assertAlmostEqual(g.score(4, (2,), POS, 2), 45.0, places=9)
        self.assertEqual(g.select(o, (0, 1, 3, 4), POS, 2), 0)

    def test_second_step_from_p1_unflagged(self):
        g = GMyopic.from_thresholds(TH)
        o = [obs(2, 0.05), obs(0, 0.05, 1)]
        self.assertEqual(g.reference_stops(o), (0, 2))
        self.assertAlmostEqual(g.score(1, (0, 2), POS, 0), 22.5, places=9)
        self.assertAlmostEqual(g.score(3, (0, 2), POS, 0), 7.5, places=9)
        self.assertAlmostEqual(g.score(4, (0, 2), POS, 0), 30.0, places=9)
        self.assertEqual(g.select(o, (1, 3, 4), POS, 0), 4)

    def test_p3_flagged_p1_not(self):
        g = GMyopic.from_thresholds(TH)
        o = [obs(2, 0.9), obs(0, 0.05, 1)]
        self.assertEqual(g.reference_stops(o), (2,))
        self.assertEqual(g.select(o, (1, 3, 4), POS, 0), 4)

    def test_occluded_at_p3_lam0_ties_to_p1(self):
        g = GMyopic(0.0, TH.travel_norm_cm, TH.tau_occ)
        self.assertEqual(g.select([obs(2, 0.9)], (0, 1, 3, 4), POS, 2), 0)
        # with P1 unaffordable the symmetric partner P5 wins
        self.assertEqual(g.select([obs(2, 0.9)], (1, 3, 4), POS, 2), 4)

    def test_lam0_ref_is_flagged_p1_only(self):
        g = GMyopic(0.0, TH.travel_norm_cm, TH.tau_occ)
        o = [obs(2, 0.05), obs(0, 0.9, 1)]
        self.assertEqual(g.reference_stops(o), (0,))
        self.assertEqual([g.score(c, (0,), POS, 0) for c in (1, 3, 4)], [30.0, 90.0, 120.0])
        self.assertEqual(g.select(o, (1, 3, 4), POS, 0), 4)

    def test_travel_penalty_shifts_choice(self):
        o = [obs(2, 0.9), obs(0, 0.05, 1)]   # ref = {P3}, at P1
        self.assertEqual(GMyopic(0.0, TH.travel_norm_cm, TH.tau_occ).select(o, (1, 3, 4), POS, 0), 4)
        # lam = 100: P2 = 30 - 25 = 5, P4 = 30 - 75 = -45, P5 = 60 - 100 = -40 -> P2
        self.assertEqual(GMyopic(100.0, TH.travel_norm_cm, TH.tau_occ).select(o, (1, 3, 4), POS, 0), 1)

    def test_equal_score_prefers_shorter_travel(self):
        # lam = 0, at P3, ref = {P1, P5} flagged: P2 = P4 = 30 deg, both 23.56 cm -> P2 (index)
        g = GMyopic(0.0, TH.travel_norm_cm, TH.tau_occ)
        o = [obs(2, 0.05), obs(0, 0.9, 1), obs(4, 0.9, 2)]
        self.assertEqual(g.select(o, (1, 3), POS, 2), 1)
        # lam = 0, at P5, ref = {P3} flagged, P1 unaffordable: P2 = P4 = 30 deg;
        # P4 is 23.56 cm away, P2 is 70.69 cm -> P4 (shorter travel beats lower index)
        o = [obs(2, 0.9), obs(4, 0.05, 1)]
        self.assertEqual(g.select(o, (1, 3), POS, 4), 3)

    def test_select_respects_candidates(self):
        g = GMyopic.from_thresholds(TH)
        self.assertEqual(g.select([obs(2, 0.05)], (3,), POS, 2), 3)


    def test_legacy_lam_and_name(self):
        g = GMyopic.from_thresholds(TH)
        self.assertEqual((g.lam, g.travel_norm_cm, g.name), (30.0, TH.travel_norm_cm, "G-myopic"))
        self.assertEqual(GMYOPIC_LAM_DEG, 30.0)
        self.assertFalse(hasattr(g, "begin_episode"))   # no lookahead

    def test_myopic_goes_to_p1_under_two_step_cap(self):
        # the failure D-G fixes: P3 -> P1 spends the whole 2-step cap; REMOVED becomes unreachable
        per = [CLEAR, CLEAR, AMBIG, CLEAR, CLEAR]
        log = run_episode(GMyopic.from_thresholds(TH), make_env(per, Budget(3, CAP2)), StoppingRule(TH))
        self.assertEqual(log.requested_stops, (2, 0))
        self.assertIs(log.decision, Decision.UNKNOWN)


class TestF(unittest.TestCase):
    def test_full_order_without_early_decision(self):
        log = run_episode(F(), make_env([AMBIG] * 5, Budget(4)), StoppingRule(TH))
        self.assertEqual(log.requested_stops, (2, 0, 1, 3, 4))
        self.assertIs(log.decision, Decision.UNKNOWN)
        self.assertEqual(log.policy, "F")

    def test_skips_unaffordable(self):
        self.assertEqual(F().select([obs(2, 0.9)], (1, 3), POS, 2), 1)
        env = make_env([AMBIG] * 5, Budget(4, travel_cap_cm=30.0))
        log = run_episode(F(), env, StoppingRule(TH))
        self.assertEqual(log.requested_stops[:2], (2, 1))
        self.assertLessEqual(log.travel_cm, 30.0 + 1e-9)

    def test_skips_visited(self):
        self.assertEqual(F().select([obs(2, 0.9), obs(0, 0.9, 1)], (1, 3, 4), POS, 0), 1)

    def test_budget_limits_views(self):
        log = run_episode(F(), make_env([AMBIG] * 5, Budget(2)), StoppingRule(TH))
        self.assertEqual(log.requested_stops, (2, 0, 1))
        self.assertEqual((log.n_views, log.n_additional_views), (3, 2))

    def test_early_stop_on_removed(self):
        log = run_episode(F(), make_env([CLEAR] * 5, Budget(4)), StoppingRule(TH))
        self.assertEqual(log.requested_stops, (2, 0))
        self.assertIs(log.decision, Decision.REMOVED)


class TestR(unittest.TestCase):
    def test_reproducible_per_seed(self):
        seq = []
        for _ in range(2):
            r = R(123)
            seq.append([r.select([obs(2, 0.9)], (0, 1, 3, 4), POS, 2) for _ in range(30)])
        self.assertEqual(seq[0], seq[1])

    def test_only_candidates_and_all_reached(self):
        seen = set()
        for s in range(200):
            c = R(s).select([obs(2, 0.9)], (0, 3, 4), POS, 2)
            self.assertIn(c, (0, 3, 4))
            seen.add(c)
        self.assertEqual(seen, {0, 3, 4})

    def test_episode_paths_reproducible(self):
        a = run_episode(R(7), make_env([AMBIG] * 5, Budget(3)), StoppingRule(TH), seed=7)
        b = run_episode(R(7), make_env([AMBIG] * 5, Budget(3)), StoppingRule(TH), seed=7)
        self.assertEqual(a.requested_stops, b.requested_stops)
        self.assertEqual(a.seed, 7)

    def test_run_r_seeds_average_within_episode(self):
        # removed only if P1 and P5 are both seen with b = 2; elsewhere ambiguous
        per = [CLEAR, AMBIG, AMBIG, AMBIG, CLEAR]
        rule = StoppingRule(TH)
        logs, res = run_r_seeds(lambda: make_env(per, Budget(2)), rule, DEFAULT_R_SEEDS, EID, spec=SPEC)
        self.assertEqual(len(logs), 20)
        self.assertEqual([g.seed for g in logs], list(range(20)))
        self.assertTrue(all(g.policy == "R" for g in logs))
        n_rem = sum(g.decision is Decision.REMOVED for g in logs)
        self.assertAlmostEqual(res.frac(Decision.REMOVED), n_rem / 20)
        self.assertAlmostEqual(res.frac(Decision.REMOVED) + res.frac(Decision.UNKNOWN), 1.0)
        self.assertAlmostEqual(res.mean_travel_cm, sum(g.travel_cm for g in logs) / 20)
        # the per-seed paths match a direct run with the derived seed
        direct = run_episode(R(derive_seed(3, EID)), make_env(per, Budget(2)), rule, seed=3)
        self.assertEqual(direct.requested_stops, logs[3].requested_stops)

    def test_salt_changes_paths(self):
        paths = lambda salt: [run_r_seeds(lambda: make_env([AMBIG] * 5, Budget(2)), StoppingRule(TH),
                                          range(10), salt, spec=SPEC)[0][i].requested_stops for i in range(10)]
        self.assertNotEqual(paths("a"), paths("b"))

    def test_salt_is_condition_free_and_matched_episodes_share_draws(self):
        # regression (SPEC-6): the salt used to be episode_id, which carries the condition, so the
        # matched occluded and removed episodes of one arrangement drew different R paths
        specs = {c: EpisodeSpec("SYN01", c, Lighting.L1, "t", Split.TEST) for c in Condition}
        salts = {r_seed_salt(sp) for sp in specs.values()}
        self.assertEqual(len(salts), 1)
        salt = salts.pop()
        for c in Condition:
            self.assertNotIn(c.value, salt)
        self.assertNotEqual(salt, r_seed_salt(EpisodeSpec("SYN01", "removed", "L2", "t", "test")))
        paths = {}
        for c, sp in specs.items():
            logs, _ = run_r_seeds(lambda: make_env([AMBIG] * 5, Budget(3)), StoppingRule(TH), range(10),
                                  r_seed_salt(sp), spec=sp)
            paths[c] = [g.requested_stops for g in logs]
            self.assertTrue(all(g.episode_id == sp.episode_id and g.condition is c for g in logs))
        self.assertEqual(paths[Condition.OCCLUDED], paths[Condition.REMOVED])
        self.assertEqual(paths[Condition.VISIBLE], paths[Condition.REMOVED])


class TestS(unittest.TestCase):
    def test_never_moves(self):
        for per in ([AMBIG] * 5, [CLEAR] * 5, [MATCH] * 5):
            log = run_episode(S(), make_env(per, Budget(4)), StoppingRule(TH, k=1))
            self.assertEqual((log.n_views, log.n_additional_views, log.travel_cm), (1, 0, 0.0))
            self.assertEqual(log.requested_stops, (2,))
            self.assertEqual(log.policy, "S")

    def test_decisions(self):
        rule = StoppingRule(TH, k=1)
        self.assertIs(run_episode(S(), make_env([MATCH] * 5, Budget(0)), rule).decision, Decision.PRESENT)
        self.assertIs(run_episode(S(), make_env([CLEAR] * 5, Budget(0)), rule).decision, Decision.REMOVED)
        self.assertIs(run_episode(S(), make_env([AMBIG] * 5, Budget(0)), rule).decision, Decision.UNKNOWN)

    def test_select_raises(self):
        with self.assertRaises(RuntimeError):
            S().select([obs(2, 0.9)], (0,), POS, 2)

    def test_requires_k1_rule(self):
        # regression (API-2): S under the k = 2 rule silently returned unknown on an all-clear-miss episode
        with self.assertRaises(ReplayError):
            run_episode(S(), make_env([CLEAR] * 5, Budget(0)), StoppingRule(TH))
        self.assertIs(run_episode(S(), make_env([CLEAR] * 5, Budget(0)), StoppingRule(TH, k=1)).decision,
                      Decision.REMOVED)


class TestFullSweep(unittest.TestCase):
    def test_visits_all_five_and_decides_on_all(self):
        # P3 is a strong match, but FULL keeps going; the decision uses all 5 views
        per = [CLEAR, CLEAR, MATCH, CLEAR, CLEAR]
        log = run_episode(FullSweep(), make_env(per, Budget(4)), StoppingRule(TH))
        self.assertEqual(log.requested_stops, (2, 0, 1, 3, 4))
        self.assertEqual((log.n_views, len(log.evidence)), (5, 5))
        self.assertIs(log.decision, Decision.PRESENT)
        self.assertEqual(log.policy, "FULL")
        for per, want in (([AMBIG] * 5, Decision.UNKNOWN), ([CLEAR] * 5, Decision.REMOVED)):
            log = run_episode(FullSweep(), make_env(per, Budget(4)), StoppingRule(TH))
            self.assertEqual(log.n_views, 5)
            self.assertIs(log.decision, want)

    def test_decide_called_once_with_all_observations(self):
        seen = []

        class Rec(StoppingRule):
            def decide(self, observations, can_continue):
                seen.append((len(observations), can_continue))
                return super().decide(observations, can_continue)
        run_episode(FullSweep(), make_env([CLEAR] * 5, Budget(4)), Rec(TH))
        self.assertEqual(seen, [(5, False)])

    def test_needs_full_budget(self):
        with self.assertRaises(ReplayError):
            run_episode(FullSweep(), make_env([AMBIG] * 5, Budget(3)), StoppingRule(TH))

    def test_rejects_travel_cap(self):
        # regression (API-1): FULL under Budget(4, 50) used to stop after 2 stops and still log as "FULL"
        for cap in (50.0, 1000.0):
            with self.subTest(cap=cap), self.assertRaises(ReplayError):
                run_episode(FullSweep(), make_env([AMBIG] * 5, Budget(4, cap)), StoppingRule(TH))


class TestDriver(unittest.TestCase):
    def test_shared_stopping_object(self):
        SpyRule.calls = []
        rule = SpyRule(TH)
        policies = [F(), R(1), G.from_thresholds(TH), GMyopic.from_thresholds(TH), FullSweep()]
        for p in policies:
            run_episode(p, make_env([AMBIG] * 5, Budget(4)), rule)
        run_r_seeds(lambda: make_env([AMBIG] * 5, Budget(2)), rule, range(3), EID, spec=SPEC)
        self.assertGreaterEqual(len(SpyRule.calls), len(policies) + 3)
        self.assertEqual(set(SpyRule.calls), {id(rule)})
        for p in policies:  # policies hold no rule of their own (G holds only EpisodeLimits)
            self.assertFalse(any(isinstance(v, StoppingRule) for v in vars(p).values()))

    def test_fcm_names(self):
        fcm = StoppingRule.first_clear_miss(TH)
        self.assertEqual(default_name(G.from_thresholds(TH), fcm), "G-fcm")
        self.assertEqual(default_name(F(), fcm), "F-fcm")
        self.assertEqual(default_name(S(), fcm), "S")
        self.assertEqual(default_name(G.from_thresholds(TH), StoppingRule(TH)), "G")
        log = run_episode(G.from_thresholds(TH), make_env([AMBIG] * 5, Budget(1)), fcm)
        self.assertEqual(log.policy, "G-fcm")
        log = run_episode(F(), make_env([AMBIG] * 5, Budget(1)), fcm, name="custom")
        self.assertEqual(log.policy, "custom")

    def test_g_episode_path(self):
        # nothing flagged anywhere (bad quality, no match): P3 -> P1 -> P5 -> ...
        log = run_episode(G.from_thresholds(TH), make_env([UNFLAG] * 5, Budget(2)), StoppingRule(TH))
        self.assertEqual(log.requested_stops, (2, 0, 4))
        self.assertIs(log.decision, Decision.UNKNOWN)

    def test_g_respects_travel_cap(self):
        log = run_episode(G.from_thresholds(TH), make_env([UNFLAG] * 5, Budget(4, 30.0)), StoppingRule(TH))
        self.assertEqual(log.requested_stops[1], 1)
        self.assertLessEqual(log.travel_cm, 30.0 + 1e-9)

    def test_choice_outside_candidates_raises(self):
        class Bad(F):
            def select(self, observations, candidates, positions, current_stop):
                return 2   # the start stop, already visited

        class NotInt(F):
            def select(self, observations, candidates, positions, current_stop):
                return "P1"
        for p in (Bad(), NotInt()):
            with self.assertRaises(ReplayError):
                run_episode(p, make_env([AMBIG] * 5), StoppingRule(TH))

    def test_select_gets_only_observations(self):
        got = []

        class Probe(F):
            def select(self, observations, candidates, positions, current_stop):
                got.append((observations, candidates, positions, current_stop))
                return super().select(observations, candidates, positions, current_stop)
        run_episode(Probe(), make_env([AMBIG] * 5, Budget(2)), StoppingRule(TH))
        for observations, candidates, positions, current_stop in got:
            self.assertIsInstance(observations, tuple)
            self.assertTrue(all(type(o) is Observation for o in observations))
            self.assertIsInstance(positions, Positions)
            self.assertIsInstance(current_stop, int)
            self.assertTrue(candidates)

    def test_cheating_policy_fails_loudly(self):
        store = make_store([AMBIG] * 5)

        class CheatingPolicy(F):
            def __init__(self, store):
                self.store = store

            def select(self, observations, candidates, positions, current_stop):
                self.store.get(EID, 4)   # unvisited evidence
                return candidates[0]
        env = ReplayEnv(store.replay_key(EID), store, POS, Budget(4), RunContext())
        with self.assertRaises(LeakError):
            run_episode(CheatingPolicy(store), env, StoppingRule(TH))

    def test_cheating_policy_cannot_rebind_store(self):
        # regression: store.bind(eid) on the sealed store used to hand any holder a reveal function
        attacks = {"bind-id": lambda st: st.bind(EID)(4),
                   "mint-and-bind": lambda st: st.bind(st.replay_key(EID))(4)}
        for label, attack in attacks.items():
            store = make_store([AMBIG] * 5)

            class Rebinder(F):
                def select(self, observations, candidates, positions, current_stop, st=store, a=attack):
                    a(st)
                    return candidates[0]
            env = ReplayEnv(store.replay_key(EID), store, POS, Budget(4), RunContext())
            with self.subTest(attack=label), self.assertRaises(LeakError):
                run_episode(Rebinder(), env, StoppingRule(TH))

    def test_cheating_policy_cannot_list_store(self):
        store = make_store([AMBIG] * 5)

        class Lister(F):
            def select(self, observations, candidates, positions, current_stop):
                list(store.items())
                return candidates[0]
        env = ReplayEnv(store.replay_key(EID), store, POS, Budget(4), RunContext())
        with self.assertRaises(LeakError):
            run_episode(Lister(), env, StoppingRule(TH))

    @unittest.skipUnless(HAVE_REAL, "needs the real verify.replay")
    def test_cheating_policy_cannot_reach_truth_through_env(self):
        env = make_env([AMBIG] * 5)
        for attr in ("spec", "store", "truth", "ground_truth", "condition", "evidence", "key"):
            with self.assertRaises(AttributeError):
                getattr(env, attr)


def limits(budget: Budget, k: int = 2) -> EpisodeLimits:
    return EpisodeLimits.of(budget, StoppingRule(TH, k=k))


def g_bound(budget: Budget, k: int = 2, lam: float = TH.lam, **kw) -> G:
    g = G(lam, TH.tau_occ, **kw)
    g.begin_episode(limits(budget, k))
    return g


def brute_force_can_visit(start, visited, n, travel_left, views_left, pos) -> bool:
    """Independent oracle: try every ordered n-tuple of unvisited stops (itertools)."""
    if n <= 0:
        return True
    if n > views_left:
        return False
    free = [s for s in range(pos.n_stops) if s not in visited]
    for seq in itertools.permutations(free, n):
        if travel_left is None or pos.path_travel_cm((start, *seq)) <= travel_left + TRAVEL_TOL_CM:
            return True
    return False


class TestCapSteps(unittest.TestCase):
    def test_nominal_step_and_caps(self):
        self.assertAlmostEqual(STEP, 23.561944901923447, places=12)
        self.assertAlmostEqual(STEP, POS.travel_cm(2, 3), places=12)
        self.assertEqual(POS.adjacent_step_deviation_cm(), 0.0)
        self.assertAlmostEqual(CAP2, POS.travel_cm(2, 0), places=12)     # 2 steps == P3 -> P1 exactly
        self.assertAlmostEqual(cap_cm_from_steps(6, POS), POS.span_cm() * 1.5, places=9)
        self.assertIsNone(cap_cm_from_steps(None, POS))
        self.assertIsNone(cap_cm_from_steps(float("inf"), POS))
        with self.assertRaises(ValueError):
            cap_cm_from_steps(-1, POS)

    def test_env_affordability_at_exact_multiples(self):
        # no slack: a 2-step cap affords P3 -> P1 (47.12 cm) and nothing after it
        env = make_env([AMBIG] * 5, Budget(4, CAP2))
        env.reset()
        self.assertEqual(env.candidates(), (0, 1, 3, 4))
        env.request(0)
        self.assertEqual(env.candidates(), ())
        env = make_env([AMBIG] * 5, Budget(4, CAP3))   # 3 steps: both neighbours, P2 then P4
        env.reset()
        env.request(1)
        self.assertIn(3, env.candidates())

    def test_env_and_lookahead_share_travel_tolerance(self):
        # integration: the env's candidates() and G's lookahead both use TRAVEL_TOL_CM (D-cap)
        hop = POS.travel_cm(2, 0)
        env = make_env([AMBIG] * 5, Budget(4, hop - 0.5 * TRAVEL_TOL_CM))
        env.reset()
        self.assertIn(0, env.candidates())
        env = make_env([AMBIG] * 5, Budget(4, hop - 2 * TRAVEL_TOL_CM))
        env.reset()
        self.assertNotIn(0, env.candidates())
        self.assertTrue(can_visit_more(2, {2}, 1, hop - 0.5 * TRAVEL_TOL_CM, 4, POS) is True
                        and can_visit_more(2, {2, 1, 3}, 1, hop - 2 * TRAVEL_TOL_CM, 4, POS) is False)


class TestFeasibilityEnumeration(unittest.TestCase):
    def test_exact_not_greedy(self):
        # at P2, visited {P3, P2}, need 2 more stops within 3 steps:
        # nearest-first (P1 then P4) costs 1 + 3 = 4 steps and fails; P4 then P5 costs 2 + 1 = 3 steps
        self.assertTrue(can_visit_more(1, {2, 1}, 2, CAP3, 4, POS))
        self.assertFalse(can_visit_more(1, {2, 1}, 2, CAP3 - 1e-3, 4, POS))
        self.assertFalse(can_visit_more(1, {2, 1}, 3, CAP3, 4, POS))      # P1, P4, P5 needs >= 5 steps
        self.assertTrue(can_visit_more(1, {2, 1}, 3, 5 * STEP, 4, POS))   # P1 -> P4 -> P5 = 1 + 3 + 1

    def test_view_budget_and_trivial_cases(self):
        self.assertFalse(can_visit_more(1, {2, 1}, 2, None, 1, POS))      # only 1 view left
        self.assertTrue(can_visit_more(1, {2, 1}, 2, None, 2, POS))
        self.assertTrue(can_visit_more(0, {2, 0}, 0, 0.0, 0, POS))        # nothing needed
        self.assertFalse(can_visit_more(0, {0, 1, 2, 3}, 2, None, 4, POS))  # only P5 is left

    def test_tolerance_is_1e6_cm(self):
        self.assertEqual(TRAVEL_TOL_CM, 1e-6)
        self.assertTrue(can_visit_more(2, {2}, 1, STEP - 5e-7, 4, POS))
        self.assertFalse(can_visit_more(2, {2}, 1, STEP - 2e-6, 4, POS))

    def test_matches_brute_force_everywhere(self):
        n_checked = 0
        for start in range(5):
            others = [s for s in range(5) if s != start]
            for r in range(len(others) + 1):
                for extra in itertools.combinations(others, r):
                    visited = {start, *extra}
                    for n in range(5):
                        for views in range(5):
                            for steps in (0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, None):
                                left = None if steps is None else steps * STEP
                                self.assertEqual(can_visit_more(start, visited, n, left, views, POS),
                                                 brute_force_can_visit(start, visited, n, left, views, POS),
                                                 (start, visited, n, views, steps))
                                n_checked += 1
        self.assertEqual(n_checked, 5 * 16 * 5 * 5 * 11)


class TestGCapAware(unittest.TestCase):
    """D-G: reachability feasibility. Hand-checked on the design geometry (23.56 cm per step)."""

    def test_two_step_cap_k2_avoids_p1(self):
        # P3 flagged occluded, so no clear miss yet: needed = k - 0 = 2. Cap 2 steps = 47.12 cm.
        #   P1: travel 47.12, left 0.00 -> the 1 more clear-miss stop it needs is unreachable -> infeasible
        #   P2: travel 23.56, left 23.56 -> P1 reachable (23.56)                              -> feasible
        #   P4: travel 23.56, left 23.56 -> P5 reachable                                      -> feasible
        #   P5: travel 47.12, left 0.00                                                       -> infeasible
        o = [obs(2, 0.9)]
        g = g_bound(Budget(3, CAP2))
        self.assertEqual([g.feasible(c, o, POS, 2) for c in (0, 1, 3, 4)], [False, True, True, False])
        # pool {P2, P4}: sep 30, 30 -> 1, 1; travel 23.56, 23.56 -> 1, 1; scores 0.5, 0.5 -> tie -> P2
        self.assertEqual(g.select(o, (0, 1, 3, 4), POS, 2), 1)
        self.assertEqual(g.last_pool, (1, 3))
        self.assertFalse(g.last_fallback)

    def test_two_step_cap_k2_exact_path(self):
        # evidence: P3 ambiguous (flagged), every other stop a clear miss
        per = [CLEAR, CLEAR, AMBIG, CLEAR, CLEAR]
        g = run_episode(G.from_thresholds(TH), make_env(per, Budget(3, CAP2)), StoppingRule(TH))
        # P3 -> P2 (clear miss 1; only P1 is still affordable) -> P1 (clear miss 2) -> REMOVED
        self.assertEqual(g.requested_stops, (2, 1, 0))
        self.assertIs(g.decision, Decision.REMOVED)
        self.assertAlmostEqual(g.travel_cm, CAP2, places=9)
        # F and G-myopic both spend the cap on P3 -> P1 and are stuck at UNKNOWN
        f = run_episode(F(), make_env(per, Budget(3, CAP2)), StoppingRule(TH))
        m = run_episode(GMyopic.from_thresholds(TH), make_env(per, Budget(3, CAP2)), StoppingRule(TH))
        for log in (f, m):
            self.assertEqual(log.requested_stops, (2, 0))
            self.assertIs(log.decision, Decision.UNKNOWN)

    def test_three_step_cap_allows_p1(self):
        # 3 steps: P3 -> P1 leaves 1 step, so P2 is still reachable: everything feasible; P1 wins (sep 1 - 0.5)
        o = [obs(2, 0.9)]
        g = g_bound(Budget(3, CAP3))
        self.assertEqual([g.feasible(c, o, POS, 2) for c in (0, 1, 3, 4)], [True, True, True, True])
        self.assertEqual(g.select(o, (0, 1, 3, 4), POS, 2), 0)
        per = [CLEAR, CLEAR, AMBIG, CLEAR, CLEAR]
        log = run_episode(G.from_thresholds(TH), make_env(per, Budget(3, CAP3)), StoppingRule(TH))
        self.assertEqual(log.requested_stops, (2, 0, 1))
        self.assertIs(log.decision, Decision.REMOVED)

    def test_k3_four_step_cap(self):
        # needed 3: c + 2 more. P1 (2 steps) leaves 2: P1->P2->P4 = 1 + 2 = 3 > 2 -> infeasible.
        # P2 (1 step) leaves 3: P2->P4->P5 = 2 + 1 = 3 -> feasible. Symmetric for P4 / P5.
        o = [obs(2, 0.9)]
        g = g_bound(Budget(3, CAP4), k=3)
        self.assertEqual([g.feasible(c, o, POS, 2) for c in (0, 1, 3, 4)], [False, True, True, False])
        g3 = g_bound(Budget(3, CAP3), k=3)            # 3 steps: nothing feasible
        self.assertEqual([g3.feasible(c, o, POS, 2) for c in (0, 1, 3, 4)], [False] * 4)

    def test_clear_miss_so_far_counts(self):
        # P3 is already a clear miss: needed = 1, the move itself may supply it -> all feasible at 1 step
        o = [Observation(0, 2, 0.0, 0.1, 0.9, 0.05)]
        g = g_bound(Budget(3, CAP1))
        self.assertEqual([g.feasible(c, o, POS, 2) for c in (1, 3)], [True, True])

    def test_view_budget_counts(self):
        # uncapped, b = 1, nothing clear yet: after the one move no view is left for the 2nd clear miss
        o = [obs(2, 0.9)]
        g = g_bound(Budget(1))
        self.assertEqual([g.feasible(c, o, POS, 2) for c in (0, 1, 3, 4)], [False] * 4)
        g2 = g_bound(Budget(2))
        self.assertEqual([g2.feasible(c, o, POS, 2) for c in (0, 1, 3, 4)], [True] * 4)

    def test_fallback_when_nothing_feasible(self):
        # 1-step cap, k = 2, P3 flagged: P2 / P4 leave 0 cm, so no candidate is feasible -> ordinary choice
        o = [obs(2, 0.9)]
        g = g_bound(Budget(3, CAP1))
        self.assertEqual(g.select(o, (1, 3), POS, 2), 1)
        self.assertTrue(g.last_fallback)
        self.assertEqual(g.last_pool, (1, 3))
        nolook = g_bound(Budget(3, CAP1), lookahead=False)
        self.assertEqual(nolook.select(o, (1, 3), POS, 2), 1)
        # the fallback can still find PRESENT: the target is visible from P2
        per = [AMBIG, MATCH, AMBIG, AMBIG, AMBIG]
        log = run_episode(G.from_thresholds(TH), make_env(per, Budget(3, CAP1)), StoppingRule(TH))
        self.assertEqual(log.requested_stops, (2, 1))
        self.assertIs(log.decision, Decision.PRESENT)

    def test_fcm_rule_makes_everything_feasible(self):
        # k = 1: needed - 1 = 0 more stops, so the lookahead never filters (G-fcm == G-nolook-fcm)
        o = [obs(2, 0.9)]
        g = G.from_thresholds(TH)
        g.begin_episode(limits(Budget(3, CAP2), k=1))
        self.assertEqual([g.feasible(c, o, POS, 2) for c in (0, 1, 3, 4)], [True] * 4)
        self.assertEqual(g.select(o, (0, 1, 3, 4), POS, 2), 0)

    def test_select_needs_limits(self):
        with self.assertRaises(ReplayError):
            G.from_thresholds(TH).select([obs(2, 0.9)], (0, 1, 3, 4), POS, 2)

    def test_driver_binds_env_budget_and_rule_k(self):
        g = G.from_thresholds(TH)
        run_episode(g, make_env([AMBIG] * 5, Budget(2, CAP3)), StoppingRule(TH, k=3))
        self.assertEqual((g.limits.additional_views, g.limits.travel_cap_cm, g.limits.k), (2, CAP3, 3))
        # F and R stay naive: no lookahead hook at all
        for p in (F(), R(0), S(), FullSweep()):
            self.assertFalse(hasattr(p, "begin_episode"))

    def test_lookahead_off_name(self):
        self.assertEqual(G.from_thresholds(TH).name, "G")
        self.assertEqual(G.from_thresholds(TH, lookahead=False).name, "G-nolook")
        self.assertEqual(default_name(G.from_thresholds(TH), StoppingRule.first_clear_miss(TH)), "G-fcm")
        self.assertEqual(default_name(GMyopic.from_thresholds(TH), StoppingRule.first_clear_miss(TH)),
                         "G-myopic-fcm")


class TestGNormalisation(unittest.TestCase):
    """D-norm: sep / max_sep - lam * travel / max_travel over the pool; 0/0 = 0; lam dimensionless."""

    def test_hand_scores_from_p3(self):
        g = G(0.5, TH.tau_occ)
        # pool P1, P2, P4, P5; ref {P3}: sep 60, 30, 30, 60 -> 1, .5, .5, 1; travel 2, 1, 1, 2 steps -> 1, .5, .5, 1
        sc = g.scores((0, 1, 3, 4), (2,), POS, 2)
        for c, want in ((0, 0.5), (1, 0.25), (3, 0.25), (4, 0.5)):
            self.assertAlmostEqual(sc[c], want, places=12)
        self.assertEqual(TH.lam, 0.5)

    def test_lam_grid_changes_choice(self):
        o = [obs(2, 0.9)]
        picks = {}
        for lam in (0.0, 0.25, 0.5, 1.0):
            g = g_bound(Budget(4), lam=lam)
            picks[lam] = g.select(o, (0, 1, 3, 4), POS, 2)
        # lam 1: every score is 0 -> tie -> shorter travel -> P2; lam < 1: P1 (P5 ties, higher index)
        self.assertEqual(picks, {0.0: 0, 0.25: 0, 0.5: 0, 1.0: 1})

    def test_normalised_over_pool_not_candidates(self):
        # from P1 with ref {P3}: pool (1, 3, 4) -> sep 30, 30, 60 -> .5, .5, 1; travel 1, 3, 4 steps -> .25, .75, 1
        g = G(0.5, TH.tau_occ)
        sc = g.scores((1, 3, 4), (2,), POS, 0)
        self.assertAlmostEqual(sc[1], 0.5 - 0.125, places=12)
        self.assertAlmostEqual(sc[3], 0.5 - 0.375, places=12)
        self.assertAlmostEqual(sc[4], 1.0 - 0.5, places=12)
        sc2 = g.scores((1, 3), (2,), POS, 0)   # drop P5: maxima are now 30 deg and 3 steps
        self.assertAlmostEqual(sc2[1], 1.0 - 0.5 / 3, places=12)
        self.assertAlmostEqual(sc2[3], 1.0 - 0.5, places=12)

    def test_zero_over_zero_is_zero(self):
        # degenerate geometry: all stops at one azimuth -> every sep and travel is 0 -> scores 0, no error
        flat = Positions(tuple(Stop(i, f"P{i + 1}", 0.0) for i in range(5)), 45.0, 2, (0, 1, 3, 4))
        g = G(0.5, TH.tau_occ)
        self.assertEqual(g.scores((0, 1, 3, 4), (2,), flat, 2), {0: 0.0, 1: 0.0, 3: 0.0, 4: 0.0})
        g.begin_episode(limits(Budget(4)))
        self.assertEqual(g.select([obs(2, 0.9)], (0, 1, 3, 4), flat, 2), 0)   # all tied -> lowest index

    def test_rejects_negative_lam(self):
        with self.assertRaises(ValueError):
            G(-0.1, TH.tau_occ)


if __name__ == "__main__":
    unittest.main()
