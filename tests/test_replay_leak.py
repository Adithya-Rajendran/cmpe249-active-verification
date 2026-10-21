"""Leak and accounting tests for verify/replay.py (stdlib unittest). Run from the repo root:
uv run --no-sync python -m unittest tests.test_replay_leak -v

Every cheating attempt must raise, never return quietly. The policies here are minimal stand-ins
with the policies.py select() signature; the driving loop is the test's own (policies.py may not
exist yet, and its run_episode is tested in tests/test_policies.py).
"""
import dataclasses
import math
import unittest

from verify.replay import ReplayEnv
from verify.schema import (
    CONTINUE,
    OBSERVATION_FIELDS,
    Budget,
    Decision,
    EpisodeSpec,
    EvidenceRecord,
    EvidenceStore,
    GroundTruth,
    LeakError,
    Observation,
    Positions,
    EpisodeLog,
    ReplayError,
    ReplayKey,
    RunContext,
)

POS = Positions.default()
STEP = math.radians(30) * 45.0  # 23.56 cm


def make_world(n_arr: int = 1):
    """Synthetic specs/store/truth built directly from schema types (not the generator)."""
    store = EvidenceStore(5, synthetic=True, source="test_replay_leak")
    specs = []
    for a in range(1, n_arr + 1):
        for cond in ("visible", "occluded", "removed"):
            for light in ("L1", "L2"):
                s = EpisodeSpec(f"A{a:02d}", cond, light, "mug", "test")
                specs.append(s)
                for stop in range(5):  # distinct values per stop so leaks are detectable
                    store.put(s.episode_id, EvidenceRecord(stop, 0.1 * stop + 0.05, 0.9, 0.02 * stop))
    return specs, store, GroundTruth.from_specs(specs, synthetic=True)


def key(store, spec):
    return store.replay_key(spec.episode_id)


def new_env(budget=Budget(4), spec_idx=2):
    specs, store, truth = make_world()
    return ReplayEnv(key(store, specs[spec_idx]), store, POS, budget), specs[spec_idx], store, truth


def drive(policy, env, max_steps=10):
    """Minimal loop: ask the policy for a stop while candidates remain."""
    obs = [env.reset()]
    for _ in range(max_steps):
        cands = env.candidates()
        if not cands:
            break
        obs.append(env.request(policy.select(tuple(obs), cands, env.positions, env.current_stop)))
    return env.finish(Decision.UNKNOWN, policy=type(policy).__name__)


# ------------------------------------------------------------ cheating policies

class StorePeekPolicy:
    """Kept the store at construction and reads an unvisited stop's evidence."""
    name = "cheat-store"

    def __init__(self, store, eid):
        self.store, self.eid = store, eid

    def select(self, observations, candidates, positions, current_stop):
        best = max(candidates, key=lambda s: self.store.get(self.eid, s).match_score)
        return best


class EnvInternalsPolicy:
    """Kept the env and pokes at its internals for evidence/spec/truth."""
    name = "cheat-env"

    def __init__(self, env, attr):
        self.env, self.attr = env, attr

    def select(self, observations, candidates, positions, current_stop):
        getattr(self.env, self.attr)
        return candidates[0]


class ObservationTruthPolicy:
    """Looks for ground truth / ids on the observations it was given."""
    name = "cheat-obs"

    def __init__(self, attr):
        self.attr = attr

    def select(self, observations, candidates, positions, current_stop):
        getattr(observations[0], self.attr)
        return candidates[0]


class OffListPolicy:
    """Ignores candidates: returns a visited, out-of-range or unaffordable stop."""
    name = "cheat-offlist"

    def __init__(self, stop):
        self.stop = stop

    def select(self, observations, candidates, positions, current_stop):
        return self.stop


class TestCheatingPolicies(unittest.TestCase):
    def test_store_peek_raises_leak(self):
        specs, store, _ = make_world()
        env = ReplayEnv(key(store, specs[0]), store, POS, Budget(4))
        with self.assertRaises(LeakError):
            drive(StorePeekPolicy(store, specs[0].episode_id), env)
        self.assertEqual(env.visited, (2,))  # nothing was revealed beyond the initial view

    def test_store_reads_after_bind(self):
        env, spec, store, _ = new_env()
        self.assertTrue(store.sealed)
        for call in (lambda: store.get(spec.episode_id, 0), lambda: list(store.items()),
                     lambda: store.episode_ids(), lambda: store.put("X", EvidenceRecord(0, 0, 0, 0)),
                     lambda: store.to_jsonl("/dev/null")):
            with self.assertRaises(LeakError):
                call()

    def test_sealed_store_cannot_be_rebound_or_unsealed(self):
        # regression: a policy holding the store used to call store.bind(eid)(stop) on the sealed store,
        # or set store._sealed = False, and read unvisited evidence without any error
        env, spec, store, _ = new_env()
        for call in (lambda: store.bind(spec.episode_id), lambda: store.replay_key(spec.episode_id),
                     lambda: store.replay_keys(), lambda: store.bind(ReplayKey("0" * 32))):
            with self.subTest(call=call), self.assertRaises(LeakError):
                call()
        with self.assertRaises(AttributeError):
            store._sealed = False  # slotted: no such attribute to flip
        with self.assertRaises(AttributeError):
            store._data  # evidence is not one underscore away
        with self.assertRaises(LeakError):
            store.get(spec.episode_id, 4)

    def test_no_public_or_single_underscore_path(self):
        """The guarantee's boundary: every non-dunder, non-mangled name is an allowed accessor."""
        env, spec, store, _ = new_env()
        allowed_store = {"put", "get", "episode_ids", "items", "seal", "bind", "replay_key", "replay_keys",
                         "to_jsonl", "from_jsonl", "sealed", "n_stops", "synthetic", "source", "_check_open"}
        for obj, cls_name, allowed in ((store, "EvidenceStore", allowed_store),
                                       (env, "ReplayEnv", {"reset", "candidates", "request", "finish",
                                                           "positions", "budget", "current_stop", "visited",
                                                           "n_views", "n_additional_views", "travel_cm",
                                                           "remaining_travel_cm", "done"})):
            names = {n for n in dir(obj) if not n.startswith("__") and not n.startswith(f"_{cls_name}__")}
            self.assertEqual(names, allowed, cls_name)
        for attr in ("evidence", "store", "evidence_store", "spec", "truth", "ground_truth", "condition",
                     "episode_id", "key", "reveal", "_reveal", "_header", "_data", "_sealed", "__dict__"):
            env, *_ = new_env()
            with self.subTest(attr=attr), self.assertRaises(AttributeError):
                drive(EnvInternalsPolicy(env, attr), env)

    def test_introspection_is_out_of_scope(self):
        """Documents the boundary honestly: name-mangled slots are reachable by deliberate introspection.

        In-process Python cannot be sandboxed, so this is not protected at runtime; policy code is
        kept honest by review and by tests/test_policy_lint.py. If this test ever fails, the
        boundary moved and README section 2 must be updated.
        """
        env, *_ = new_env()
        self.assertTrue(callable(getattr(env, "_ReplayEnv__reveal")))
        self.assertFalse(hasattr(env, "_ReplayEnv__header"))  # no header slot: the env never held the condition

    def test_env_has_no_dict_and_rejects_new_attrs(self):
        env, *_ = new_env()
        with self.assertRaises(AttributeError):
            env.foo = 1
        with self.assertRaises(AttributeError):
            env.travel_cm = 0.0  # read-only property
        public = {n for n in dir(env) if not n.startswith("_")}
        self.assertEqual(public, {"reset", "candidates", "request", "finish", "positions", "budget",
                                  "current_stop", "visited", "n_views", "n_additional_views", "travel_cm",
                                  "remaining_travel_cm", "done"})

    def test_observation_has_no_truth(self):
        for attr in ("truth", "ground_truth", "condition", "episode_id", "arrangement_id", "path", "label"):
            env, *_ = new_env()
            with self.subTest(attr=attr), self.assertRaises(AttributeError):
                drive(ObservationTruthPolicy(attr), env)
        env, *_ = new_env()
        o = env.reset()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            o.match_score = 1.0

    def test_ground_truth_never_accepted(self):
        specs, store, truth = make_world()
        k = key(store, specs[0])
        with self.assertRaises(TypeError):
            ReplayEnv(k, truth, POS, Budget(2))
        for i in range(5):
            args = [k, store, POS, Budget(2), RunContext()]
            args[i] = truth
            with self.subTest(position=i), self.assertRaises(TypeError):
                ReplayEnv(*args)
        with self.assertRaises(TypeError):
            ReplayEnv(specs[0].episode_id, store, POS, Budget(2))  # an id string encodes the condition

    def test_env_never_receives_the_condition(self):
        # regression (LEAK-2): the env used to take the EpisodeSpec and keep its condition in a slot
        specs, store, _ = make_world()
        with self.assertRaises(TypeError):
            ReplayEnv(specs[0], store, POS, Budget(2))
        keys = store.replay_keys()
        self.assertEqual({f.name for f in dataclasses.fields(ReplayKey)}, {"token"})
        self.assertEqual(len({k.token for k in keys.values()}), len(specs))  # one opaque token per episode
        for eid, k in keys.items():
            for word in ("visible", "occluded", "removed", "A01", "L1", "L2", eid):
                self.assertNotIn(word, k.token)
            self.assertEqual(store.replay_key(eid), k)  # stable per episode
        env = ReplayEnv(keys[specs[5].episode_id], store, POS, Budget(1))
        env.reset()
        run = env.finish(Decision.UNKNOWN, policy="x")
        dumped = repr(run) + repr(dataclasses.asdict(run))
        for word in ("visible", "occluded", "removed", "A01", "episode"):
            self.assertNotIn(word, dumped)

    def test_off_list_requests_raise(self):
        # visited (start stop), out of range, bool, non-int
        for stop in (2, 5, -1, True, 1.0, "P1"):
            env, *_ = new_env()
            with self.subTest(stop=stop), self.assertRaises(ReplayError):
                drive(OffListPolicy(stop), env)
            self.assertEqual(env.visited, (2,))
            self.assertEqual(env.travel_cm, 0.0)
        # unaffordable under a 30 cm cap: P1 is 47.12 cm from P3
        env, *_ = new_env(Budget(4, 30.0))
        with self.assertRaises(ReplayError):
            drive(OffListPolicy(0), env)
        self.assertEqual((env.visited, env.travel_cm), ((2,), 0.0))

    def test_continue_or_string_is_not_a_decision(self):
        for d in (CONTINUE, "present", None):
            env, *_ = new_env()
            env.reset()
            with self.subTest(d=d), self.assertRaises(ReplayError):
                env.finish(d, policy="x")
            self.assertFalse(env.done)


class TestReplayRules(unittest.TestCase):
    def test_lifecycle_errors(self):
        env, *_ = new_env(Budget(1))
        self.assertEqual(env.candidates(), ())
        with self.assertRaises(ReplayError):
            env.request(1)  # before reset
        with self.assertRaises(ReplayError):
            env.finish(Decision.UNKNOWN, policy="x")  # before reset
        env.reset()
        with self.assertRaises(ReplayError):
            env.reset()
        env.request(1)
        with self.assertRaises(ReplayError):
            env.request(3)  # budget used up
        self.assertEqual(env.candidates(), ())
        env.finish(Decision.PRESENT, policy="x")
        self.assertTrue(env.done)
        self.assertEqual(env.candidates(), ())
        with self.assertRaises(ReplayError):
            env.request(3)  # after finish
        with self.assertRaises(ReplayError):
            env.finish(Decision.PRESENT, policy="x")

    def test_budget_exhaustion(self):
        for b in range(5):
            env, *_ = new_env(Budget(b))
            env.reset()
            while env.candidates():
                env.request(env.candidates()[0])
            self.assertEqual(env.n_additional_views, b)
            self.assertEqual(env.n_views, b + 1)
            with self.assertRaises(ReplayError):
                env.request(next(s for s in range(5) if s not in env.visited) if b < 4 else 0)

    def test_budget_zero_has_no_candidates(self):
        env, *_ = new_env(Budget(0))
        env.reset()
        self.assertEqual(env.candidates(), ())

    def test_candidates_travel_cap(self):
        env, *_ = new_env(Budget(4, 30.0))
        env.reset()
        self.assertEqual(env.candidates(), (1, 3))
        self.assertAlmostEqual(env.remaining_travel_cm, 30.0)
        env.request(1)
        self.assertAlmostEqual(env.remaining_travel_cm, 30.0 - STEP)
        self.assertEqual(env.candidates(), ())  # every unvisited stop is >= 23.56 cm from P2; only 6.44 cm left
        env2, *_ = new_env(Budget(4, 2 * STEP))  # exact boundary is affordable (EPS)
        env2.reset()
        self.assertEqual(env2.candidates(), (0, 1, 3, 4))
        env3, *_ = new_env(Budget(4, 0.0))
        env3.reset()
        self.assertEqual(env3.candidates(), ())
        env4, *_ = new_env(Budget(4, None))
        env4.reset()
        self.assertIsNone(env4.remaining_travel_cm)
        self.assertEqual(env4.candidates(), (0, 1, 3, 4))

    def test_travel_accounting(self):
        env, *_ = new_env(Budget(2))
        env.reset()
        self.assertEqual(env.travel_cm, 0.0)
        env.request(0)
        self.assertAlmostEqual(env.travel_cm, 47.12388980384690, delta=1e-6)
        self.assertAlmostEqual(env.travel_cm, 47.12, delta=0.005)
        env.request(1)
        self.assertAlmostEqual(env.travel_cm, 70.68583470577035, delta=1e-6)
        self.assertEqual(env.current_stop, 1)

    def test_f_tour_travel(self):
        env, *_ = new_env(Budget(4))
        env.reset()
        for s in POS.f_order:
            env.request(s)
        self.assertAlmostEqual(env.travel_cm, 141.3716694115407, delta=1e-6)

    def test_observations_and_log(self):
        env, spec, _, _ = new_env(Budget(3), spec_idx=4)
        seen = [env.reset(), env.request(4), env.request(3)]
        for step, o in enumerate(seen):
            self.assertIsInstance(o, Observation)
            self.assertEqual({f.name for f in dataclasses.fields(o)}, OBSERVATION_FIELDS)
            self.assertIn(o.stop, env.visited)
            self.assertEqual(o.step, step)
            self.assertEqual(o.azimuth_deg, POS.azimuth(o.stop))
            self.assertAlmostEqual(o.match_score, 0.1 * o.stop + 0.05)
        ctx = RunContext(model_id="m", config_version="v", config_sha256="abc", synthetic=True)
        specs, store, _ = make_world()
        env = ReplayEnv(key(store, specs[4]), store, POS, Budget(3, 100.0), ctx)
        seen = [env.reset(), env.request(4), env.request(3)]
        log = EpisodeLog.from_run(specs[4], env.finish(Decision.REMOVED, policy="G", seed=7))
        self.assertEqual(log.requested_stops, (2, 4, 3))
        self.assertEqual(log.evidence, tuple(seen))
        self.assertEqual({o.stop for o in log.evidence}, {2, 3, 4})  # nothing from unvisited stops
        self.assertEqual((log.n_views, log.n_additional_views), (3, 2))
        self.assertAlmostEqual(log.travel_cm, 3 * STEP)
        self.assertEqual((log.budget_additional_views, log.travel_cap_cm), (3, 100.0))
        self.assertEqual((log.model_id, log.config_version, log.config_sha256, log.synthetic),
                         ("m", "v", "abc", True))
        self.assertEqual((log.policy, log.seed, log.decision), ("G", 7, Decision.REMOVED))
        self.assertEqual(log.episode_id, specs[4].episode_id)
        d = log.to_dict()
        self.assertFalse({"truth", "ground_truth"} & set(d))

    def test_incomplete_episode_rejected(self):
        store = EvidenceStore(5)
        spec = EpisodeSpec("A01", "visible", "L1", "mug", "dev")
        for s in range(4):
            store.put(spec.episode_id, EvidenceRecord(s, 0.5, 0.5, 0.5))
        with self.assertRaises(ReplayError):
            store.replay_key(spec.episode_id)

    def test_second_env_on_sealed_store(self):
        specs, store, _ = make_world()
        keys = store.replay_keys()  # minted before the first bind seals the store
        a = ReplayEnv(keys[specs[0].episode_id], store, POS, Budget(1))
        b = ReplayEnv(keys[specs[1].episode_id], store, POS, Budget(1))
        self.assertNotEqual(a.reset().match_score, None)
        self.assertEqual(b.reset().stop, 2)


if __name__ == "__main__":
    unittest.main()
