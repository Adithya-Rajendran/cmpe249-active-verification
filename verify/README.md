# verify/: offline verification core (interface contract)

This is the contract for the offline replay, stopping rule, policies, scoring and synthetic
generator. The code runs on the GB10, not the Pi. It is pure Python 3.13 plus numpy, with
no new dependencies. Until study data exist, everything runs on **synthetic** episodes, and
every synthetic output is labelled as such (`schema.SYNTHETIC_LABEL`). Synthetic numbers
are not results.

The protocol is `docs/EXPERIMENT_PLAN.md`. Where this file and the protocol disagree, the
protocol wins; report the conflict, don't resolve it silently.

Builders work in parallel and don't talk to each other. Each one writes only its own files
and **imports only what this file names**. All shared types are already implemented in
`verify/schema.py`. Don't edit `schema.py`. If you need a change there, say so in your notes.

| Module | Owner writes | May import from |
| --- | --- | --- |
| `verify/schema.py` | done (contract author) | stdlib |
| `verify/manifest.py` | manifest builder | schema |
| `verify/replay.py` | replay builder | schema |
| `verify/stopping.py` | stopping builder | schema |
| `verify/policies.py` | policies builder | schema, stopping, replay |
| `verify/synthetic.py` | synthetic builder | schema, numpy |
| `analysis/metrics.py`, `analysis/bootstrap.py` | analysis builder | schema, numpy |
| `verify/run.py` | run builder | everything above |
| `analysis/intervals.py` | select builder | stdlib only |
| `verify/lamgrid.py` | select builder (review F1) | schema, policies |
| `verify/select.py` | select builder | schema, stopping, replay, policies, lamgrid, synthetic, analysis (+ numpy) |
| `tests/test_<module>.py` | the builder of that module | the module under test + schema (+ numpy) |

Tests use stdlib `unittest` and run from the repo root:

```
uv run --no-sync python -m unittest discover -s tests -t . -v
```

Imports are absolute from the repo root (`from verify.schema import ...`,
`from analysis.metrics import compute`). Each test file has to pass on its own. Where a
test needs another builder's module, it either builds the inputs directly with `schema` types
or skips with `unittest.skipUnless(importlib.util.find_spec(...))`. It never copies that
module's logic.

Style follows `capture/pi_capture.py`: small typed functions, frozen dataclasses where they
make sense, short docstrings, no frameworks, no prints in library code.

---

## 0. Fixed facts (already encoded in `schema.py`)

- **Stops** are indices `0..4` = P1..P5, at azimuths -60, -30, 0, +30, +60 deg on an arc of
  r = 45 cm. `Positions.travel_cm(i, j) = radians(|az_i - az_j|) * r`, so one step is
  23.56 cm. The start stop is P3 (`start_index = 2`). `f_order = (0, 1, 3, 4)`. F's full tour
  is 141.37 cm. `Positions.span_cm()` = 94.25 cm (P1 to P5) is the default
  `Thresholds.travel_norm_cm` (used only by the G-myopic ablation). `Positions.nominal_step_cm()`
  = (max azimuth - min azimuth) x r / (n_stops - 1) = 23.5619 cm on the design values; it is the
  unit of travel caps (D-cap). `Positions.adjacent_step_deviation_cm()` is the largest
  |adjacent travel - nominal step| (0 on the design geometry; `verify.run` warns in `table.md`
  when it exceeds `TRAVEL_TOL_CM`, because then N steps no longer buy exactly N adjacent moves).
  `Positions.load()` reads `configs/positions.json` (since 2026-10-09 the measured shelf arc:
  r = 90 cm, stops -45/-22.5/0/+22.5/+45 deg, 35.34 cm per step; `Positions.default()` still returns
  the design values above without touching the file).
  `Positions.from_dict` raises `ValueError` unless `schema == "cmpe249.positions.v1"`
  (`schema.POSITIONS_SCHEMA`), and `to_dict` writes that key. `tests/test_positions.py` checks that
  the file loads to the measured shelf geometry and that the default is unchanged. `verify.run` loads the file (`--positions`) and hashes
  its path and sha256 into the config. Unit tests use `Positions.default()`.
- **Episode** = (arrangement, condition in {visible, occluded, removed}, lighting in {L1, L2}).
  That gives 6 per arrangement. `schema.episode_id(arr, cond, light)` returns `"A01-occluded-L2"`.
- **Truth** is `TruthState.PRESENT` (visible or occluded) or `TruthState.REMOVED`. `UNKNOWN`
  is a `Decision`, never a truth.
- **`EpisodeSpec`** has `arrangement_id, condition, lighting, target_id, split`, and no truth
  field. Because `condition` implies the truth, **a spec is never given to a policy**.
- **`GroundTruth`** (`states: episode_id -> TruthState`, `indistinguishable: frozenset[episode_id]`,
  `synthetic`) is used for scoring only. It is never passed to `ReplayEnv`, a policy or the
  stopping rule. `GroundTruth.from_specs(specs, indistinguishable_arrangements)` marks the
  occluded and removed episodes of those arrangements. Visible episodes stay distinguishable.
- **`EvidenceRecord(stop, match_score, quality, obstruction)`**: all three scores are in [0, 1]
  and none is a calibrated probability.
- **`Observation(step, stop, azimuth_deg, match_score, quality, obstruction)`** is all a
  policy ever sees. It has no paths, ids or truth. `step` 0 is the initial view.
  `schema.OBSERVATION_FIELDS` is the whitelist.
- **`EvidenceStore`** is the only container of evidence (`put`, `get`, `items`,
  `episode_ids`, `to_jsonl`, `from_jsonl`). It is slotted, and its state is in name-mangled
  slots (no `_data` or `_sealed` one underscore away). While unsealed it mints opaque
  **`ReplayKey(token)`** handles: `replay_key(episode_id)` (stable per episode, `ReplayError` if
  the episode doesn't have records for all 5 stops) and `replay_keys()` (all episodes). The token
  is random and encodes nothing about the episode. The **first `bind(key)` seals the store**.
  After that, `get`, `items`, `episode_ids`, `put`, `to_jsonl`, `replay_key` and `replay_keys`
  raise `LeakError`, and so does `bind` with anything other than a key this store minted
  before sealing (a plain episode id, a forged key). So a policy that holds the sealed store
  cannot mint a key and rebind. Mint every key the run needs before building the first env.
  `bind` returns a private `reveal(stop) -> EvidenceRecord` for the keyed episode.
- **`Thresholds(tau_match=0.6, tau_q=0.5, tau_occ=0.3, k=2, lam=0.5, travel_norm_cm=94.2477...)`**:
  these defaults are synthetic placeholders. The real values are chosen on dev by
  `verify/select.py` (section 10) and frozen. **`lam` is G's dimensionless travel weight** (D-norm,
  section 3; selection grid {0, 0.7, 0.9, 1.2}, see section 10; 0.5 is a placeholder,
  *proposed default - confirm*). **At the design geometry lam = 0.5 behaves exactly like lam = 0**
  (G's choice changes only at lam = 2/3, 3/4, 1, 4/3, 3/2, 2; `verify/lamgrid.py`), so runs at the
  placeholder have no effective travel penalty. `lam` and `travel_norm_cm` must be finite
  (NaN or inf raise `ValueError`; review F3). `travel_norm_cm` is read only by G-myopic.
- **`Budget(additional_views: 0..4, travel_cap_cm: float | None = None)`**: the initial view
  is always taken. It counts in `n_views` but not in `n_additional_views`.
- **Travel caps (D-cap)** are an operating condition reported as curves, never tuned. They are
  given in arc steps: `CAP_STEPS_DEFAULT = (1, 2, 3, 4, 6, inf)` (inf = uncapped);
  `cap_cm_from_steps(steps, positions)` = steps x `nominal_step_cm()` with **no slack**
  (None or inf -> None), and `cap_steps_from_cm` is its inverse for labels. Every
  affordability comparison (the env's `candidates()` and G's lookahead) uses
  `TRAVEL_TOL_CM = 1e-6` cm and nothing else. `EPS = 1e-9` is for score and rate comparisons only.
- **`RunContext(model_id, config_version, config_sha256, synthetic)`** is copied into every log.
- **`EpisodeRun`** is what `ReplayEnv.finish` returns: the run's provenance, policy, budget,
  seed, requested stops, revealed evidence, decision, travel and view counts, with **no episode
  header** (no id, arrangement, condition or lighting). The driver joins the header outside the
  env with `EpisodeLog.from_run(spec, run)`.
- **`EpisodeLog`** is one run (one seed) on one episode, with the EXPERIMENT_PLAN "Logs"
  fields and no truth. **`EpisodeResult.from_logs(logs)`** aggregates the runs of one
  (episode, policy, budget, cap) into `decision_frac: {Decision: fraction}` plus mean views,
  mean additional views and mean travel. `schema.as_results(items)` accepts logs, results or
  a mix.
- Helpers: `config_sha256(obj)` (sha256 of `canonical_json`), `file_sha256`,
  `derive_seed(*parts)` (stable 63-bit int), `to_jsonable`,
  `split_arrangements(ids, n_dev, seed) -> {arr: Split}`, `truth_from_condition`.
- Errors: `LeakError` (RuntimeError), `ReplayError` (ValueError), `ManifestError` (ValueError).
- `CONTINUE` (`Continue.CONTINUE`, value `"continue"`) is the stopping rule's non-terminal
  output. It is not a `Decision`.

---

## 1. `verify/stopping.py`

```python
class StoppingRule:
    def __init__(self, thresholds: Thresholds, k: int | None = None): ...
        # k defaults to thresholds.k; ValueError if k < 1.
    k: int                      # read-only property
    thresholds: Thresholds      # read-only property
    name: str                   # "shared-k{k}", e.g. "shared-k2"; "shared-k1" is first-clear-miss
    @classmethod
    def first_clear_miss(cls, thresholds: Thresholds) -> "StoppingRule": ...   # k = 1
    def is_strong_match(self, o: Observation) -> bool: ...    # o.match_score >= tau_match
    def is_clear_miss(self, o: Observation) -> bool: ...
        # o.quality >= tau_q and o.obstruction <= tau_occ and o.match_score < tau_match
    def is_flagged_occluded(self, o: Observation) -> bool: ... # o.obstruction > tau_occ
    def clear_miss_stops(self, observations) -> frozenset[int]: ...  # distinct stops with a clear miss
    def decide(self, observations: Sequence[Observation], can_continue: bool) -> Decision | Continue: ...
```

`decide` rules, applied in this order:

1. Empty `observations` raises `ValueError`.
2. If **any** observation is a strong match, return `Decision.PRESENT`. PRESENT takes precedence
   over REMOVED.
3. If `len(clear_miss_stops(observations)) >= k`, return `Decision.REMOVED`. Stops are counted
   **distinct**: two observations of the same stop count once.
4. If `can_continue`, return `CONTINUE`.
5. Otherwise return `Decision.UNKNOWN`.

The rule is stateless and deterministic. It never multiplies scores, and it never reads
anything except the observations it is given. The boundaries are inclusive exactly as written:
`>= tau_match`, `>= tau_q`, `<= tau_occ`, and flagged means `> tau_occ`.

**Truth table** (required in `tests/test_stopping.py`, with the default thresholds 0.6/0.5/0.3).
Each `obs` entry below is `(stop, match, quality, obstruction)`.

| # | obs | k | can_continue | expected |
| --- | --- | --- | --- | --- |
| 1 | (2, 0.60, 0.9, 0.0) | 2 | True | PRESENT (boundary: 0.60 >= 0.6) |
| 2 | (2, 0.59, 0.9, 0.0) | 2 | True | CONTINUE (one clear miss < k) |
| 3 | (2, 0.59, 0.9, 0.0) | 2 | False | UNKNOWN |
| 4 | (2, 0.1, 0.9, 0.1), (0, 0.1, 0.9, 0.1) | 2 | True | REMOVED |
| 5 | (2, 0.1, 0.9, 0.1), (2, 0.1, 0.9, 0.1) | 2 | False | UNKNOWN (same stop twice) |
| 6 | (2, 0.1, 0.9, 0.1), (0, 0.1, 0.9, 0.1), (4, 0.7, 0.9, 0.9) | 2 | False | PRESENT (precedence) |
| 7 | (2, 0.1, 0.49, 0.1), (0, 0.1, 0.9, 0.1) | 2 | False | UNKNOWN (bad quality is not a clear miss) |
| 8 | (2, 0.1, 0.5, 0.3), (0, 0.1, 0.9, 0.1) | 2 | False | REMOVED (boundaries 0.5 and 0.3 qualify) |
| 9 | (2, 0.1, 0.9, 0.31), (0, 0.1, 0.9, 0.1) | 2 | True | CONTINUE (0.31 is flagged occluded, not a clear miss) |
| 10 | (2, 0.1, 0.9, 0.1) | 1 | True | REMOVED (first-clear-miss) |
| 11 | (2, 0.1, 0.9, 0.9) | 1 | True | CONTINUE |
| 12 | (2, 0.1, 0.9, 0.9) | 1 | False | UNKNOWN |
| 13 | [] | 2 | True | ValueError |

Also test that `is_flagged_occluded` and `is_clear_miss` are never both True for the same
observation.

---

## 2. `verify/replay.py`

The replay is the **only** holder of evidence during a run.

```python
class ReplayEnv:
    __slots__ = (...)   # required: no instance __dict__, no ad hoc attributes
    def __init__(self, key: ReplayKey, evidence_store: EvidenceStore, positions: Positions,
                 budget: Budget, context: RunContext = RunContext()): ...
    def reset(self) -> Observation: ...
    def candidates(self) -> tuple[int, ...]: ...
    def request(self, stop: int) -> Observation: ...
    def finish(self, decision: Decision, *, policy: str, seed: int | None = None) -> EpisodeRun: ...
    # read-only properties:
    positions: Positions
    budget: Budget
    current_stop: int               # last visited stop (start stop after reset)
    visited: tuple[int, ...]        # request order, includes the start stop
    n_views: int                    # len(visited)
    n_additional_views: int         # n_views - 1
    travel_cm: float                # cumulative commanded travel
    remaining_travel_cm: float | None   # None when travel_cap_cm is None
    done: bool                      # True after finish()
```

Semantics:

- `__init__` takes an opaque `ReplayKey`, **never the spec**: the spec's condition and its
  episode id (`"A01-removed-L1"`) both determine the truth. It raises `TypeError` if any
  argument is a `GroundTruth` or an `EpisodeSpec`, if `key` isn't a `ReplayKey` (a plain id string
  is rejected), or if `evidence_store` isn't an `EvidenceStore`. It calls
  `evidence_store.bind(key)`, which seals the store, and keeps only the returned `reveal`
  function in a private slot. It never keeps a reference to the store, and it holds nothing
  that encodes the condition.
- `reset()` visits `positions.start_index` with zero travel and returns `Observation(step=0, ...)`.
  Calling it a second time raises `ReplayError`.
- `candidates()` returns the stops, in ascending index order, that are **unvisited** and
  **affordable**: `n_additional_views < budget.additional_views`, and when there is a cap,
  `travel_cm(current, s) <= remaining_travel_cm + schema.TRAVEL_TOL_CM`. It returns `()` before
  `reset`, after `finish`, or when the view budget is used up.
- `request(stop)` raises `ReplayError` if the env isn't reset, is already done, or if `stop`
  isn't an int in range, is already visited, or isn't in `candidates()` (over the view budget
  or the travel cap). Otherwise it adds travel, appends to `visited`, and returns
  `Observation.from_record(reveal(stop), step=len(visited) - 1, positions)`. A rejected
  request changes no state.
- `finish(decision, policy=..., seed=...)` requires a `Decision` (`CONTINUE` and plain strings
  raise `ReplayError`). It may be called only once, sets `done`, and returns the header-free
  `EpisodeRun`. Its `requested_stops` and `evidence` cover exactly the visited stops in
  order, `budget_additional_views` and `travel_cap_cm` come from the budget, and the
  provenance comes from `context` (`synthetic = context.synthetic`). The driver adds the
  header with `EpisodeLog.from_run(spec, run)`.
- The public names on an instance (`[n for n in dir(env) if not n.startswith("_")]`) are
  exactly the methods and properties listed above.

**Guarantee and its boundary.** Runtime protection covers every *public or single-underscore*
path: no such name on the env or the sealed store leads to unvisited evidence, the condition or
the truth, and every direct read of a sealed store raises `LeakError`. Deliberate introspection
is **out of scope**, because in-process Python cannot be sandboxed: name-mangled slots
(`env._ReplayEnv__reveal`), closures (`__closure__`), frame walking (`sys._getframe`), `gc`, or
opening files. A policy written to do that would succeed. Two things cover the gap instead:
(1) `tests/test_policy_lint.py` statically bans introspection, I/O and private-attribute access in
`verify/policies.py` (imports are whitelisted; `gc`, `sys`, `inspect`, `open`, `getattr`,
`__closure__`, mangled names and the like fail the test), and policy code is reviewed;
(2) nothing truth-bearing is on disk while policies run: `verify.run` writes `episodes.json`,
`evidence.jsonl` and `ground_truth.json` only after the run stage (section 8). The separation of
truth is therefore procedural. `condition` and `episode_id` are still log header fields (the
protocol asks for them) and the driver holds the specs, but neither reaches the env or a policy.
`tests/test_replay_leak.py::test_introspection_is_out_of_scope` documents the boundary; if it
ever fails, update this paragraph.

**Leak test** (`tests/test_replay_leak.py`, required; each case must raise, not return quietly):

1. After the env is built, `store.get(eid, s)`, `store.items()` and `store.episode_ids()`
   raise `LeakError`.
2. A `CheatingPolicy` that is given the store at construction and calls
   `store.get(<eid>, <unvisited stop>)` inside `select()` makes `run_episode` raise
   `LeakError`. The test lives in `tests/test_policies.py`; `run_episode` must not swallow
   exceptions.
3. `env.evidence`, `env.store`, `env.spec`, `env.truth`, `env.ground_truth`, `env.condition`,
   `env.key`, `env._reveal`, `env._header`, `store._data` and `store._sealed` raise `AttributeError`,
   and `env.foo = 1` and `store._sealed = False` raise `AttributeError` (slots). Every non-dunder,
   non-mangled name on the env and the store is in an explicit allowlist. A policy holding the
   sealed store that calls `store.bind(eid)`, or mints a key and binds it, gets `LeakError`.
4. `ReplayEnv(key, truth_obj, ...)` raises `TypeError`, and so does passing a `GroundTruth`
   as any argument. `ReplayEnv(spec, ...)` and `ReplayEnv(episode_id_str, ...)` raise `TypeError`
   (the env never receives the condition); a `ReplayKey` has only a random `token`, and a
   finished `EpisodeRun` contains no id or condition string.
5. `request` on a visited stop, an unaffordable stop (travel cap), past the view budget, before
   `reset`, and after `finish` each raise `ReplayError`.
6. Every `Observation` returned has fields equal to `OBSERVATION_FIELDS`, and its `stop` is in
   `visited`.
7. Hand-checked travel: start P3, request P1 then P2 (b=2, no cap) gives travel 47.12 + 23.56
   = 70.69 cm (to 1e-6). With cap 30 cm at P3, `candidates() == (1, 3)`.

---

## 3. `verify/policies.py`

```python
class Policy(Protocol):
    name: str                       # used in logs: "S", "F", "R", "G", "G-myopic", "FULL" (+ variants, below)
    max_additional_views: int | None  # 0 for S; None = use the env budget
    early_stop: bool                # False only for FullSweep
    def select(self, observations: Sequence[Observation], candidates: Sequence[int],
               positions: Positions, current_stop: int) -> int: ...
```

`select` gets only these four arguments. `candidates` is non-empty and comes from
`env.candidates()`. The return value must be in `candidates`; `run_episode` checks this and
raises `ReplayError` otherwise. Ranker A (later) implements the same protocol. The features
it needs, such as reference visibility per stop, will be passed to its constructor, which
keeps the protocol unchanged.

Implementations:

- `S()`: `name="S"`, `max_additional_views=0`. `select` raises `RuntimeError` (it is never called).
- `F()`: `name="F"`. Returns the first stop of `positions.f_order` that is in `candidates`.
- `R(seed: int)`: `name="R"`. `self.rng = numpy.random.default_rng(seed)` is created in
  `__init__`. Returns `int(rng.choice(sorted(candidates)))`. Each (episode, seed) run gets a
  **fresh** `R`.
- `G(lam: float, tau_occ: float, *, lookahead: bool = True)`: `name="G"` (`"G-nolook"` when
  `lookahead=False`). `G.from_thresholds(th, *, lookahead=True)` returns
  `G(th.lam, th.tau_occ, lookahead=...)`; this is the only way `verify.run` and `verify.select`
  build G. Per decision:
  1. `ref` = the visited stops whose observation is flagged occluded (`obstruction > tau_occ`),
     or all visited stops if none is flagged (`reference_stops(observations, tau_occ)`).
  2. **Reachability feasibility (D-G).** Candidate `c` is feasible if, after moving to `c`, the
     episode could still collect enough distinct clear-miss stops to reach REMOVED:
     `needed = k - #distinct clear-miss stops so far`; `c` may supply one; the other `needed - 1`
     must be reachable as distinct unvisited stops within the remaining travel cap **and** the
     remaining view budget. This is `can_visit_more(start, visited, n, travel_left_cm,
     views_left, positions)`, an exact depth-first enumeration of ordered paths over the 5 stops
     (at most 4! orders), checked with `TRAVEL_TOL_CM`; it is tested against an itertools oracle
     and is not greedy. It is optimistic about evidence (it assumes future views could be clear
     misses) and reads no evidence. If no candidate is feasible, G falls back to **all**
     affordable candidates (it can still find PRESENT). Under the k = 1 rule every candidate is
     feasible, so G-fcm never filters.
  3. **Per-decision normalisation (D-norm).** Over the pool (feasible set, or all affordable on
     fallback): `score(c) = sep(c) / max_sep - lam * travel(c) / max_travel`, with
     `sep(c) = min(angle_deg(c, v) for v in ref)`, both maxima over the pool, and 0/0 = 0.
  4. Argmax. Scores within `schema.EPS` tie; ties go to the shorter travel, then the lower index.

  G learns k, the rule's `is_clear_miss` and the env's public `Budget` through the optional hook
  `begin_episode(limits: EpisodeLimits)`, which `run_episode` calls once per episode on any policy
  that defines it (only G does). `EpisodeLimits(additional_views, travel_cap_cm, k, is_clear_miss)`
  is built by `EpisodeLimits.of(env.budget, rule)`, so G counts clear misses exactly as the shared
  rule does and never holds a second copy of the thresholds. `select` keeps its four arguments,
  G still sees no unvisited evidence and no truth, and the decision still comes only from
  `rule.decide`. `G.select` without `begin_episode` raises `ReplayError`. **The lookahead is part
  of G's selection intelligence, not a shared rule: F and R stay naive** (they choose from the
  affordable set exactly as specified) and the stopping rule is unchanged.

  *Geometric note (holds on any data at the design geometry).* From P3 the reference set is
  {P3}, so G's first move is pure geometry: P1/P5 score 1 - lam and P2/P4 score 0.5 - 0.5 lam.
  For lam < 1 G goes to P1, as F does; at lam = 1 the scores tie exactly and the tie-break
  sends G to P2 (so lam = 1 behaves like lam > 1). More generally G's behaviour is piecewise
  constant in lam: `verify.lamgrid.g_lam_breakpoints(positions)` enumerates every decision state G
  can be in on a geometry and returns the lam values where any choice changes. At the design
  geometry: 2/3, 3/4, 1, 4/3, 3/2 and 2. These are geometric facts, not synthetic results; they
  move when measured azimuths replace the template, so re-derive them then
  (`python -m verify.select --lam-breakpoints`). Under a 3-step cap, P3 -> P1 leaves exactly one step, so F and G then coincide;
  under a 2-step cap with k = 2 the lookahead rejects P1 (no second stop reachable) and G differs
  from F.
- `GMyopic(lam: float, travel_norm_cm: float, tau_occ: float)`: `name="G-myopic"`. The pre-D-G G,
  kept unchanged as an ablation: no lookahead, no per-decision normalisation,
  `score(c) = min angle(c, ref) - lam * travel(c) / travel_norm_cm` over all affordable
  candidates, same ties. `GMyopic.from_thresholds(th, lam_deg=GMYOPIC_LAM_DEG)` uses
  `GMYOPIC_LAM_DEG = 30.0` (degrees per `travel_norm_cm`), because `Thresholds.lam` now means
  G's dimensionless weight. G vs G-myopic changes lookahead **and** normalisation at once;
  `G(lookahead=False)` ("G-nolook") isolates the lookahead but is not in the default run
  (*proposed default - confirm*). A receding-horizon "G-h" variant was not built: scoring future
  steps needs the obstruction of views not yet taken, which would require an invented observation
  model.
- `FullSweep()`: `name="FULL"`, `early_stop=False`. Visits the remaining stops in F order: returns the first
  `f_order` stop in `candidates`. It is run with `Budget(4, None)`. `run_episode`
  raises `ReplayError` if the env's budget is below 4 **or has a travel cap** (a capped sweep
  would stop early and still be logged as `FULL`).
- The **first-clear-miss variant** is not a separate class. It is the same selector driven by
  `StoppingRule.first_clear_miss(th)`. Its log name is `f"{policy.name}-fcm"` (e.g. `"G-fcm"`),
  which `run_episode` sets when `rule.k == 1` and the policy isn't S.

```python
def run_episode(policy: Policy, env: ReplayEnv, rule: StoppingRule, *, seed: int | None = None,
                name: str | None = None) -> EpisodeRun: ...
```

It returns the header-free `EpisodeRun`; the caller joins the header with
`EpisodeLog.from_run(spec, run)`.

The driving loop is the only place a policy and an env meet:

```
if hasattr(policy, "begin_episode"):        # G only: public budget + the rule's k and predicate
    policy.begin_episode(EpisodeLimits.of(env.budget, rule))
obs = [env.reset()]
while True:
    cap_ok = policy.max_additional_views is None or env.n_additional_views < policy.max_additional_views
    can_continue = cap_ok and len(env.candidates()) > 0
    if not policy.early_stop and can_continue:
        decision = CONTINUE                 # FullSweep: keep going while stops remain
    else:
        decision = rule.decide(tuple(obs), can_continue)
    if decision is not CONTINUE:
        break
    stop = policy.select(tuple(obs), env.candidates(), env.positions, env.current_stop)
    obs.append(env.request(stop))
return env.finish(decision, policy=name or <default name>, seed=seed)
```

**All policies use the one `StoppingRule` object the caller passes in.** The caller builds
`StoppingRule(th)` (k = th.k) for F, R, G and FULL, and `StoppingRule(th, k=1)` for S. S has
one view, so k = 2 could never return REMOVED; the protocol's S is "removed on a clear
estimated region with no match". S is reported at its actual effort (1 view, 0 travel) and is
budget-independent. Run it once, with `Budget(0)`. `run_episode` enforces this: S with a rule
whose `k != 1` raises `ReplayError` (under k = 2 it could never say removed, but would still be
logged as `S`). `verify.run` passes S the same first-clear-miss object it uses for the `-fcm`
variants.

```python
def r_seed_salt(spec: EpisodeSpec) -> str: ...   # f"{arrangement_id}|{lighting}"
def run_r_seeds(make_env: Callable[[], ReplayEnv], rule: StoppingRule, seeds: Sequence[int],
                seed_salt: str, *, spec: EpisodeSpec,
                name: str | None = None) -> tuple[list[EpisodeLog], EpisodeResult]: ...
```

For each base seed `s`, this runs `run_episode(R(derive_seed(s, seed_salt)), make_env(), rule,
seed=s, name=name)` and joins the header from `spec` (which never reaches the env or the
policy). The caller passes `r_seed_salt(spec)`, which is **condition-free**: the matched
visible, occluded and removed episodes of one arrangement and lighting share R's random stream
(common random numbers), so they draw the same stops whenever their candidate sets agree, and
the salt is not a function of a label-bearing string. Different arrangements and lightings get
different paths. The policy only ever sees an int. The function returns the per-seed logs and
`EpisodeResult.from_logs(logs)`, which averages within the episode. The default number of seeds
is 20 (`range(20)`).

Required hand-checked tests (`tests/test_policies.py`). The legacy scores below are now
asserted on **G-myopic** (lam = 30 deg, norm = 94.2478), whose behaviour is unchanged:

- F from P3 with b = 4 and no early decision visits 2, 0, 1, 3, 4. With cap 30 cm, from P3
  the candidates are (1, 3), and F picks 1.
- G-myopic from P3 with nothing flagged: P1 scores 60 - 30 * 47.12 / 94.25 = 45.0, and so does P5.
  The tie has equal travel, so it goes to the lower index, **P1 (0)**. Next, from P1 with
  visited {2, 0} unflagged: P2 = 30 - 7.5 = 22.5, P4 = 30 - 22.5 = 7.5, P5 = 60 - 30 = 30.
  The pick is **P5 (4)**.
- G-myopic with the P3 view flagged and the P1 view not flagged, at P1: ref = {2}. P2 = 30 - 7.5,
  P4 = 30 - 22.5, P5 = 60 - 30 = 30, so the pick is **P5**.
- G-myopic with lam = 0, current stop P1, visited P3 (not flagged) then P1 (flagged): ref = {0}.
  P2 = 30, P4 = 90, P5 = 120, so the pick is **P5**.
- G (D-norm) hand scores from P3, and lam grid effects; normalisation over the pool, not over
  all candidates; 0/0 = 0; negative, NaN and inf lam rejected (`tests/test_lamgrid.py` also pins
  the design-geometry lam breakpoints and checks that the enumerated states cover every state G
  reaches on synthetic episodes).
- G (D-G): under a 2-step cap with k = 2 and b >= 2, G avoids P1 from P3 (and G-myopic goes to
  P1); a 3-step cap allows P1; k = 3 with a 4-step cap; clear misses so far reduce `needed`; the
  view budget counts; fallback when nothing is feasible; k = 1 makes everything feasible; the
  driver passes the env budget and the rule's k. `can_visit_more` matches a brute-force oracle,
  uses a 1e-6 cm tolerance, and is exact rather than greedy (at P2 needing 2 more stops within
  3 steps, nearest-first P1 -> P4 costs 4 steps, but P4 -> P5 costs 3 and fits).
- R with the same seed gives the same sequence. Over many seeds, every candidate gets chosen.
- S never calls `select` and logs `n_views == 1`, `travel_cm == 0`; S with a k = 2 rule raises.
- FullSweep always logs `n_views == 5` and decides on all 5 observations; with a travel cap it raises.
- `r_seed_salt` is the same for all three conditions of an arrangement and lighting, and the
  matched episodes get identical R paths when their evidence keeps the rule continuing.
- The cheating-policy `LeakError` case from section 2.

---

## 4. `verify/manifest.py`

```python
@dataclass(frozen=True)
class ArrangementRecord:
    arrangement_id: str            # ^[A-Z][A-Z0-9]{1,15}$ (same as capture's RE_ARRANGEMENT)
    target_id: str
    split: Split
    lightings: tuple[Lighting, ...] = (Lighting.L1, Lighting.L2)
    indistinguishable: bool = False

def episodes_from_arrangements(arrs: Sequence[ArrangementRecord]) -> tuple[list[EpisodeSpec], GroundTruth]: ...
def load_manifest(path: str | Path) -> tuple[list[EpisodeSpec], GroundTruth]: ...
def save_manifest(arrs: Sequence[ArrangementRecord], path: str | Path) -> None: ...
def assign_splits(arrangement_ids: Sequence[str], n_dev: int, seed: int) -> dict[str, Split]: ...
def validate(specs: Sequence[EpisodeSpec], truth: GroundTruth | None = None) -> None: ...
def filter_split(specs: Sequence[EpisodeSpec], split: Split) -> list[EpisodeSpec]: ...
def summary(specs: Sequence[EpisodeSpec]) -> dict[str, int]: ...
```

- `episodes_from_arrangements` produces 3 conditions x the given lightings per arrangement,
  sorted by (arrangement_id, condition order visible/occluded/removed, lighting). Truth comes
  from `GroundTruth.from_specs(specs, [a.arrangement_id for a in arrs if a.indistinguishable])`.
- `assign_splits` delegates to `schema.split_arrangements`.
- The manifest JSON (`configs/arrangements.json` later) looks like this. Only
  `load_manifest`/`save_manifest` read or write it.
  `{"schema": "cmpe249.arrangements.v1", "arrangements": [{"arrangement_id", "target_id", "split", "lightings": ["L1","L2"], "indistinguishable": false}]}`.
- `validate` raises `ManifestError` (with an informative message) if:
  - an episode_id is duplicated;
  - an arrangement appears in more than one split (**the split is by arrangement only**);
  - an arrangement's target_id differs between its episodes;
  - an arrangement doesn't have exactly {visible, occluded, removed} x its lightings, all with
    the same lighting set (6 with L1+L2, or 3 with L1 only);
  - `truth` is given and its keys aren't exactly the spec ids;
  - `truth` disagrees with `truth_from_condition`;
  - an indistinguishable id is a visible episode;
  - an arrangement id fails the regex.
- `summary` returns `{"n_arrangements", "n_episodes", "n_dev_arrangements",
  "n_test_arrangements", "n_dev_episodes", "n_test_episodes"}`.
- Tests: 6 episodes per arrangement; validate catches each failure above; a round trip
  through save and load is identity; the split is never mixed.

---

## 5. `verify/synthetic.py`

```python
@dataclass(frozen=True)
class SyntheticParams:
    n_dev: int | None = None            # None -> round(n_arrangements / 3)
    indistinguishable_frac: float = 0.1 # arrangements whose occluder blocks all 5 stops (at least 1 if n >= 3)
    block_widths: tuple[int, ...] = (1, 2, 3, 4)      # contiguous blocked-stop count (distinguishable arrangements)
    block_width_probs: tuple[float, ...] = (0.25, 0.35, 0.25, 0.15)
    block_includes_start: bool = True   # the occluder always hides the start stop P3 (see the caveat below)
    edge_obstruction: tuple[float, float] = (0.15, 0.45)   # U range at stops adjacent to the block
    blocked_obstruction: tuple[float, float] = (0.5, 0.95) # U range at blocked stops
    clear_obstruction: tuple[float, float] = (0.0, 0.15)   # U range elsewhere (and everywhere in visible)
    match_present: tuple[float, float] = (0.80, 0.08)      # N(mean, sd) for an unblocked present target
    match_absent: tuple[float, float] = (0.15, 0.08)       # N for removed targets / blocked present targets
    match_edge_present: tuple[float, float] = (0.55, 0.15) # N at edge stops when present
    quality_L1: tuple[float, float] = (0.85, 0.05)         # N
    quality_L2: tuple[float, float] = (0.60, 0.12)         # N (L2 lowers quality; some views fail tau_q)
    l2_match_drop: float = 0.08                            # subtracted from present-match means under L2
    target_id: str = "synthetic-target"

@dataclass(frozen=True)
class OccluderLayout:
    arrangement_id: str
    blocked: tuple[int, ...]   # contiguous, ascending stop indices
    indistinguishable: bool    # True iff blocked == all stops

def generate_layouts(n_arrangements: int, seed: int, params: SyntheticParams = SyntheticParams(),
                     n_stops: int = 5, start_index: int = 2) -> dict[str, OccluderLayout]: ...
def generate(n_arrangements: int, seed: int, params: SyntheticParams = SyntheticParams(),
             positions: Positions | None = None) -> tuple[list[EpisodeSpec], EvidenceStore, GroundTruth]: ...
```

- The arrangement ids are `SYN01`, `SYN02`, and so on (zero-padded to 2 digits, or 3 when
  n >= 100). Splits come from `schema.split_arrangements(ids, n_dev, seed)`. Specs are in the
  same order as `manifest.episodes_from_arrangements`, but the module doesn't import manifest.
- RNG: `numpy.random.default_rng(seed)`. `generate_layouts` consumes it first, and `generate`
  reproduces the identical layouts by calling `generate_layouts` with the same seed. The
  evidence then uses `default_rng(derive_seed("evidence", seed))`. The same (n, seed, params)
  gives identical output.
- **Matched occluder layouts**: the occluded and removed episodes of an arrangement share the
  same `blocked` set, at both lightings. Without that, the occluder itself would reveal the
  label. Visible episodes have the occluder present but not blocking (D5): `clear_obstruction`
  at every stop.
- The indistinguishable arrangements (`blocked = (0..4)`) are chosen first, without
  replacement. In their occluded and removed episodes no view can match or show a clear miss,
  so UNKNOWN is the intended answer. The `GroundTruth.indistinguishable` field marks these
  occluded and removed episodes.
- Per view (all values clipped to [0, 1]):
  - obstruction is `blocked_obstruction` at blocked stops, `edge_obstruction` at stops next to
    the block, and `clear_obstruction` elsewhere. That applies to the occluded and removed
    conditions; visible uses clear everywhere.
  - match is `match_present` (minus `l2_match_drop` under L2) at unblocked non-edge stops when
    present, `match_edge_present` at edge stops when present, and `match_absent` at blocked
    stops or when removed.
  - quality is `quality_L1` or `quality_L2`.
- **Caveat: the default regime favours G and dooms S.** With `block_includes_start=True` every
  distinguishable occluder is one contiguous block containing P3, and obstruction falls off with
  distance from the block. That is exactly G's max-angular-separation prior, and S's single P3
  view is flagged occluded in every occluded and removed episode (S recall 0 by construction).
  S rows and the G - S, F - S and G - F comparisons on this regime reflect the generator, not the
  policies. `verify.run --block-anywhere` runs the second regime (`block_includes_start=False`,
  block placed anywhere); report both. `table.md` prints the caveat for whichever regime ran.
- Labelling: the store is `EvidenceStore(5, synthetic=True, source="verify.synthetic seed=<seed>")`
  and the truth has `synthetic=True`. The module docstring states that the generator parameters
  are invented for testing the pipeline and say nothing about real perception.
- Tests: determinism; 6 episodes per arrangement; matched layouts (obstruction is high at the
  same stops in occluded and removed); the indistinguishable subset exists and has all 5 stops
  blocked; L2 mean quality < L1; nothing in the store is outside [0, 1]; the store isn't sealed
  on return.

---

## 6. `analysis/metrics.py`

```python
METRICS = ("false_removal_rate", "removal_recall", "coverage", "resolved_accuracy",
           "mean_views", "mean_additional_views", "mean_travel_cm")
def compute(results: Iterable[EpisodeLog | EpisodeResult], truth: GroundTruth,
            group_by: str | None = None) -> dict: ...
def compute_all(results, truth) -> dict: ...   # {"overall": ..., "by_condition": ..., "by_lighting": ...}
```

- Inputs are normalised with `schema.as_results`. The results must be for **one policy and
  one budget**; mixed input raises `ValueError`. Every `episode_id` must be in `truth`, or
  `KeyError` is raised. Truth is read only here and in bootstrap.
- Decisions are fractional: each episode contributes `frac(d)` to each decision (R is
  averaged within the episode).
- `group_by` is `None`, `"condition"`, `"lighting"` or `"split"`. With `None` the function
  returns one metrics dict; with a group key it returns `{group_value: metrics dict}`, with
  keys in enum order and only the groups that are present.
- Each metrics dict contains:
  - `n_arrangements`, `n_episodes`, `n_present`, `n_removed`, `n_indistinguishable`
  - `false_removal_rate` = Σ frac(REMOVED) over present episodes / n_present
  - `removal_recall` = Σ frac(REMOVED) over removed episodes / n_removed
  - `coverage` = Σ (1 - frac(UNKNOWN)) / n_episodes (unknown stays in the denominator)
  - `resolved_accuracy` = Σ frac(correct decision) / Σ (1 - frac(UNKNOWN))
  - `max_achievable_coverage` = 1 - n_indistinguishable / n_episodes
  - `mean_views`, `mean_additional_views`, `mean_travel_cm` (means of the per-episode means)
  - `confusion`: `{"present": {"present": x, "removed": x, "unknown": x}, "removed": {...}}`
    (float counts)
  - `defined`: `{metric: bool}`. A metric whose denominator is 0 is `None` with
    `defined[metric] = False`, never 0 and never NaN.
  - `policy`, `budget_additional_views`, `travel_cap_cm`, `synthetic` (True if any input is)
- Tests: known-answer synthetic cases, including fractional R (e.g. one present episode with
  frac REMOVED 0.25 gives FRR 0.25), all-unknown (`resolved_accuracy` is None and undefined,
  coverage 0), a removed-only group (FRR undefined), and counts reported in every group.

## 7. `analysis/bootstrap.py`

```python
@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    policy_a: str
    policy_b: str
    estimate: float | None     # metric(a) - metric(b) on the full sample (None if undefined)
    ci_lo: float | None        # 2.5th percentile of defined draws (None if no draw is defined)
    ci_hi: float | None        # 97.5th percentile
    n_arrangements: int
    n_episodes: int
    B: int
    n_undefined: int           # draws where either side was undefined (excluded)
    seed: int
    synthetic: bool
    def to_dict(self) -> dict: ...

def paired_diff(results_a, results_b, truth: GroundTruth, metric: str, B: int = 10000,
                seed: int = 0) -> BootstrapResult: ...
```

- `results_a` and `results_b` are normalised with `as_results` and must cover **the same
  episode_id set**. If they don't, `ValueError`.
- The bootstrap is paired and at the arrangement level. Each draw resamples the unique
  arrangement ids with replacement (`numpy.random.default_rng(seed)`; `rng.integers` over the
  sorted id list), takes **all** episodes of each drawn arrangement, with duplicates repeated,
  computes the metric for A and for B on that same multiset, and records A - B. The metric is
  computed as in `metrics.compute`. Share a helper that works on a multiset of results; don't
  call `compute` with duplicated ids.
- The CI is the percentile interval at 2.5 and 97.5 (`numpy.percentile`, default method).
- Tests: identical policies give an estimate of 0 and a CI of [0, 0]; a constructed case with
  a known difference; determinism for a fixed seed; the episodes of an arrangement are never
  split (resampling one arrangement brings all of its episodes); undefined draws are counted.

---

## 8. `verify/run.py` (CLI)

```
uv run --no-sync python -m verify.run --synthetic --n-arr 30 --seed 0 --budgets 0..4 \
    [--cap-steps 1,2,3,4,6 | none] [--block-anywhere] [--positions configs/positions.json] [--n-dev N] \
    [--r-seeds 20] [--bootstrap-B 10000] [--primary-budget 3] [--tag TAG] [--out runs/synthetic-<tag>/]
```

- Only `--synthetic` is implemented. Without it the CLI exits with code 2 and the message
  "real data not available yet". `--budgets` takes a comma list or a range such as `0..4`.
  `--cap-steps` takes a comma list of caps **in arc steps** (`inf` allowed) or `none`; the
  default is `1,2,3,4,6`, and the uncapped setting always runs in addition. Caps are an
  operating condition reported as curves, never tuned (D-cap): `cap_cm = cap_cm_from_steps(steps,
  positions)` with no slack, and there is deliberately no cm input. The geometry decides where
  selection can matter: at 1 step only P2/P4 are reachable; at 3 and 4 steps G's first move (P1,
  section 3) leaves no choice, so G and F coincide for lam < 1; the 2-step cap is where the D-G
  lookahead separates G from F; at 6 steps (141.37 cm = F's full tour) G's path P3 -> P1 -> P5
  costs 47.12 + 94.25 = 141.37 cm, exactly the cap and affordable, so the cap blocks the *third*
  move P5 -> P4 (cumulative 164.93 cm). Under the cap G either stops after P3 -> P1 -> P5 (cap used
  up), or, when the lookahead needs one more stop, goes P3 -> P1 -> P2 -> P5 (141.37 cm) instead
  (review F2 corrected an earlier "160 cm" claim). `--block-anywhere` selects the second synthetic regime (section 5). `--positions`
  (default `configs/positions.json`) is loaded and validated; a bad file exits with code 2. The default `--tag` is `n<N>-s<seed>`, and `--out`
  defaults to `runs/synthetic-<tag>/` (`runs/` is gitignored).
- Stages are kept separate, and the order matters:
  1. **Generate** (`generate_stage`): `synthetic.generate` produces specs, store and truth;
     `generate_stage` **drops the truth**, returns only specs and store, and writes nothing.
     Stale `ground_truth.json`, `evidence.jsonl` and `episodes.json` in the output dir are
     deleted first, so no truth-bearing or unvisited-evidence file exists while policies run.
  2. **Run** (`run_stage`, no truth parameter): mint every `ReplayKey` (`store.replay_keys()`),
     then for each episode run S once (`Budget(0)`, the k=1 rule object) and FULL once
     (`Budget(4)`). For each cap setting and each budget b, run `MULTI_VIEW` = F, R, G and
     G-myopic (R with `--r-seeds` seeds, salt `r_seed_salt(spec)`) with `StoppingRule(th)`, plus
     the first-clear-miss variants F-fcm, R-fcm, G-fcm and G-myopic-fcm with `StoppingRule.first_clear_miss(th)` (the same object S uses). Each run
     gets a fresh `ReplayEnv` built from the episode's key, never its spec, and the header is
     joined with `EpisodeLog.from_run`. Every `EpisodeLog` goes to `logs.jsonl`, one per line,
     with R's per-seed lines included.
  2b. **Scoring files** (`write_scoring_files`, after the run): regenerate deterministically from
     (n, seed, params), check the specs and every revealed observation in the logs against the
     regenerated evidence (`ReplayError` on any mismatch), then write `episodes.json`,
     `evidence.jsonl` and `ground_truth.json`.
  3. **Score** (`score_stage`): load the truth from `ground_truth.json`, then call
     `metrics.compute_all` per (policy, budget, cap) on the **test** split (scope `"test"`,
     primary) and on dev (scope `"dev"`, kept in results.json only). Then
     `bootstrap.paired_diff` on test at the primary budget, per cap setting, for G - F,
     G - R, G - S, F - S, R - S and G - G-myopic (and the same pairs for the `-fcm` variants),
     plus the H1 pairs F - F-fcm, R - R-fcm, G - G-fcm and G-myopic - G-myopic-fcm (`H1_PAIRS`), on
     `false_removal_rate`, `removal_recall` and `coverage`. S and FULL are reused at every
     budget and cap. Each bootstrap row carries `identical_runs`: True when A and B made the
     same decisions with the same effort on every episode, in which case the [0, 0] CI is
     degenerate and is not evidence (e.g. G and F at the 1-, 3- and 4-step caps).
  4. **Diagnostics** (`diagnostics_stage`, logs and geometry only, no truth): per (cap, b) the
     reachable view sets (`max views` and the number of distinct view sets of that size, from
     geometry), the number of distinct view sets each of F/G/G-myopic/R actually used, and path /
     decision disagreement for `DISAGREE_PAIRS` = (G, F), (G, R), (G, G-myopic), (G-myopic, F),
     per scope test/dev/all (pairs with R are averaged over R's seeds within each episode).
- `config.json` holds `{"label", "config", "config_sha256"}`. `results.json` contains
  `{"label": SYNTHETIC_LABEL, "synthetic": true, "config": {...}, "config_sha256": ...,
  "positions": ..., "thresholds": ..., "generator": {...}, "files": {name: sha256},
  "n_logs", "metrics": [{"scope", "policy", "budget", "travel_cap_cm", "role",
  ...compute_all...}], "bootstrap": [...], "diagnostics": {"reachable_view_sets",
  "used_view_sets", "disagreement"}}`. The `config` dict (thresholds, positions,
  budgets, `cap_steps`, `travel_caps_cm`, `nominal_step_cm`, `travel_tol_cm`, `g_myopic`
  {lam_deg, travel_norm_cm}, r_seeds, generator params, n_arr, seed, bootstrap B, primary budget,
  config_version (`synthetic-v3`), model_id, positions file path and sha256) is hashed with `config_sha256`, and
  the same hash is in every log line.
- `table.md` begins with `# SYNTHETIC - generated evidence, not study results`. It has the
  primary table (one row per policy at the primary budget, per cap setting): FRR, recall,
  coverage, resolved accuracy, mean views, mean travel, and n_arr/n_ep. S and FULL are marked
  "actual effort" and "higher-effort reference". Undefined values print as `undef`. The
  header carries the generator-regime caveat (section 5), the D-cap note (caps in steps, 1e-6 cm
  tolerance; a warning if measured stops deviate from the nominal step), the G / G-myopic note, and the maximum achievable coverage on test with the predeclared indistinguishable count.
  Under a capped heading FULL is printed as `FULL (uncapped)` with the note "uncapped reference
  (exceeds cap)". Below that come the budget curves, the per-condition table (with a max
  achievable coverage column), the **cap curve at the primary budget** (each policy across the
  cap grid), the two diagnostic tables (distinct view sets; path / decision disagreement), the
  per-lighting table and the bootstrap lines with their n and an `identical` note on degenerate
  pairs. The default run (30 arrangements, 6 cap settings) writes about 250k log lines
  (`logs.jsonl` about 200 MB) in about 16 s.
- `main(argv: list[str] | None = None) -> int`, for testing. `tests/test_end_to_end.py` runs
  `--n-arr 6 --r-seeds 3 --bootstrap-B 200` into a temp dir, checks the files, the label and
  the required log fields, checks that `logs.jsonl` lines contain no `truth` key, spies on
  every `select` and `decide` call to check they only receive Observations, and checks that
  running twice with the same arguments gives byte-identical `results.json`. It also checks that
  no scoring file is in the output dir when `run_stage` starts, that no `ReplayEnv` argument is a
  spec, truth or string or mentions a condition, the table caveats, and `identical_runs`.

---

## 9. `analysis/intervals.py` (stdlib only)

```python
def clopper_pearson_upper(x: int, n: int, conf: float = 0.95) -> float: ...   # one-sided
def clopper_pearson_lower(x: int, n: int, conf: float = 0.95) -> float: ...
def clopper_pearson(x: int, n: int, conf: float = 0.95) -> tuple[float, float]: ...  # two-sided
def jeffreys_upper / jeffreys_lower / jeffreys(x, n, conf=0.95): ...
def bounds_summary(x: int, n: int, conf: float = 0.95) -> dict: ...  # rate + every bound, for reports
def betainc(a, b, x) -> float: ...       # regularized incomplete beta (Lentz continued fraction)
def beta_ppf(q, a, b) -> float: ...      # bisection on betainc
def binom_cdf(x, n, p) -> float: ...     # exact sum, independent cross-check
def clopper_pearson_upper_bisect(x, n, conf=0.95) -> float: ...  # CP upper by bisection on binom_cdf
```

- Both families are beta quantiles: CP lower = Beta(x, n-x+1).ppf(alpha) (0 at x = 0), CP upper =
  Beta(x+1, n-x).ppf(1-alpha) (1 at x = n); Jeffreys uses Beta(x+1/2, n-x+1/2) with the
  Brown-Cai-DasGupta convention (lower 0 at x = 0, upper 1 at x = n). alpha = 1 - conf one-sided,
  (1 - conf)/2 per side two-sided.
- Counts must be integers. R's seed-averaged fractional counts are refused (`TypeError`): they
  are not binomial counts, which is why selection uses a deterministic policy.
- Reference values (one-sided 95%, asserted in `tests/test_intervals.py`): CP upper 0/40 =
  0.0722, 1/40 = 0.1132, 0/10 = 0.2589; CP lower 16/20 = 0.5990. Closed-form checks: Beta(1, b),
  Beta(a, 1) and the arcsine Beta(1/2, 1/2).

---

## 10. `verify/select.py` (D-select; dev only)

```
uv run --no-sync python -m verify.select --synthetic [--n-arr 30] [--seed 0] [--block-anywhere] \
    [--policy G|F] [--budget 3] [--cap-steps inf|<steps>] [--tau-match-grid ...] [--k-grid 3,2] \
    [--lam-grid 0,0.7,0.9,1.2] [--tau-q-grid ...] [--tau-occ-grid ...] [--alpha-clear 0.10] \
    [--frr-limit 0.10] [--recall-target 0.80] [--bootstrap-B 2000] [--tag TAG] [--out runs/select-<tag>/]
uv run --no-sync python -m verify.select --lam-breakpoints [--lam-grid ...] [--positions ...]
```

The procedure follows the literature report (`notes/2026-10-02-caps-thresholds-literature.md`,
section 3). Every setting below is a *proposed default - confirm*, to be predeclared in
`docs/PREREGISTRATION.md` before any dev data are looked at.

- **Data interfaces.** `StoreFactory` is a Protocol (`n_stops`, `__call__() -> EvidenceStore`)
  that returns a **fresh, unsealed** store holding exactly the dev episodes, identical on every
  call; `RecordStoreFactory(records, n_stops, synthetic=..., source=...)` implements it from an
  immutable snapshot (`RecordStoreFactory.from_store(unsealed_store, episode_ids)`), with
  `evidence_sha256()` for provenance. `ViewLabel(episode_id, stop, diagnostic)` is the per-view
  dev annotation (ambiguous views are non-diagnostic); files use schema `cmpe249.view_labels.v1`
  via `save_view_labels` / `load_view_labels`. `DevData(specs, store_factory, truth, view_labels,
  positions, synthetic, label_source, evidence_sha256)` requires a label for every (dev episode,
  stop). Real data go through the library (build a `DevData`, call `run_selection`); the CLI is
  `--synthetic` only, where `synthetic_dev_data` uses stand-in labels from the generator's hidden
  layout (blocked or edge stop -> non-diagnostic), clearly marked as such.
- **Split refusal.** `DevData.__post_init__` and `run_selection` raise `SplitRefused`
  (a `ManifestError`) for any non-DEV spec (test and pilot), and if the store or the labels hold
  any episode outside the dev specs. `synthetic_dev_data` snapshots only DEV episodes.
- **Evaluation.** `Evaluator` builds one fresh store per `Thresholds` config, mints all keys before
  the first bind, runs the policy through `ReplayEnv` / `run_episode` (G built only through
  `G.from_thresholds(th)`, so `Thresholds.lam` is G's dimensionless D-norm weight), caches
  outcomes per config, and joins dev truth only to score. `SelectionSettings.policy` is `"G"`
  (default) or `"F"`; with F the lam grid collapses to the schema default (lam is not selected,
  and the placeholder 0.5 is what G and A would then use unless a separate lam rule is declared;
  review P6). All grids must be finite; lam values must be >= 0 (review F3).
  The operating point is `budget` (default 3) and `cap_steps` (default uncapped, in D-cap steps).
- **Step 2 (gate).** Over the fixed grids tau_q 0.30..0.80 and tau_occ 0.10..0.50 (step 0.05),
  Neyman-Pearson style: null set = non-diagnostic views of occluded-present episodes (passing the
  gate there is a false clear), power set = diagnostic views of removed episodes. Keep gates with
  false-clear rate <= `alpha_clear` (0.10), maximise the clear rate; ties: lower false-clear,
  higher tau_q, lower tau_occ. If no gate qualifies, the lowest false-clear gate is taken with a
  warning.
- **Step 3 (fixed-sequence walk + D10).** The walk is k-major in `k_grid` order (3 then 2), then
  tau_match ascending (conservative -> liberal). A (k, tau_match) step passes if its **worst** FRR
  over the lam grid is <= `frr_limit` (0.10), so the lam choice can never rescue the FRR
  constraint. The accepted set is the prefix before the first failing step. Consequence of the
  k-major order (review P11): one failing k = 3 step excludes every k = 2 config, even a k = 2
  step with lower FRR; "conservative -> liberal" is a predeclared order, not a proven monotone
  one. The FRR filter uses FRR measured on the arrangements selection runs on (inner-training
  arrangements inside a fold, all dev arrangements for the candidate), not a grouped-CV FRR;
  nested CV is only the honest estimate (review F5). D10 over accepted
  (k, tau_match, lam): keep configs with recall >= `recall_target` (0.80) if any, maximise
  coverage, then recall, then lower mean travel, then grid order. If even step 0 fails, the most
  conservative step is returned with `meets_frr_limit = false` and a warning. Monotonicity is
  checked empirically and warnings are logged and reported (a later step would have passed; worst
  FRR decreases along the walk; FRR not non-decreasing in tau_match; FRR higher at the larger k).
- **Step 4 (nested grouped CV).** Outer leave-one-arrangement-out; the inner selection is the
  same Steps 2 + 3 on the other arrangements only (Step 2 is refit per fold). Pooled held-out
  decisions are the honest dev estimate; the candidate is the rule applied to all dev arrangements.
- **Bounds.** For both the nested-CV estimate (the one to quote) and the resubstitution
  candidate (labelled optimistic): one-sided 95% Clopper-Pearson and Jeffreys upper bounds on
  episode-level FRR and arrangement-level FRR (x = arrangements with any false removal; the
  0/10-style bound), lower bounds on recall and coverage, plus a percentile cluster bootstrap over
  arrangements (`analysis.bootstrap.arrangement_draws`).
- **Outputs** (deterministic, no timestamps): `frozen_candidate.json` (status
  "CANDIDATE - not frozen until the student signs FREEZE.md"; thresholds, settings sha256, dev
  evidence sha256, dev arrangement list, label source, positions, travel_cap_cm, warnings) with
  `frozen_candidate.json.sha256` (sha256sum format), `lam_grid_check` (breakpoints, which
  grid values are behaviourally identical, which sit on a breakpoint; G only), and on `--synthetic`
  `synthetic_provenance` (generator parameters incl. `block_includes_start`, n_arr, n_dev, seed),
  which also enters the settings sha256 (review F4), `selection_report.md` and
  `selection_details.json` (full grid and per-fold numbers). On `--synthetic` every file is
  labelled SYNTHETIC and nothing may be carried into FREEZE.md.
- **lam grid (review F1).** The old grid {0, 0.25, 0.5, 1.0} had only two G behaviours at the
  design geometry: {0, 0.25, 0.5} are identical in every decision state and 1.0 sits on the
  breakpoint at 1 (the tie-break makes it behave like lam > 1). The default is now
  **{0, 0.7, 0.9, 1.2}**: one value inside each of the intervals [0, 2/3), (2/3, 3/4), (3/4, 1),
  (1, 4/3), none on a breakpoint (*proposed default - confirm*; lam > 4/3 is left out as strongly
  travel-dominated). `run_selection` runs `verify.lamgrid.check_lam_grid` for G and logs and reports
  a warning when grid values collapse or sit on a breakpoint, so a grid that is stale for the
  measured geometry is caught. The D10 tie-break "lower mean travel" still favours the largest lam
  among otherwise equal configs.

---

## 11. Non-negotiables (from AGENTS.md and the protocol)

- Policies see only requested views. Ground truth is joined only at scoring. The env never
  receives the spec or the condition. The runtime guarantee covers public and single-underscore
  paths only; introspection is excluded by the policy lint and by review (section 2).
- F, R, G, G-myopic (and A later) share one `StoppingRule`. Unknown counts against coverage.
  G's lookahead reads the rule's k and predicate but never changes the decision.
- Threshold selection reads DEV arrangements only (`SplitRefused` otherwise).
- Splits and the bootstrap work at the arrangement level. An arrangement's episodes are never
  split.
- R is averaged within the episode before any paired comparison. Seeds add no samples.
- Synthetic output is labelled SYNTHETIC everywhere and is never reported as a result.
- Don't add dependencies, don't edit `pyproject.toml` or `uv.lock`, and don't commit.
