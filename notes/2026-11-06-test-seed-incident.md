# Test-seed incident: one shelf test arrangement generated temporarily by a unit test (record, 2026-11-06)

**Recorded before the freeze, at the researcher's request (2026-11-06), so that the freeze covers it.** Times are
from the session transcript (UTC) and the git log (PST). Local time on the Jetson is PST, UTC−8. The session's own
progress log dates the incident "2026-11-06 (early)", and the test file's docstring dates it 2026-11-05; both are the
same event.

## What happened

On 2026-11-05 at 18:20:59–18:21:20 PST (2026-11-06 02:20:59–02:21:20 UTC), I ran a first, uncommitted version of
`tests/test_freeze_gate.py` once, while building the test-seed gate. The gate's first commit, 12b56f5, came later, at
18:54 PST.

- **The test.** `GeneratorTest.test_a_test_arrangement_records_its_freeze` called sim.arrange's command-line entry,
  `main(["--split", "test", "--freeze", "x.md", "--slots", "0", "--out", <temporary directory>/o])`.
  - `sim.freeze.require_freeze` was replaced by a mock. The mock returned a stand-in freeze record: freeze_id
    "freeze-test", sha256 "f" × 64.
  - sim/arrange.py then carried a first, uncommitted version of the gate, which accepted that record.
  - The CLI ran with its default config, `configs/sim/arrange_shelf_v3.json`.
- **What it generated.** The study's shelf test arrangement for **slot 0**.
  - Slot *i* has primary seed 100000 + *i*, so slot 0's primary seed is **100000**.
  - A placement retry keeps the seed. A reserve seed of the slot's family replaces it only after 21 failed tries.
  - The test did not record which seed the arrangement used. The same deterministic generation after the freeze
    writes it to `index/generation_log.json`.
  - The placement code was the committed generator's: the uncommitted edits changed only the seed check, the id,
    the split and the freeze field. So the geometry generated then is the geometry that slot 0 gets after the freeze.
- **What it wrote**, all inside the temporary directory:
  - the arrangement file;
  - `index/index.json`;
  - `index/slots.json`, the whole shelf test slot list: family, M16 side, target and primary seed of each of the 100
    slots;
  - `index/generation_log.json`.
- **What the test inspected.**
  - It listed the output directory's top-level `*.json` files and asserted that there was exactly one.
  - It read that one file into memory and asserted only four things: split "test"; seed ≥ 100000; arrangement_id =
    "ST" followed by the seed in six digits; and the freeze field equal to the stand-in record.
  - It also asserted that the mocked gate was asked for `configs/sim/arrange_shelf_v3.json` and `sim/arrange.py`,
    and that the unlock was closed after the run.
  - Every assertion passed, so no value appeared in a failure message. The test printed only unittest's summary
    ("Ran 8 tests ... OK").
  - The CLI's per-arrangement summary line (slot, seed, target, family, P3 coverage, distractors) and its
    generation-log line went to a redirected stdout (`io.StringIO`) and were discarded unread.
- **What no one saw.** No placement, pose, object, target, occluder geometry, slot-list entry or generation-log content
  was printed, displayed or read by a person, and none was passed to any other code.
- **Deleted.** The temporary directory was removed when the test's `with` block closed, at the end of that run.
- **Not rendered or scored.** Nothing was copied to the shared volume or to the GB10. The arrangement was never
  rendered, built into episodes, run through evidence or scored. No outcome of any test arrangement exists.

## A second, smaller exposure: the test slot list

The first repair (below) added `test_main_asks_the_freeze_for_the_generator_and_config`.

- It called the same CLI with `--split test` and no `--slots`, under the same mock. It generated no arrangement.
- The CLI still wrote `index/index.json`, with no arrangement, and `index/slots.json`, the shelf test slot list, into a
  temporary directory.
- The test asserted only that the directory held no top-level arrangement file.
- It ran three times: 18:22:33 and 18:23:37 PST, and once in the full suite, which ended at 18:28:15 PST.
- Each directory was deleted at the end of its test. The slot list was not printed or read.

The slot list is not test geometry. It is a deterministic function of the frozen config: the seeded permutation of
the quota table, primary seed 100000 + *i*, and the seeded target lists. `sim/README.md` allows inspecting it before the
freeze.

**What the test suite itself does with the slot lists, by design.** This is part of the record for completeness; none
of it is an incident.
- Every run of the suite computes the shelf test slot list in memory:
  - `tests/test_sim_arrange.py` checks its length, its seed range, and that generating from it is refused;
  - the shelf goldens pin its sha256 (`tests/shelf_golden.py`, `slot_list_hashes`).
- Two tests move the test block's seeds to the synthetic range, 900000+, and keep the permutation and target seeds:
  - `tests/test_freeze_gate.py` generates a synthetic arrangement for slot 0;
  - `tests/test_test_validity.py` generates synthetic arrangements for slots 0 and 1, and writes them with a
    `slots.json` into a temporary directory.
- So those slot lists carry the study's family, M16 side and target sequence, while the seeds, and so the placements,
  are synthetic. No study seed's geometry is generated.
- None of these tests prints a slot or an arrangement.

## How it was found and how the gate was repaired

1. **Found.** At 18:21:45 PST I printed the configs' test seed blocks (shelf primary 100000–100099, reserves
   101000–101419; tabletop 110000–110099 and 111000–111419). The test had used the study's shelf block.
2. **18:22 PST.** The generating test was rewritten. It now generates only from a synthetic seed block, 900000+, of an
   in-memory copy of the shelf config.
3. **18:23 PST.** The uncommitted gate re-checked the FREEZE.md file itself on disk. A stand-in record that names no
   real file could no longer open test seeds.
4. **18:42 PST.** sim/arrange.py was restored to its committed bytes, because an arrangement's sha256 covers the
   generator's own and the edits would have changed every development arrangement's sha256. The gate moved to
   `sim/testsplit.py`. It wraps sim.arrange's seed check, id and finish functions only inside `unlocked`, and
   restores them afterwards.
5. **12b56f5 (18:54 PST).** The gate was committed. Its tests generate only from the synthetic block. The test file's
   docstring states the rule: never generate an arrangement from the study's test blocks before the freeze.
6. **28c7e96 (21:32 PST).** This followed an Opus/high review of the gate.
   - A seed below 900000, which covers every study seed and reserve, opens only if the committed `docs/FREEZE.md`
     passes the `require_freeze` captured when the module is imported, and has the freeze's sha256. A mock of
     `sim.freeze.require_freeze`, or a stand-in record with a matching file elsewhere, therefore cannot open a study
     seed.
   - `require_freeze` checks the file's committed bytes at HEAD, and that the record's commit is an ancestor of HEAD.
   - `tests/test_freeze_gate.py` checks that a mocked gate cannot open a study seed.

No later test or review generated, rendered or read an arrangement from the study's test blocks. Since 12b56f5, the
gate tests use the synthetic block only, and the refusal tests abort before any generation.

## Decision

**The original seed schedule is kept** (the researcher, 2026-11-06, on these disclosed facts). The shelf and tabletop
test blocks, their slot lists, targets and reserves, and slot 0 with primary seed 100000, are generated after the
freeze exactly as configured. No seed is replaced or excluded, and no outcome exists that could motivate a
replacement.

## Statements this record qualifies

Statements that no test arrangement, test scene or test seed was ever generated before the freeze are qualified by
this record. One shelf test arrangement (slot 0) was generated temporarily, as above, and the shelf test slot list
was written four times to temporary directories. No test arrangement was rendered, scored or viewed.

- docs/SIM_PREREGISTRATION.md, docs/SIMULATION_STUDY.md, docs/REPRODUCE.md and the pooled-calibration note carry a
  pointer to this record where they state it.
- Several frozen files cannot change without moving their pinned hashes:
  - `sim/arrange.py`'s comments and help texts;
  - the generator configs' `slots.note`;
  - Amendment 2 of `notes/2026-11-04-registration-repair-plan.md`, which says "no seed >= 100000 is generated, rendered
    or scored before FREEZE.md".

  Those statements were true when written. This record supersedes them, as FREEZE.md says.
