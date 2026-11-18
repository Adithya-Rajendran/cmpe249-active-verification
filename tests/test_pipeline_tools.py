"""tools/pipeline.py helpers, tools/evidence_summary.py and tools/quality_resampling.py (stdlib tools, no GB10)."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evidence_summary as es  # noqa: E402
import pipeline  # noqa: E402
import provenance  # noqa: E402
import quality_resampling as qr  # noqa: E402


class LocalShell:
    """Stands in for Remote: runs the command locally."""
    def __init__(self, cfg: dict | None = None):
        pass

    def argv(self, command: str) -> list[str]:
        return ["bash", "-c", command]

    def run(self, command: str, timeout: int = 60, stdin: bytes | str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(self.argv(command), input=stdin, capture_output=True, timeout=timeout,
                              text=not isinstance(stdin, bytes))


class TestReadNewLines(unittest.TestCase):
    def test_carriage_returns_and_partial_lines(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / "log"
            log.write_bytes("a\nLoading 0%\rLoading █ 100%\nEVIDENCE_STEP start {}\npart".encode())
            lines, off = pipeline.read_new_lines(LocalShell(), str(log), 0)
            self.assertIn("EVIDENCE_STEP start {}", lines)
            self.assertIn("Loading █ 100%", lines)
            self.assertNotIn("part", lines)                       # incomplete: kept for the next poll
            with log.open("ab") as f:
                f.write(b"ial\nnext\n")
            lines, off2 = pipeline.read_new_lines(LocalShell(), str(log), off)
            self.assertEqual(lines, ["partial", "next"])
            self.assertEqual(off2, log.stat().st_size)
            self.assertEqual(pipeline.read_new_lines(LocalShell(), str(log), off2), ([], off2))


def spec_with_jobs(t: Path, args: list[str]) -> dict:
    (t / "jobs_refL1.json").write_text(json.dumps({"jobs": [{}, {}, {}]}))
    return {"set_root": str(t), "remote": {"evidence": {"uv": "UV", "code_dir": "HOST"}},
            "evidence": {"out_dir": "/out", "code_dir": "SPEC", "jobs": ["jobs_refL1.json"],
                         "variants": {"v": {"args": args, "outputs": ["{out}/v_{part}.jsonl"]}}}}


class TestEvidenceSteps(unittest.TestCase):
    def test_placeholders_records_and_host_settings(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            spec = spec_with_jobs(t, ["--jobs", "{jobs}", "--set", "{set}", "--out", "{out}/v_{part}.jsonl"])
            (st,) = pipeline.evidence_steps(spec, "/out")
            self.assertEqual(st["name"], "v_refL1")
            self.assertEqual(st["args"], ["--jobs", f"{t}/jobs_refL1.json", "--set", str(t), "--out", "/out/v_refL1.jsonl"])
            self.assertEqual((st["records"], st["outputs"]), (3, ["/out/v_refL1.jsonl"]))
            ev = pipeline.evidence_settings(spec)
            self.assertEqual((ev["uv"], ev["code_dir"]), ("UV", "SPEC"))   # the spec wins over the host file

    def test_key_changes_with_the_steps_and_the_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            a = pipeline.evidence_steps(spec_with_jobs(t, ["--jobs", "{jobs}", "--align", "image"]), "{out}")
            b = pipeline.evidence_steps(spec_with_jobs(t, ["--jobs", "{jobs}", "--align", "tag"]), "{out}")
            self.assertEqual(a[0]["outputs"], ["{out}/v_refL1.jsonl"])     # the key never depends on the out dir
            ka, kb = pipeline.steps_key(a), pipeline.steps_key(b)
            self.assertNotEqual(ka, kb)
            self.assertEqual(ka, pipeline.steps_key(a))
            (t / "jobs_refL1.json").write_text(json.dumps({"jobs": [{}, {}, {}], "x": 1}))
            self.assertNotEqual(ka, pipeline.steps_key(a))       # same commands, different jobs content


# The GB10 side of a job, run locally: `uv run --no-sync --project P python ARGS` -> python3 ARGS.
FAKE_UV = '#!/bin/bash\nwhile [ "$1" != python ]; do shift; done\nshift\nexec python3 "$@"\n'
# One step: write a single record naming the variant to the output (no braces: args go through str.format).
WRITE = "import json, sys; open(sys.argv[1], 'w').write(json.dumps(dict(variant=sys.argv[2])) + chr(10))"


def job_spec(t: Path, variant: str) -> dict:
    return {"run_id": "same_run", "set_root": str(t), "poll_seconds": 0, "heartbeat_seconds": 3600,
            "remote": {"run_dir": str(t / "remote_run"),
                       "evidence": {"uv": str(t / "uv"), "project": str(t / "project"), "code_dir": str(t / "code"),
                                    "checkpoints": str(t / "checkpoints")}},
            "evidence": {"out_dir": str(t / "evidence"), "jobs": ["jobs_refL1.json"],
                         "variants": {"v": {"args": ["-c", WRITE, "{out}/v_{part}.jsonl", variant],
                                            "outputs": ["{out}/v_{part}.jsonl"]}}}}


class TestEvidenceJobs(unittest.TestCase):
    """stage_evidence end to end with the GB10 replaced by a local shell (git archive of the real repo)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        t = self.t = Path(self.tmp.name)
        for d in ("remote_run", "project", "checkpoints"):
            (t / d).mkdir()
        (t / "project" / "uv.lock").write_text("lock\n")
        (t / "uv").write_text(FAKE_UV)
        (t / "uv").chmod(0o755)
        (t / "jobs_refL1.json").write_text(json.dumps({"jobs": [{}]}))
        real_git = pipeline.git   # the working tree may hold uncommitted edits: report it clean
        self.patches = [patch.object(pipeline, "Remote", LocalShell),
                        patch.object(pipeline, "git", lambda *a: "" if a[0] == "status" else real_git(*a))]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def run_job(self, variant: str, run_dir: str) -> dict:
        run = pipeline.Run(job_spec(self.t, variant), self.t / run_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.stage_evidence(run)
        return json.loads((run.dir / "evidence_provenance.json").read_text())

    def test_a_then_b_then_resume_a_keeps_a_outputs(self) -> None:
        a = self.run_job("A", "run_a")
        b = self.run_job("B", "run_b")
        self.assertNotEqual(a["key"], b["key"])
        self.assertNotEqual(a["out"], b["out"])                     # each job has its own output dir
        (out_a,), (out_b,) = a["outputs"], b["outputs"]
        self.assertEqual(json.loads(Path(out_a).read_text())["variant"], "A")
        self.assertEqual(json.loads(Path(out_b).read_text())["variant"], "B")
        resumed = self.run_job("A", "run_a_resumed")                 # attaches to the finished A job
        self.assertEqual(resumed["outputs"], a["outputs"])
        self.assertEqual(resumed["finished"], a["finished"])
        events = (self.t / "run_a_resumed" / "events.jsonl").read_text()
        self.assertIn("attached", events)
        self.assertIn("reused outputs match the saved sha256", events)
        self.assertEqual(provenance.output(resumed, "{out}/v_refL1.jsonl"), Path(out_a))

    def test_resume_refuses_outputs_changed_after_the_job_finished(self) -> None:
        a = self.run_job("A", "run_a")
        (out_a,) = a["outputs"]
        Path(out_a).write_text(json.dumps({"variant": "B"}) + "\n")   # same record count, other bytes
        with self.assertRaises(SystemExit):
            self.run_job("A", "run_a_resumed")
        last = json.loads((self.t / "run_a_resumed" / "events.jsonl").read_text().splitlines()[-1])
        self.assertTrue(last["data"]["terminal"])
        self.assertEqual(last["data"]["changed"], [out_a])
        with self.assertRaises(SystemExit):                         # downstream readers refuse it too
            provenance.output(a, "{out}/v_refL1.jsonl")

    def test_deleting_the_output_dir_relaunches_a_finished_job(self) -> None:
        a = self.run_job("A", "run_a")
        Path(a["outputs"].popitem()[0]).unlink()
        Path(a["out"], "provenance.json").unlink()
        Path(a["out"]).rmdir()
        again = self.run_job("A", "run_a_again")
        self.assertIn("relaunching", (self.t / "run_a_again" / "events.jsonl").read_text())
        self.assertEqual(again["key"], a["key"])
        self.assertNotEqual(again["finished"], None)

    def test_moving_out_dir_is_a_new_job(self) -> None:
        a = self.run_job("A", "run_a")
        spec = job_spec(self.t, "A")
        spec["evidence"]["out_dir"] = str(self.t / "evidence2")
        run = pipeline.Run(spec, self.t / "run_moved")
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.stage_evidence(run)
        moved = json.loads((run.dir / "evidence_provenance.json").read_text())
        self.assertNotEqual(moved["key"], a["key"])
        self.assertTrue(moved["out"].startswith(str(self.t / "evidence2")))


DUMMY_RENDER = """import sys, time
print('ISAAC_BATCH {"arrangement_id": "A1", "total_s": 1.0}', flush=True)
time.sleep(float(sys.argv[1]))
"""


@unittest.skipUnless(shutil.which("pgrep"), "needs pgrep")
class TestRenderProbe(unittest.TestCase):
    """The render liveness probe against REAL local processes (the GB10 replaced by a local bash). The probe's own
    shell carries the pattern in its command line, so an unbracketed `pgrep -f` always said "running"."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.arr = self.t / "arr_x"
        self.pattern = pipeline.render_pattern({"arrangements_dir": str(self.arr)})
        self.rc, self.log = str(self.t / "x.rc"), str(self.t / "x.log")
        self.procs: list[subprocess.Popen] = []

    def tearDown(self) -> None:
        for p in self.procs:
            p.kill()
            p.wait()
        self.tmp.cleanup()

    def dummy(self) -> subprocess.Popen:
        """An owned process whose command line contains the pattern, the way the real isaac_batch.py's does."""
        script = self.t / "dummy.py"
        script.write_text(DUMMY_RENDER)
        p = subprocess.Popen([sys.executable, str(script), "30", "isaac_batch.py", "--arrangements",
                              f"{self.arr}/a.json"], stdout=subprocess.DEVNULL)
        self.procs.append(p)
        return p

    def probe(self) -> dict:
        return pipeline.probe_render(LocalShell(), self.pattern, self.rc, self.log)

    def test_the_legacy_probe_always_said_running_and_the_new_command_cannot(self) -> None:
        legacy = f"pgrep -f {shlex.quote(self.pattern)} >/dev/null && echo RUNNING=1 || echo RUNNING=0"
        self.assertEqual(LocalShell().run(legacy).stdout.strip(), "RUNNING=1")   # no renderer exists: the old bug
        new = pipeline.render_probe_command(self.pattern, self.rc, self.log)
        out = LocalShell().run(new).stdout.split()                               # the command string itself, in bash
        self.assertIn("RUNNING=0", out)
        self.assertNotIn("RUNNING=1", out)

    def test_self_excluding_pattern(self) -> None:
        self.assertEqual(pipeline.self_excluding("isaac_batch.py --arrangements a/"),
                         "[i]saac_batch.py --arrangements a/")
        for bad in ("", "^isaac", "]x", "\\x"):
            with self.assertRaises(ValueError):
                pipeline.self_excluding(bad)
        self.assertIn("'[i]saac_batch.py", pipeline.render_probe_command(self.pattern, self.rc, self.log))

    def test_absent_then_running_then_exited(self) -> None:
        state = self.probe()
        self.assertEqual((state["status"], state["rc"], state["lines"]), (pipeline.STOPPED, "", 0))
        p = self.dummy()
        deadline = time.time() + 10
        while self.probe()["status"] != pipeline.RUNNING and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.probe()["status"], pipeline.RUNNING)
        p.kill()
        p.wait()
        self.assertEqual(self.probe()["status"], pipeline.STOPPED)

    def test_other_arrangements_dir_is_not_matched_and_rc_and_lines_are_read(self) -> None:
        p = self.dummy()
        time.sleep(0.5)
        other = pipeline.render_pattern({"arrangements_dir": str(self.t / "arr_y")})
        self.assertEqual(pipeline.probe_render(LocalShell(), other, self.rc, self.log)["status"], pipeline.STOPPED)
        Path(self.rc).write_text("0\n")
        Path(self.log).write_text("a\nb\n")
        state = self.probe()
        self.assertEqual((state["status"], state["rc"], state["lines"]), (pipeline.RUNNING, "0", 2))
        self.assertIsNone(p.poll())

    def test_failed_probes_are_not_stopped(self) -> None:
        good = "RUNNING=0\nRC=\nLINES=0\nPROBE_END\n"
        cases = [(255, "", "ssh: connect to host: Connection refused"), (1, good, ""), (127, "", "pgrep: not found"),
                 (0, "", ""), (0, "garbage\n", ""), (0, "RUNNING=0\nRC=\nLINES=0\n", ""),           # cut short
                 (0, "RUNNING=ERR2\nRC=\nLINES=0\nPROBE_END\n", ""), (0, "RUNNING=0\nRC=\nLINES=x\nPROBE_END\n", "")]
        for rc, out, err in cases:
            with self.subTest(rc=rc, out=out):
                self.assertEqual(pipeline.parse_render_probe(rc, out, err)["status"], pipeline.PROBE_FAILED)
        self.assertEqual(pipeline.parse_render_probe(0, good)["status"], pipeline.STOPPED)

        class Exits255(LocalShell):
            def argv(self, command: str) -> list[str]:
                return ["bash", "-c", "echo 'ssh: connection refused' >&2; exit 255"]

        class Garbage(LocalShell):
            def argv(self, command: str) -> list[str]:
                return ["bash", "-c", "echo hello"]

        class TimesOut(LocalShell):
            def run(self, command, timeout=60, stdin=None):
                raise subprocess.TimeoutExpired("ssh", timeout)

        for shell in (Exits255, Garbage, TimesOut):
            with self.subTest(shell=shell.__name__):
                state = pipeline.probe_render(shell(), self.pattern, self.rc, self.log)
                self.assertEqual(state["status"], pipeline.PROBE_FAILED)
                self.assertTrue(state["error"])


class RenderStageBase(unittest.TestCase):
    """stage_render end to end: Remote is a local bash, the launcher a script that runs a dummy "isaac_batch.py"."""

    def setUp(self) -> None:
        if not shutil.which("pgrep"):
            self.skipTest("needs pgrep")
        self.tmp = tempfile.TemporaryDirectory()
        self.t = t = Path(self.tmp.name)
        (t / "remote_run").mkdir()
        (t / "dummy.py").write_text(DUMMY_RENDER)
        (t / "remote_run" / "run_render.sh").write_text(
            f"#!/bin/bash\necho started >> {t}/launches\n"
            f"{sys.executable} {t}/dummy.py ${{DUMMY_SECONDS:-1}} isaac_batch.py --arrangements {t}/arr_s/a.json > $1 2>&1\necho $? > $2\n")
        self.spec = {"run_id": "render-test", "poll_seconds": 0.1, "heartbeat_seconds": 3600, "stall_minutes": 60,
                     "slow_factor": 2.5, "remote": {"run_dir": str(t / "remote_run")},
                     "render": {"log": "s.log", "rc_file": "s.rc", "expected": 1, "arrangements_dir": str(t / "arr_s"),
                                "launch_script": "run_render.sh s.log s.rc", "launch_wait_seconds": 20}}
        real_sleep = time.sleep
        self.patches = [patch.object(pipeline, "Remote", self.shell_class()),
                        patch.object(pipeline.time, "sleep", lambda s: real_sleep(min(s, 0.1)))]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        subprocess.run(["pkill", "-f", f"isaac_batch.py --arrangements {self.t}/"])
        self.tmp.cleanup()

    def shell_class(self) -> type:
        return LocalShell

    def run_render(self, launch: bool) -> tuple[int | None, list[dict]]:
        run = pipeline.Run(self.spec, self.t / "run")
        code = None
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                pipeline.stage_render(run, launch)
            except SystemExit as e:
                code = e.code
        return code, [json.loads(x) for x in (run.dir / "events.jsonl").read_text().splitlines()]

    @property
    def launches(self) -> int:
        f = self.t / "launches"
        return len(f.read_text().split()) if f.exists() else 0


class TestRenderStage(RenderStageBase):
    def test_launch_with_no_renderer_launches_instead_of_attaching(self) -> None:
        code, ev = self.run_render(launch=True)
        msgs = [e["msg"] for e in ev]
        self.assertIsNone(code)
        self.assertIn("launched", msgs)
        self.assertNotIn("attached", msgs)
        self.assertEqual(self.launches, 1)
        self.assertEqual(msgs[-1], "render finished")

    def test_the_launch_call_returns_while_the_render_runs(self) -> None:
        # the launcher must not hold ssh's pipes: with `cd X && setsid ... &` the whole list went to the background
        # in a subshell that kept stdout open, so ssh waited for the render and the 60 s launch call timed out
        class ShortTimeout(LocalShell):
            def run(self, command, timeout=60, stdin=None):
                return super().run(command, timeout=min(timeout, 3), stdin=stdin)
        with patch.object(pipeline, "Remote", ShortTimeout), patch.dict(os.environ, {"DUMMY_SECONDS": "6"}):
            code, ev = self.run_render(launch=True)
        msgs = [e["msg"] for e in ev]
        self.assertIsNone(code, msgs)
        self.assertIn("launched", msgs)
        self.assertEqual(self.launches, 1)
        self.assertEqual(msgs[-1], "render finished")

    def test_without_launch_and_no_renderer_fails_instead_of_attaching(self) -> None:
        code, ev = self.run_render(launch=False)
        self.assertEqual(code, 1)
        self.assertIn("rerun with --launch", ev[-1]["msg"])
        self.assertEqual(self.launches, 0)

    def test_attaches_to_a_running_render_and_alerts_when_it_is_confirmed_gone(self) -> None:
        with (self.t / "remote_run" / "s.log").open("w") as out:
            p = subprocess.Popen([sys.executable, str(self.t / "dummy.py"), "1", "isaac_batch.py", "--arrangements",
                                  f"{self.t}/arr_s/a.json"], stdout=out)
            time.sleep(0.3)
            code, ev = self.run_render(launch=True)   # the dummy exits after 1 s and never writes an rc file
            p.wait()
        self.assertEqual(ev[0]["msg"], "attached")
        self.assertTrue(ev[0]["data"]["running"])
        self.assertEqual(self.launches, 0)
        self.assertEqual(code, 1)
        self.assertEqual(ev[-1]["msg"], "render process is gone and wrote no rc file")
        self.assertTrue(ev[-1]["data"]["terminal"])


class TestRenderStageFailedProbe(RenderStageBase):
    """A failing probe is PROBE_FAILED: no launch, no "gone" alert; transient failures are retried."""
    fail_calls: set = set()     # which probe calls (1-based) fail; None: all of them
    garbage = False

    def shell_class(self) -> type:
        outer = self

        class Flaky(LocalShell):
            calls = 0

            def run(self, command, timeout=60, stdin=None):
                if "PROBE_END" in command:
                    Flaky.calls += 1
                    if outer.garbage:
                        return subprocess.CompletedProcess(["ssh"], 0, "Welcome to the GB10\n", "")
                    if outer.fail_calls is None or Flaky.calls in outer.fail_calls:
                        return subprocess.CompletedProcess(["ssh"], 255, "", "ssh: connection refused")
                return super().run(command, timeout, stdin)
        return Flaky

    def assert_nothing_launched_or_gone(self, ev: list[dict]) -> None:
        self.assertEqual(self.launches, 0)
        self.assertNotIn("launched", [e["msg"] for e in ev])
        self.assertFalse(any("gone" in e["msg"] for e in ev))

    def test_persistent_failure_neither_launches_nor_reports_gone(self) -> None:
        self.fail_calls = None
        code, ev = self.run_render(launch=True)
        self.assertEqual(code, 1)
        self.assertIn("probe failed 5 times in a row", ev[-1]["msg"])
        self.assertIn("not treating the render as stopped", ev[-1]["msg"])
        self.assertTrue(ev[-1]["data"]["terminal"])
        self.assert_nothing_launched_or_gone(ev)

    def test_garbage_output_is_a_failed_probe_too(self) -> None:
        self.garbage = True
        code, ev = self.run_render(launch=True)
        self.assertEqual(code, 1)
        self.assertIn("probe failed 5 times in a row", ev[-1]["msg"])
        self.assert_nothing_launched_or_gone(ev)

    def test_transient_failures_are_retried_then_the_launch_and_the_follow_go_on(self) -> None:
        self.fail_calls = {1, 2, 5, 6}      # before the launch, and while following the render
        code, ev = self.run_render(launch=True)
        msgs = [e["msg"] for e in ev]
        self.assertIsNone(code)
        self.assertEqual(self.launches, 1)
        self.assertEqual(msgs[-1], "render finished")
        self.assertFalse(any("gone" in m for m in msgs))


CHECK_STAND_IN = """import json, sys
sys.path.insert(0, {repo!r})
from unittest import mock
from sim import check_batch as CB
status = {status!r}
arr = {{"arrangement_id": "TD5000", "environment": "tabletop", "table_clipping": {{
    "captures": [{{"phase": ph, "stop": "P1", "lighting": L, "n_table_px": 100, "n_clipped": 0,
                  "share": 0.5 if status == "fail" else 0.0}} for ph in CB.PHASES for L in CB.LIGHTINGS],
    "max_share_per_stop": {{"P1": 0.5 if status == "fail" else 0.0}}}}}}
res = {{"root": "X", "arrangements": [], "summary": {{
    "tag_checks": "SKIPPED", "true_alignment_corner_displacement_px": {{"n_views": 0}}, "leftover_folders_skipped": [],
    "label_sanity_flagged": {{}}, "table_clipping": CB.table_clipping_summary([arr])}}}}
argv = sys.argv[sys.argv.index("sim.check_batch") + 1:]
with mock.patch.object(CB, "check_set", return_value=res):
    sys.exit(CB.main([*argv, "--out", {out!r}]))
"""


class TestCheckStage(unittest.TestCase):
    """The check stage runs sim/check_batch.py's real CLI (check_set stood in): a table-clipping failure with clean
    label sanity exits 4, which the driver treats as a failed stage (terminal alert, driver exit 1)."""

    def run_check(self, status: str) -> tuple[int | None, list[dict]]:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "stand_in.py").write_text(CHECK_STAND_IN.format(repo=str(pipeline.REPO), status=status,
                                                                 out=str(t / "check.json")))
            spec = {"run_id": "check-test", "set_root": str(t / "set"), "local_python": [sys.executable, str(t / "stand_in.py")]}
            run = pipeline.Run(spec, t / "run")
            args = argparse.Namespace(from_stage="check", to_stage="check", launch=False, retry=False, spec=t / "spec.json")
            code = None
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    pipeline.run_stages(run, spec, args)
                except SystemExit as e:
                    code = e.code
            return code, [json.loads(x) for x in (run.dir / "events.jsonl").read_text().splitlines()]

    def test_clipping_failure_fails_the_stage(self) -> None:
        code, ev = self.run_check("fail")
        self.assertEqual(code, 1)
        self.assertEqual((ev[-1]["stage"], ev[-1]["level"], ev[-1]["data"]["terminal"], ev[-1]["data"]["rc"]),
                         ("check", "alert", True, 4))
        self.assertIn("check exited with rc 4", ev[-1]["msg"])

    def test_clipping_pass_finishes_the_run(self) -> None:
        code, ev = self.run_check("pass")
        self.assertIsNone(code)
        self.assertEqual([(e["stage"], e["level"]) for e in ev[-2:]], [("check", "info"), ("driver", "done")])
        self.assertEqual(ev[-2]["data"]["rc"], 0)


class TestProvenance(unittest.TestCase):
    def test_only_recorded_outputs_with_their_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "v_refL1.jsonl"
            f.write_text("{}\n")
            prov = {"key": "k", "out": t, "outputs": {str(f): {"sha256": provenance.sha256(f), "records": 1}}}
            self.assertEqual(provenance.output(prov, "{out}/v_{part}.jsonl", part="refL1"), f)
            with self.assertRaises(SystemExit):
                provenance.output(prov, "{out}/v_{part}.jsonl", part="design_refL1")   # not recorded
            f.write_text("{} \n")
            with self.assertRaises(SystemExit):
                provenance.output(prov, "{out}/v_{part}.jsonl", part="refL1")          # changed bytes

    def test_old_provenance_without_out_uses_the_shared_dir(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "evidence_provenance.json").write_text(json.dumps({"outputs": {"/d/a.jsonl": {}, "/d/b": {}}}))
            self.assertEqual(provenance.load(Path(t))["out"], "/d")


class TestEvidenceSummary(unittest.TestCase):
    def test_view_classes_follow_the_label_rules(self) -> None:
        vis, occ, rem = ({"condition": c} for c in ("visible", "occluded", "removed"))
        self.assertEqual(es.view_class(vis, {"visible_fraction": 1.0}), "present")
        self.assertEqual(es.view_class(occ, {"visible_fraction": 0.9}), "occluded C")
        self.assertEqual(es.view_class(occ, {"visible_fraction": 0.5}), "occluded p")
        self.assertEqual(es.view_class(occ, {"visible_fraction": 0.05}), "occluded H")
        self.assertEqual(es.view_class(rem, {"diagnostic": True}), "removed diag")
        self.assertEqual(es.view_class(rem, {"diagnostic": False}), "removed non-diag")

    def test_held_out_rates_confusion_and_uncapped_only(self) -> None:
        truth = {"A-visible-L1": "present", "A-removed-L1": "removed", "B-removed-L1": "removed"}
        eps = [{"policy": "G", "lighting": "L1", "cap_steps": None, "episode_id": "A-visible-L1",
                "decision_frac": {"present": 1.0, "removed": 0.0, "unknown": 0.0}},
               {"policy": "G", "lighting": "L1", "cap_steps": None, "episode_id": "A-removed-L1",
                "decision_frac": {"present": 0.25, "removed": 0.5, "unknown": 0.25}},
               {"policy": "G", "lighting": "L1", "cap_steps": None, "episode_id": "B-removed-L1",
                "decision_frac": {"present": 0.0, "removed": 0.0, "unknown": 1.0}},
               {"policy": "G", "lighting": "L1", "cap_steps": 2.0, "episode_id": "B-removed-L1",   # capped: ignored
                "decision_frac": {"present": 1.0, "removed": 0.0, "unknown": 0.0}}]
        g = es.held_out({"held_out_episodes": eps}, truth)["G"]["L1"]
        self.assertEqual(g["frr"], [0.0, 1])
        self.assertEqual(g["recall"], [0.5, 2])
        self.assertEqual(g["false_present"], [0.25, 2])
        self.assertEqual(g["coverage"], [1.75, 3])
        self.assertEqual(g["confusion"]["removed->unknown"], 1.25)
        self.assertEqual(g["accuracy_resolved"], [1.5, 1.75])   # (1 present + 0.5 removed) / (1 + 0.5 + 0.25)

    def test_gate_matches_the_stopping_rule(self) -> None:
        th = {"tau_q": 0.3, "tau_occ": 0.35}
        self.assertTrue(es.gate({"quality": 0.3, "obstruction": 0.35}, th))
        self.assertFalse(es.gate({"quality": 0.29, "obstruction": 0.0}, th))
        self.assertFalse(es.gate({"quality": 1.0, "obstruction": 0.36}, th))


class TestEvidenceSummaryFixedThresholds(unittest.TestCase):
    """main() on a set without design cases, at thresholds fixed elsewhere (the tabletop transfer diagnostic)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        t = self.t = Path(self.tmp.name)
        (t / "set").mkdir()
        (t / "out").mkdir()
        eps = {"A-visible-L1": {"arrangement_id": "A", "lighting": "L1", "condition": "visible", "truth": "present",
                                "stops": [{"stop": 0, "visible_fraction": 1.0}]},
               "A-occluded-L2": {"arrangement_id": "A", "lighting": "L2", "condition": "occluded", "truth": "present",
                                 "stops": [{"stop": 0, "visible_fraction": 0.0}]}}
        (t / "set" / "view_labels.json").write_text(json.dumps({"episodes": eps}))
        diag = t / "out" / "v_refL1.diag.jsonl"
        diag.write_text("\n".join(json.dumps({"episode_id": e, "stop": 0, "quality": q, "obstruction": o,
                                               "obstruction_v021": o, "match_score": 0.9, "align_ok": ok})
                                    for e, q, o, ok in (("A-visible-L1", 0.5, 0.1, True),
                                                        ("A-occluded-L2", 0.5, 0.9, False))) + "\n")
        (t / "run").mkdir()
        (t / "run" / "evidence_provenance.json").write_text(json.dumps(
            {"out": str(t / "out"), "outputs": {str(diag): {"sha256": provenance.sha256(diag)}}}))
        self.cand = t / "cand.json"
        self.cand.write_text("{}")
        self.fixed = {"tau_q": 0.3, "tau_occ": 0.35, "tau_match": 0.65, "source": str(self.cand),
                      "sha256": provenance.sha256(self.cand)}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def main(self, fixed: dict) -> int:
        spec = {"set_root": str(self.t / "set"), "summary": {
            "design": False, "illustrative": {"tau_q": 0.6, "tau_occ": 0.3}, "fixed_thresholds": fixed,
            "variants": {"v03_image": {"diag": "{out}/v_{part}.diag.jsonl", "obstruction": "obstruction"},
                         "v021_image": {"diag": "{out}/v_{part}.diag.jsonl", "obstruction": "obstruction_v021"}}}}
        (self.t / "spec.json").write_text(json.dumps(spec))
        with patch.object(sys, "argv", ["es", "--spec", str(self.t / "spec.json"), "--run-dir", str(self.t / "run")]), \
                contextlib.redirect_stdout(io.StringIO()):
            return es.main()

    def test_fixed_thresholds_and_no_design_cases(self) -> None:
        self.assertEqual(self.main({"v03_image": self.fixed, "v021_image": None}), 0)
        rep = json.loads((self.t / "run" / "summary.json").read_text())["variants"]
        v03, v021 = rep["v03_image"], rep["v021_image"]
        self.assertEqual(v03["chosen"], {"tau_q": 0.3, "tau_occ": 0.35, "tau_match": 0.65})
        self.assertIn("not calibrated on this set", v03["outcome"])
        self.assertEqual(v03["gate_chosen"]["L1|present"], [1, 1])
        self.assertEqual(v03["gate_chosen"]["L2|occluded H"], [0, 1])
        self.assertEqual(v03["align_refused"], {"L1": [0, 1], "L2": [1, 1]})
        self.assertIsNone(v03["held_out"])
        self.assertIsNone(v021["chosen"])
        self.assertIsNone(v021["gate_chosen"])
        self.assertIn("illustrative setting only", v021["outcome"])
        self.assertNotIn("DES cases", (self.t / "run" / "summary.md").read_text())

    def test_a_changed_threshold_source_is_refused(self) -> None:
        self.cand.write_text('{"edited": true}')
        with self.assertRaises(SystemExit) as cm:
            self.main({"v03_image": self.fixed, "v021_image": None})
        self.assertIn("not the recorded", str(cm.exception))


class TestQualityResampling(unittest.TestCase):
    def test_decisions_and_flips(self) -> None:
        th = {"tau_q": 0.3, "tau_occ": 0.35, "tau_match": 0.65}
        self.assertEqual(qr.decisions({"quality": 0.9, "obstruction": 0.1, "match_score": 0.2}, th),
                         {"gate": True, "clear_miss": True, "strong_match": False})
        self.assertEqual(qr.decisions({"quality": 0.2, "obstruction": 0.1, "match_score": 0.7}, th),
                         {"gate": False, "clear_miss": False, "strong_match": True})
        parts = {"sharpness": 1.0, "exposure": 1.0, "valid": 1.0}
        base = {("A-removed-L2", 4): {"quality": 0.29, "obstruction": 0.2, "match_score": 0.3, "quality_parts":
                                      {**parts, "sharpness": 0.29}}}
        other = {("A-removed-L2", 4): {"quality": 1.0, "obstruction": 0.2, "match_score": 0.3, "quality_parts": parts}}
        c = qr.compare(base, other, {"A-removed-L2": {"lighting": "L2"}}, {"s": th})
        r = c["per_lighting"]["L2"]
        self.assertEqual((r["n"], r["dq"]["0.3"], r["part_moved"]["sharpness"]), (1, 1, 1))
        self.assertEqual(r["flips"]["s"], {"gate": 1, "clear_miss": 1, "strong_match": 0})
        self.assertEqual(c["per_lighting"]["L1"]["n"], 0)

    def test_each_spec_says_how_its_set_was_chosen(self) -> None:
        cfg = Path(__file__).resolve().parents[1] / "configs" / "pipeline"
        desc = {n: json.loads((cfg / f"{n}.json").read_text())["summary"]["set_description"]
                for n in ("shelf_dev_v3_truealign", "tabletop_transfer11_truealign")}
        self.assertEqual(desc["shelf_dev_v3_truealign"], "random dev arrangements")
        self.assertIn("class coverage", desc["tabletop_transfer11_truealign"])
        self.assertNotIn("random dev", desc["tabletop_transfer11_truealign"])


if __name__ == "__main__":
    unittest.main()
