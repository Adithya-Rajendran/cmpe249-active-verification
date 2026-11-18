"""The frozen test path through tools/pipeline.py: the render process may be render_test.py (the freeze gate around
isaac_batch.py), a spec's "freeze" reaches sim.episodes as --freeze, and the two test specs (configs/pipeline) render
through sim/run_render_test.sh and compute exactly the development sets' evidence. Nothing here generates, renders or
reads a test arrangement."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SIM = "/mnt/bench-data/datasets/cmpe249-active-verification/sim"
CONFIGS = {"shelf": "arrange_shelf_v3", "tabletop": "arrange_tabletop_v1"}
KNOWN = {"jobs", "out", "part", "set"}

ARGV_STAND_IN = """import json, sys
open({out!r}, "w").write(json.dumps(sys.argv[1:]))
"""


def quota_total(quota: dict) -> int:
    return sum(v if isinstance(v, int) else sum(v.values()) for v in quota.values())


class RenderPatternTest(unittest.TestCase):
    def test_default_and_test_scripts(self) -> None:
        self.assertEqual(pipeline.render_pattern({"arrangements_dir": "arr_x"}), "isaac_batch.py --arrangements arr_x/")
        self.assertEqual(pipeline.render_pattern({"arrangements_dir": "arr_x", "script": "render_test.py"}),
                         "render_test.py --arrangements arr_x/")
        self.assertEqual(pipeline.self_excluding("render_test.py --arrangements arr_x/"),
                         "[r]ender_test.py --arrangements arr_x/")

    def test_any_other_script_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            pipeline.render_pattern({"arrangements_dir": "arr_x", "script": "python"})


class EpisodesArgsTest(unittest.TestCase):
    def test_freeze_only_when_the_spec_names_one(self) -> None:
        spec = {"set_root": "/s", "generation_log": "/g.json"}
        self.assertEqual(pipeline.episodes_args(spec), ["-m", "sim.episodes", "--root", "/s", "--contact-sheets",
                                                        "--generation-log", "/g.json"])
        self.assertEqual(pipeline.episodes_args({**spec, "freeze": "docs/FREEZE.md"})[-2:],
                         ["--freeze", "docs/FREEZE.md"])

    def test_the_episodes_stage_passes_the_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "stand_in.py").write_text(ARGV_STAND_IN.format(out=str(t / "argv.json")))
            spec = {"run_id": "episodes-test", "set_root": str(t / "set"), "generation_log": str(t / "g.json"),
                    "freeze": "docs/FREEZE.md", "local_python": [sys.executable, str(t / "stand_in.py")]}
            run = pipeline.Run(spec, t / "run")
            args = argparse.Namespace(from_stage="episodes", to_stage="episodes", launch=False, retry=False,
                                      spec=t / "spec.json")
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.run_stages(run, spec, args)
            argv = json.loads((t / "argv.json").read_text())
            self.assertEqual(argv[:2], ["-m", "sim.episodes"])
            self.assertEqual(argv[-2:], ["--freeze", "docs/FREEZE.md"])


class FrozenCodeCheckTest(unittest.TestCase):
    """The evidence stage of a test set ships code only when FREEZE.md passes sim/freeze.py and no shipped code path
    changed since the frozen commit (a temporary repository; nothing of the study is touched)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        for args in (["init", "-q"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"]):
            self.git(*args)
        for rel in (*pipeline.EVIDENCE_FROZEN, "perception/other.py"):
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text(f"{rel}\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "code")
        commit = self.git("rev-parse", "HEAD").strip()
        files = {rel: hashlib.sha256((self.repo / rel).read_bytes()).hexdigest() for rel in pipeline.EVIDENCE_FROZEN}
        doc = {"schema": "cmpe249.freeze.v1", "freeze_id": "t", "commit": commit, "files": files}
        (self.repo / "docs").mkdir()
        (self.repo / "docs" / "FREEZE.md").write_text("# FREEZE\n\n```json\n" + json.dumps(doc) + "\n```\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True, text=True).stdout

    def check(self) -> None:
        pipeline.frozen_code_check("docs/FREEZE.md", list(pipeline.EVIDENCE_CODE), repo=self.repo)

    def test_frozen_code_passes_and_any_later_code_change_fails(self) -> None:
        with self.assertRaises(RuntimeError):                       # FREEZE.md not committed yet
            self.check()
        self.git("add", "docs/FREEZE.md")
        self.git("commit", "-q", "-m", "freeze")
        self.check()
        (self.repo / "perception" / "other.py").write_text("changed after the freeze\n")   # shipped, not listed
        self.git("commit", "-q", "-am", "post-freeze change")
        with self.assertRaises(RuntimeError):
            self.check()


class EvidenceFreezeGateTest(unittest.TestCase):
    """stage_evidence's hook: a test spec runs frozen_code_check and fails the run when it refuses; a dev spec skips
    it."""

    def run_gate(self, spec: dict, error: Exception | None) -> tuple[list, list, BaseException | None]:
        calls, events = [], []

        class StubRun:
            def __init__(self) -> None:
                self.spec = spec

            def emit(self, stage, level, msg, **data) -> None:
                events.append((stage, level, msg))

            def fail(self, stage, msg, **data) -> None:
                events.append((stage, "alert", msg))
                raise SystemExit(1)

        def check(freeze, code_paths):
            calls.append((freeze, tuple(code_paths)))
            if error is not None:
                raise error
        raised = None
        with mock.patch.object(pipeline, "frozen_code_check", side_effect=check):
            try:
                pipeline.evidence_freeze_gate(StubRun(), {})
            except SystemExit as e:
                raised = e
        return calls, events, raised

    def test_a_test_spec_is_checked_and_a_refusal_fails_the_run(self) -> None:
        calls, events, raised = self.run_gate({"freeze": "docs/FREEZE.md"}, None)
        self.assertEqual(calls, [("docs/FREEZE.md", tuple(pipeline.EVIDENCE_CODE))])
        self.assertIsNone(raised)
        self.assertEqual(events[-1][:2], ("evidence", "info"))
        calls, events, raised = self.run_gate({"freeze": "docs/FREEZE.md"}, RuntimeError("changed since the freeze"))
        self.assertIsNotNone(raised)
        self.assertIn("changed since the freeze", events[-1][2])

    def test_a_dev_spec_is_not_checked(self) -> None:
        calls, events, raised = self.run_gate({}, None)
        self.assertEqual((calls, events, raised), ([], [], None))


class TestSpecsTest(unittest.TestCase):
    def test_specs(self) -> None:
        dev49 = json.loads((REPO / "configs" / "pipeline" / "tabletop_dev49_evidence.json").read_text())
        for env, cfg in CONFIGS.items():
            run = f"{env}_test"
            with self.subTest(spec=run):
                spec = json.loads((REPO / "configs" / "pipeline" / f"{run}.json").read_text())
                quota = json.loads((REPO / "configs" / "sim" / f"{cfg}.json").read_text())["slots"]["test"]["quota"]
                r = spec["render"]
                self.assertEqual(spec["run_id"], run)
                self.assertEqual(spec["freeze"], "docs/FREEZE.md")
                # the test's own GB10 run dir (frozen renderer copies), never the development sets' run dir
                self.assertEqual(spec["remote"], "configs/pipeline/remote.test.local.json")
                ignored = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", spec["remote"]])
                self.assertEqual(ignored.returncode, 0, "host details stay out of git")
                self.assertEqual(spec["set_root"], f"{SIM}/{run}")
                self.assertEqual(spec["generation_log"], f"{SIM}/{run}_geometry/arr/index/generation_log.json")
                self.assertEqual(r["script"], "render_test.py")
                self.assertEqual(r["arrangements_dir"], f"arr_{run}")
                self.assertEqual(r["expected"], quota_total(quota))
                self.assertEqual(r["expected"], 100)
                self.assertEqual(r["launch_script"].split(),
                                 ["run_render_test.sh", r["arrangements_dir"], spec["set_root"], r["log"],
                                  r["rc_file"], "FREEZE.md"])
                self.assertEqual(pipeline.render_pattern(r), f"render_test.py --arrangements arr_{run}/")
                self.assertNotIn("select", spec)
                self.assertNotIn("summary", spec)
                ev = spec["evidence"]
                self.assertEqual(ev["out_dir"], f"{spec['set_root']}/evidence")
                self.assertEqual(ev["jobs"], ["jobs_refL1.json"])              # no design cases in a test set
                self.assertEqual(ev["variants"], dev49["evidence"]["variants"])  # the development sets' evidence
                self.assertEqual(spec["local_python"], dev49["local_python"])
                for v in ev["variants"].values():
                    for text in (*v["args"], *v["outputs"]):
                        self.assertLessEqual(set(re.findall(r"{(\w+)}", text)), KNOWN, text)
                self.assertEqual(ev["variants"]["v03_image"]["args"][:4],
                                 ["-m", "perception.evidence_v03", "--align", "image"])

    def test_the_test_launcher_differs_from_the_dev_launcher_only_in_the_gate(self) -> None:
        def body(name: str) -> str:
            text = (REPO / "sim" / name).read_text()
            return "\n".join(x for x in text.splitlines() if x and not x.startswith("#"))
        expected = (body("run_render.sh")
                    .replace("RC=$4; shift 4", "RC=$4; FREEZE=$5; shift 5")
                    .replace("isaac_batch.py \\", "render_test.py \\")
                    .replace('--out "$OUT" "$@" > "$LOG" 2>&1',
                             '--out "$OUT" \\\n  --freeze "$FREEZE" "$@" > "$LOG" 2>&1'))
        self.assertEqual(body("run_render_test.sh"), expected)


if __name__ == "__main__":
    unittest.main()
