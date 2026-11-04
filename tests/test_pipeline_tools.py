"""tools/pipeline.py helpers, tools/evidence_summary.py and tools/quality_resampling.py (stdlib tools, no GB10)."""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
