"""tools/test_validity.py: each validity check passes on consistent inputs and fails on the inconsistency it exists for.

RULE (as tests/test_freeze_gate.py): no arrangement of the study's test blocks is generated. Generation runs only on a
synthetic block (900000+) of an in-memory config, under a stand-in freeze file, which the synthetic block accepts."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from sim import arrange as A
from sim import freeze as F
from sim import testsplit as TS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import test_validity as TV  # noqa: E402

SYNTHETIC_START = 900000


def synthetic_ctx() -> tuple[A.Context, list[int]]:
    cfg = json.loads((REPO / TV.CONFIGS["shelf"]).read_text())
    blk = cfg["slots"]["test"]
    blk["primary_seed_start"] = SYNTHETIC_START
    blk["reserve"] = {f: [v + SYNTHETIC_START - 99000 for v in r] for f, r in blk["reserve"].items()}
    ctx = A.load_context(cfg)
    seeds = TS.test_block_seeds(ctx)
    assert min(seeds) >= TS.SYNTHETIC_SEED_MIN
    return ctx, seeds


class ValidityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        fz = t / "FREEZE.md"
        fz.write_text("# FREEZE stand-in\n")
        cls.freeze_sha = F.file_sha256(fz)
        freeze = {"record": {"freeze_id": "freeze-test"}, "sha256": cls.freeze_sha, "path": str(fz)}
        cls.ctx, seeds = synthetic_ctx()
        with TS.unlocked(freeze, seeds):
            arrs, log = A.generate_set(cls.ctx, "test", [0, 1])
        cls.geometry = t / "arr"
        TS.write_set(cls.geometry, cls.ctx, arrs, A.slot_list(cls.ctx, "test"), log)
        cls.ids = sorted(a["arrangement_id"] for a in arrs)
        cls.rec = {"freeze_id": "freeze-test", "files": {
            TV.CONFIGS["shelf"]: TV.sha(REPO / TV.CONFIGS["shelf"]), "sim/testsplit.py": TV.sha(REPO / "sim/testsplit.py"),
            "sim/isaac_batch.py": "b" * 64, "sim/isaac_rig.py": "c" * 64, "uv.lock": "d" * 64,
            "frozen/frozen_candidate.json": "e" * 64, TV.POSITIONS["shelf"]: "f" * 64,
            "frozen/model_A.json": "1" * 64, "frozen/model_A-nocue.json": "2" * 64}}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def checks(self, rep: TV.Report) -> dict[str, bool]:
        return {c["check"]: c["ok"] for c in rep.checks}

    def generation(self) -> tuple[TV.Report, dict]:
        rep = TV.Report()
        rows = TV.check_generation(rep, self.rec, self.freeze_sha, "shelf", self.geometry, ctx=self.ctx)
        return rep, rows

    def test_generation_checks(self) -> None:
        rep, rows = self.generation()
        got = self.checks(rep)
        self.assertEqual(sorted(rows), self.ids)
        self.assertFalse(got.pop("one arrangement per test slot"))     # 2 of the block's 100 slots were generated
        self.assertTrue(all(got.values()), got)
        # another FREEZE.md, or an arrangement file edited after generation
        rep = TV.Report()
        TV.check_generation(rep, self.rec, "0" * 64, "shelf", self.geometry, ctx=self.ctx)
        self.assertFalse(self.checks(rep)["each records this FREEZE.md, its freeze_id and the frozen testsplit"])
        p = self.geometry / f"{self.ids[0]}.json"
        original = p.read_text()
        try:
            arr = json.loads(original)
            arr["target_id"] = "something_else"
            p.write_text(json.dumps(arr, indent=1) + "\n")
            got = self.checks(self.generation()[0])
            self.assertFalse(got["each file has its index sha256"])
            self.assertFalse(got["arrangement_sha256 recomputes"])
        finally:
            p.write_text(original)

    def render_set(self, t: Path, rows: dict, **override) -> Path:
        root = t / "set"
        for aid, r in rows.items():
            (root / aid).mkdir(parents=True)
            h = {"arrangement_file_sha256": r["sha256"], "arrangement_sha256": r["arrangement_sha256"],
                 "isaac_batch_sha256": "b" * 64, "isaac_rig_sha256": "c" * 64,
                 "render_settings_sha256": TV.RENDER_SETTINGS["shelf"], **override}
            (root / aid / "labels.json").write_text(json.dumps({"hashes": h}))
        (root / "manifest.json").write_text(json.dumps({"arrangements": [
            {"arrangement_id": a, "split": "test", "environment": "shelf"} for a in rows]}))
        (root / "set.json").write_text(json.dumps({"split": "test", "environment": "shelf"}))
        return root

    def test_render_checks(self) -> None:
        _, rows = self.generation()
        with tempfile.TemporaryDirectory() as t:
            rep = TV.Report()
            TV.check_render(rep, self.rec, "shelf", rows, self.render_set(Path(t), rows))
            self.assertTrue(rep.ok, rep.checks)
        with tempfile.TemporaryDirectory() as t:
            rep = TV.Report()
            TV.check_render(rep, self.rec, "shelf", rows, self.render_set(Path(t), rows, isaac_batch_sha256="9" * 64))
            self.assertFalse(self.checks(rep)["each render ran the frozen isaac_batch.py and isaac_rig.py"])
        with tempfile.TemporaryDirectory() as t:
            rep = TV.Report()
            TV.check_render(rep, self.rec, "shelf", rows,
                            self.render_set(Path(t), rows, render_settings_sha256=TV.RENDER_SETTINGS["tabletop"]))
            self.assertFalse(self.checks(rep)["each render has the development render settings"])

    def test_run_checks(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "manifest.json").write_text("{}")
            (t / "view_labels.json").write_text("{}")
            cand = json.loads((REPO / "frozen" / "frozen_candidate.json").read_text())["thresholds"]
            cfg = {"config_version": TV.RUN_CONFIG_VERSION, "environment": "shelf", "scoring_split": "test",
                   "model_id": "v03_image", "bootstrap_B": 10000, "candidate_sha256": "e" * 64,
                   "positions_file_sha256": "f" * 64, "manifest_sha256": TV.sha(t / "manifest.json"),
                   "view_labels_sha256": TV.sha(t / "view_labels.json"), "thresholds": dict(cand),
                   "ranker_models": {"A": {"sha256": "1" * 64}, "A-nocue": {"sha256": "2" * 64}},
                   "evidence_sha256": "7" * 64, "side_file_sha256": "8" * 64}
            run = t / "run"
            run.mkdir()
            outs = {"/x/v03_image_refL1.jsonl": {"sha256": "7" * 64}, "/x/side_refL1.jsonl": {"sha256": "8" * 64}}
            for change, check in ((None, None), ({"bootstrap_B": 1000}, "frozen candidate, positions, set files and settings"),
                                  ({"thresholds": {**cand, "tau_match": 0.65}}, "the frozen thresholds"),
                                  ({"ranker_models": {"A": {"sha256": "3" * 64}, "A-nocue": {"sha256": "2" * 64}}},
                                   "the frozen A and A-nocue model files"),
                                  ({"side_file_sha256": "9" * 64},
                                   "the evidence of record and the descriptors of this set's evidence job")):
                with self.subTest(change=change):
                    (run / "config.json").write_text(json.dumps({"config": {**cfg, **(change or {})}}))
                    rep = TV.Report()
                    TV.check_run(rep, self.rec, "shelf", t, run, outs)
                    if check is None:
                        self.assertTrue(rep.ok, rep.checks)
                    else:
                        self.assertFalse(self.checks(rep)[check])

    def test_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "ck.sha256"
            p.write_text("".join(f"{v}  ./{k}\n" for k, v in TV.CHECKPOINTS.items()) + f"{'0' * 64}  other.pt\n")
            rep = TV.Report()
            TV.check_checkpoints(rep, p)
            self.assertTrue(rep.ok, rep.checks)
            p.write_text(f"{'0' * 64}  yolov8s-worldv2.pt\n")
            rep = TV.Report()
            TV.check_checkpoints(rep, p)
            self.assertFalse(rep.ok)

    def test_main_refuses_an_uncommitted_freeze(self) -> None:
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stderr(io.StringIO()):
            p = Path(t) / "FREEZE.md"
            p.write_text("x")
            self.assertEqual(TV.main(["--freeze", str(p), "--env", "shelf"]), 2)


if __name__ == "__main__":
    unittest.main()
