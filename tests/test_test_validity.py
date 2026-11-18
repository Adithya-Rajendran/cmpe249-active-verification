"""tools/test_validity.py: each validity check passes on consistent inputs and fails on the inconsistency it exists for.

RULE (as tests/test_freeze_gate.py): no arrangement of the study's test blocks is generated. Generation runs only on a
synthetic block (900000+) of an in-memory config, under a stand-in freeze file, which the synthetic block accepts."""
from __future__ import annotations

import hashlib
import json
import subprocess
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
            TV.CONFIGS["shelf"]: TV.sha(REPO / TV.CONFIGS["shelf"]),
            "sim/testsplit.py": TV.sha(REPO / "sim/testsplit.py"),
            "sim/isaac_batch.py": "b" * 64, "sim/isaac_rig.py": "c" * 64, "uv.lock": "d" * 64,
            "frozen/frozen_candidate.json": "e" * 64, TV.POSITIONS["shelf"]: "f" * 64,
            "frozen/model_A.json": "1" * 64, "frozen/model_A-nocue.json": "2" * 64, "sim/render_test.py": "4" * 64,
            "verify/run.py": "5" * 64, "analysis/bootstrap.py": "6" * 64}}

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

    def render_set(self, t: Path, rows: dict, manifest_extra: dict | None = None, **override) -> Path:
        manifest_extra = manifest_extra or {}
        root = t / "set"
        for aid, r in rows.items():
            (root / aid).mkdir(parents=True)
            h = {"arrangement_file_sha256": r["sha256"], "arrangement_sha256": r["arrangement_sha256"],
                 "isaac_batch_sha256": "b" * 64, "isaac_rig_sha256": "c" * 64,
                 "render_settings_sha256": TV.RENDER_SETTINGS["shelf"], **override}
            (root / aid / "labels.json").write_text(json.dumps({"hashes": h}))
        # the shelf's records carry no environment key (sim/episodes.py): absent means the shelf
        (root / "manifest.json").write_text(json.dumps({"arrangements": [
            {"arrangement_id": a, "split": "test", **manifest_extra} for a in rows]}))
        (root / "set.json").write_text(json.dumps({"split": "test"}))
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

    def test_a_manifest_of_another_environment_fails(self) -> None:
        _, rows = self.generation()
        with tempfile.TemporaryDirectory() as t:
            rep = TV.Report()
            TV.check_render(rep, self.rec, "shelf", rows,
                            self.render_set(Path(t), rows, manifest_extra={"environment": "tabletop"}))
            self.assertFalse(self.checks(rep)["manifest: the generated arrangements, split test, this environment"])

    def test_an_arrangement_off_its_frozen_slot_fails(self) -> None:
        p = self.geometry / f"{self.ids[0]}.json"
        original = p.read_text()
        try:
            for change, check in (
                    (lambda a: a["slot"].update(target=a["slot"]["target"] + 100),
                     "each is its frozen slot (family, M16 side, target, primary seed), and no slot twice"),
                    (lambda a: a.update(seed=a["seed"] + 7),
                     "each seed is its slot's primary seed or its family's reserve, as used, and unique"),
                    (lambda a: a["provenance"]["config"].update(sha256="0" * 64),
                     "each was made by the frozen generator, config and geometry (provenance hashes)")):
                with self.subTest(check=check):
                    arr = json.loads(original)
                    change(arr)
                    p.write_text(json.dumps(arr, indent=1) + "\n")
                    self.assertFalse(self.checks(self.generation()[0])[check])
        finally:
            p.write_text(original)

    def test_render_log(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "head.txt"
            line = (f"render_test.py sha256 {'4' * 64}; isaac_batch.py {'b' * 64}; isaac_rig.py {'c' * 64}; "
                    f"FREEZE.md copy {self.freeze_sha}")
            for text, ok in ((line, True), (line.replace("b" * 64, "9" * 64), False), ("something else", False)):
                with self.subTest(ok=ok):
                    p.write_text(text + "\n")
                    rep = TV.Report()
                    TV.check_render_log(rep, self.rec, self.freeze_sha, p)
                    self.assertEqual(rep.ok, ok, rep.checks)

    def test_run_checks(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "manifest.json").write_text("{}")
            (t / "view_labels.json").write_text("{}")
            cand = json.loads((REPO / "frozen" / "frozen_candidate.json").read_text())["thresholds"]
            cfg = {"config_version": TV.RUN_CONFIG_VERSION, "environment": "shelf", **TV.RUN_SETTINGS,
                   "candidate_sha256": "e" * 64,
                   "positions_file_sha256": "f" * 64, "manifest_sha256": TV.sha(t / "manifest.json"),
                   "view_labels_sha256": TV.sha(t / "view_labels.json"), "thresholds": dict(cand),
                   "ranker_models": {"A": {"sha256": "1" * 64}, "A-nocue": {"sha256": "2" * 64}},
                   "evidence_sha256": "7" * 64, "side_file_sha256": "8" * 64,
                   "sensitivity": {"k": [3], "lam": [0.7, 0.9, 1.2]},
                   "gate_off_ablation": {"policies": ["F", "R"], "tau_occ": 1.0},
                   "tau_match_sweep": {"frozen_tau_match": 0.7, "values_run": list(TV.SWEEP_VALUES)},
                   "by_family": {"levels": {"by_family": {}, "by_occluder_family": {}}},
                   "code_sha256": {"verify/run.py": "5" * 64, "analysis/bootstrap.py": "6" * 64}}
            run = t / "run"
            run.mkdir()
            outs = {"/x/v03_image_refL1.jsonl": {"sha256": "7" * 64}, "/x/side_refL1.jsonl": {"sha256": "8" * 64}}
            settings = "frozen candidate, positions, set files and every declared setting"
            declared = "every declared secondary run was on, as declared"
            for change, check in ((None, None),
                                  ({"bootstrap_B": 1000}, settings),
                                  ({"primary_budget": 2}, settings),
                                  ({"bootstrap_stratified": False}, settings),
                                  ({"code_sha256": {"verify/run.py": "0" * 64, "analysis/bootstrap.py": "6" * 64}},
                                   "the run's verify/ and analysis/ code is the frozen code"),
                                  ({"code_sha256": {"verify/run.py": "5" * 64}},
                                   "the run's verify/ and analysis/ code is the frozen code"),
                                  ({"thresholds": {**cand, "tau_match": 0.65}}, "the frozen thresholds"),
                                  ({"ranker_models": {"A": {"sha256": "3" * 64}, "A-nocue": {"sha256": "2" * 64}}},
                                   "the frozen A and A-nocue model files"),
                                  ({"side_file_sha256": "9" * 64},
                                   "the evidence of record and the descriptors of this set's evidence job"),
                                  ({"tau_match_sweep": None}, declared),
                                  ({"tau_match_sweep": {"frozen_tau_match": 0.7, "values_run": [0.4, 0.65]}},
                                   declared),
                                  ({"by_family": {"levels": {"by_family": {}}}}, declared)):
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

    def test_main_refuses_a_freeze_outside_the_repository(self) -> None:
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stderr(io.StringIO()):
            p = Path(t) / "FREEZE.md"
            p.write_text("x")
            self.assertEqual(TV.main(["--freeze", str(p), "--env", "shelf"]), 2)

    def test_evidence_checks(self) -> None:
        def at_head(rel: str) -> str:
            out = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True, check=True)
            return hashlib.sha256(out.stdout).hexdigest()
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        rec = {"freeze_id": "t", "files": {rel: at_head(rel) for rel in TV.EVIDENCE_FROZEN}}
        locked = TV.locked_versions(REPO / "uv.lock")
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            jobs = [{"episode_id": "X-removed-L1", "stop": s} for s in range(3)]
            (t / "jobs_refL1.json").write_text(json.dumps({"jobs": jobs}))

            def diag(name: str, match: float, refused=None) -> str:
                p = t / name
                p.write_text("".join(json.dumps({**j, "match_score": match, "quality": 0.9, "obstruction": 0.1,
                                                 "registration": {"refused": refused}}) + "\n" for j in jobs))
                return str(p)

            def provenance(c0_match: float, packages: dict) -> Path:
                outs = {str(t / n): {"records": 3, "sha256": "0" * 64}
                        for n in ("v03_image_refL1.jsonl", "c0_refL1.jsonl", "side_refL1.jsonl")}
                outs[diag("v03_image_refL1.diag.jsonl", 0.5)] = {"records": 3, "sha256": "0" * 64}
                outs[diag("c0_refL1.diag.jsonl", c0_match)] = {"records": 3, "sha256": "0" * 64}
                p = t / "prov.json"
                p.write_text(json.dumps({"code_commit": head, "set_root": str(t), "outputs": outs,
                                         "gb10_env": {"uv_lock_sha256": rec["files"]["uv.lock"],
                                                      "packages": packages}}))
                return p
            good = {"numpy": locked["numpy"], "torch": locked["torch"]}
            for c0_match, packages, failing in ((0.5, good, None),
                                                (0.51, good, "the harness C0 reproduces the evidence of record "
                                                             "(largest difference 0, equal refusals)"),
                                                (0.5, {**good, "numpy": "0.0.1"},
                                                 "every installed GB10 package has its uv.lock version")):
                with self.subTest(c0_match=c0_match, failing=failing):
                    rep = TV.Report()
                    TV.check_evidence(rep, rec, t, provenance(c0_match, packages))
                    if failing is None:
                        self.assertTrue(rep.ok, rep.checks)
                    else:
                        got = self.checks(rep)
                        self.assertFalse(got.pop(failing))
                        self.assertTrue(all(got.values()), got)


if __name__ == "__main__":
    unittest.main()
