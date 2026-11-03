"""Tests for sim/arrange.py (the quota-slot arrangement generator) and configs/sim/positions_shelf_A.json. numpy only.

Pins: the slot list (counts, balance, determinism, no cross-family fallback, reserve use, retry cap), the asymmetric M16
(side definition, hidden stops on the slot's side), the revisit draws (determinism, bounds, one pose per stop shared by all
query captures, references unperturbed, frame check and redraw), hash provenance (arrangement_sha256 depends only on the
seed and the config content), valid placements (inside the compartments, no overlapping footprints, the target fits #3),
the matched occluded/removed layout, the dev/test seed guard, the design cases, and the per-stop pitch file.

Generation is slow, so the tests use six known-good (seed, attempt) pairs, one per slot class, found with
generate_slot on shelf_v1 (the retries before them are real failed tries).
"""
import contextlib
import copy
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sim import arrange as A
from sim import rig as R

REPO = Path(__file__).resolve().parents[1]
POS = REPO / "configs" / "sim" / "positions_shelf_A.json"
DRAFT_A = REPO / "data" / "raw" / "2026-10-10_far-camera" / "positions_far_A_draft.json"
CFG_REL = "configs/sim/arrange_shelf_v1.json"
# (seed, family, m16_side, attempt): the first successful try of the first slot of each class in the dev slot list
KNOWN = [(1002, "S8", None, 0), (1010, "white_box", None, 0), (1005, "ycb_box", None, 1), (1001, "IND28", None, 9),
         (1000, "M16", "L", 10), (1007, "M16", "R", 6)]


def _ctx():
    if not hasattr(_ctx, "c"):
        _ctx.c = A.load_context()
    return _ctx.c


def _gen(key):
    """The arrangement of one KNOWN entry (seed, family, side, attempt), cached."""
    if not hasattr(_gen, "cache"):
        _gen.cache = {}
    if key not in _gen.cache:
        seed, fam, side, attempt = key
        _gen.cache[key] = A.generate(seed, fam, side, attempt, ctx=_ctx())
    return _gen.cache[key]


class TestSeedGuard(unittest.TestCase):
    def test_dev_seeds_pass(self):
        self.assertEqual([A.check_dev_seed(s) for s in (0, 1, 9999)], [0, 1, 9999])
        self.assertEqual(A.arrangement_id(3), "SD0003")
        self.assertEqual(A.arrangement_id(9999), "SD9999")
        self.assertEqual(A.parse_seeds("0-2,7"), [0, 1, 2, 7])

    def test_test_seeds_are_refused(self):
        for s in (100000, 100001, 123456, 10**9):
            with self.subTest(seed=s), self.assertRaisesRegex(A.SeedError, "TEST"):
                A.check_dev_seed(s)
            with self.subTest(seed=s), self.assertRaisesRegex(A.SeedError, "TEST"):
                A.generate(s, "S8", ctx=_ctx())
        with self.assertRaisesRegex(A.SeedError, "TEST"):
            A.parse_seeds("99990-100002")

    def test_other_non_dev_seeds_are_refused(self):
        for s in (-1, 10000, 99999, "3", 3.0, True, None):
            with self.subTest(seed=s), self.assertRaises(A.SeedError):
                A.check_dev_seed(s)

    def test_test_slots_are_listed_but_never_generated(self):
        ctx = _ctx()
        slots = A.slot_list(ctx, "test")
        self.assertEqual(len(slots), 100)
        self.assertTrue(all(sl["primary_seed"] >= A.TEST_SEED_MIN for sl in slots))
        with self.assertRaisesRegex(A.SeedError, "TEST"):
            A.generate_slot(ctx, slots[0], {})
        with self.assertRaisesRegex(A.SeedError, "TEST"):
            A.generate_set(ctx, "test", [0])

    def test_cli_refuses_test_split_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "out"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(A.main(["--split", "test", "--slots", "all", "--out", str(out)]), 2)
            self.assertIn("TEST", err.getvalue())
            self.assertFalse(out.exists())

    def test_config_seed_ranges_are_the_generators(self):
        cfg = json.loads(A.DEFAULT_CONFIG_PATH.read_text())
        self.assertEqual(tuple(cfg["seeds"]["dev"]), A.DEV_SEED_RANGE)
        self.assertEqual(cfg["seeds"]["test_reserved_min"], A.TEST_SEED_MIN)
        bad = json.loads(json.dumps(cfg))
        bad["seeds"]["test_reserved_min"] = 50000
        with self.assertRaises(ValueError):
            A.load_config(bad)


def _by_class(slots):
    out = {}
    for sl in slots:
        key = sl["family"] + ("" if sl["m16_side"] is None else "-" + sl["m16_side"])
        out.setdefault(key, []).append(sl)
    return out


class TestSlotList(unittest.TestCase):
    def test_counts_dev_and_test(self):
        ctx = _ctx()
        for split, want in (("dev", {"S8": 12, "M16-L": 9, "M16-R": 9, "IND28": 6, "white_box": 12, "ycb_box": 12}),
                            ("test", {"S8": 20, "M16-L": 15, "M16-R": 15, "IND28": 10, "white_box": 20, "ycb_box": 20})):
            slots = A.slot_list(ctx, split)
            self.assertEqual({k: len(v) for k, v in _by_class(slots).items()}, want, split)
            self.assertEqual(len(slots), sum(want.values()))
            self.assertEqual([sl["index"] for sl in slots], list(range(len(slots))))
            self.assertEqual(len({sl["primary_seed"] for sl in slots}), len(slots))
            self.assertTrue(all((sl["m16_side"] is not None) == (sl["family"] == "M16") for sl in slots))
        self.assertEqual((sum(1 for _ in A.slot_list(ctx, "dev")), sum(1 for _ in A.slot_list(ctx, "test"))), (60, 100))

    def test_shares_are_the_confirmed_quotas(self):
        slots = A.slot_list(_ctx(), "dev")
        share = {f: sum(sl["family"] == f for sl in slots) / len(slots) for f in A.FAMILIES}
        self.assertEqual({k: round(v, 2) for k, v in share.items()}, {"S8": 0.2, "M16": 0.3, "IND28": 0.1, "white_box": 0.2, "ycb_box": 0.2})

    def test_deterministic_and_a_permutation(self):
        ctx = _ctx()
        a, b = A.slot_list(ctx, "dev"), A.slot_list(A.load_context(), "dev")
        self.assertEqual(a, b)
        sorted_classes = sorted((sl["family"], sl["m16_side"] or "") for sl in a)
        table = sorted((f, s) for f, s in A._quota_entries(ctx.config["slots"]["dev"]["quota"]))
        self.assertEqual(sorted_classes, sorted((f, s or "") for f, s in table))
        self.assertNotEqual([sl["family"] for sl in a], sorted(sl["family"] for sl in a))     # shuffled, not grouped
        self.assertNotEqual([(sl["family"], sl["m16_side"]) for sl in A.slot_list(ctx, "dev")],
                            [(sl["family"], sl["m16_side"]) for sl in A.slot_list(ctx, "test")][:60])

    def test_independent_of_config_key_order(self):
        cfg = json.loads((REPO / CFG_REL).read_text())
        shuffled = json.loads(json.dumps(cfg, sort_keys=True))               # sorted keys
        rev = {k: shuffled[k] for k in reversed(list(shuffled))}
        rev["slots"] = {k: rev["slots"][k] for k in reversed(list(rev["slots"]))}
        q = rev["slots"]["dev"]["quota"]
        rev["slots"]["dev"]["quota"] = {k: q[k] for k in reversed(list(q))}
        self.assertEqual(A.slot_list(cfg, "dev"), A.slot_list(rev, "dev"))
        self.assertEqual(A.load_config(cfg)[2], A.load_config(rev)[2])

    def test_seeds_are_disjoint_and_in_range(self):
        cfg = _ctx().config
        seen = set()
        for split in ("dev", "test"):
            seeds = [sl["primary_seed"] for sl in A.slot_list(cfg, split)]
            for fam, lst in cfg["slots"][split]["reserve"].items():
                seeds += list(lst)
            self.assertEqual(len(seeds), len(set(seeds)), split)
            self.assertTrue(seen.isdisjoint(seeds))
            seen |= set(seeds)
            for v in seeds:
                self.assertTrue(v >= A.TEST_SEED_MIN if split == "test" else 0 <= v <= 9999, v)
        self.assertEqual(cfg["slots"]["retry_max"], 20)

    def test_bad_slot_configs_are_refused(self):
        cfg = json.loads((REPO / CFG_REL).read_text())
        dup = copy.deepcopy(cfg)
        dup["slots"]["dev"]["reserve"]["S8"][0] = dup["slots"]["dev"]["primary_seed_start"]
        with self.assertRaisesRegex(ValueError, "twice"):
            A.load_config(dup)
        out_of_range = copy.deepcopy(cfg)
        out_of_range["slots"]["dev"]["reserve"]["S8"][0] = 15000
        with self.assertRaisesRegex(ValueError, "outside"):
            A.load_config(out_of_range)
        sided = copy.deepcopy(cfg)
        sided["slots"]["dev"]["quota"]["S8"] = {"L": 6, "R": 6}
        with self.assertRaisesRegex(ValueError, "sides"):
            A.load_config(sided)

    def test_first_slot_of_each_class(self):
        slots = A.slot_list(_ctx(), "dev")
        first = A.first_slot_per_class(slots)
        self.assertEqual(len(first), 6)
        self.assertEqual({(slots[i]["family"], slots[i]["m16_side"]) for i in first},
                         {("S8", None), ("M16", "L"), ("M16", "R"), ("IND28", None), ("white_box", None), ("ycb_box", None)})
        for i in first:
            same = [sl for sl in slots[:i] if (sl["family"], sl["m16_side"]) == (slots[i]["family"], slots[i]["m16_side"])]
            self.assertEqual(same, [])
        self.assertEqual(A.parse_slot_indices("first-per-class", slots), sorted(first))
        self.assertEqual(A.parse_slot_indices("0-2,9", slots), [0, 1, 2, 9])
        with self.assertRaises(ValueError):
            A.parse_slot_indices("60", slots)

    def test_known_pairs_are_the_first_slots(self):
        slots = A.slot_list(_ctx(), "dev")
        first = {(slots[i]["family"], slots[i]["m16_side"]): slots[i]["primary_seed"] for i in A.first_slot_per_class(slots)}
        for seed, fam, side, _ in KNOWN:
            self.assertEqual(first[(fam, side)], seed)


def _tiny_config():
    """The shelf config with a handful of cheap slots (S8 3, white_box 3) so slot runs are fast."""
    cfg = json.loads((REPO / CFG_REL).read_text())
    cfg["slots"]["dev"]["quota"] = {"S8": 3, "M16": {"L": 0, "R": 0}, "IND28": 0, "white_box": 3, "ycb_box": 0}
    return cfg


def _forcing(fail):
    """A try_place replacement that fails every (seed, attempt) for which fail(seed, attempt, family) is true and records
    the calls; otherwise the real try_place."""
    real = A.try_place
    calls = []

    def fake(ctx, seed, attempt, family, side):
        calls.append((seed, attempt, family, side))
        if fail(seed, attempt, family):
            return None, "occluder", "forced"
        return real(ctx, seed, attempt, family, side)

    return fake, calls


class TestSlotGeneration(unittest.TestCase):
    def test_retries_stay_in_the_family_and_stop_at_the_cap(self):
        ctx = A.load_context(_tiny_config())
        slot = next(sl for sl in A.slot_list(ctx, "dev") if sl["family"] == "S8")
        fake, calls = _forcing(lambda seed, attempt, fam: seed == slot["primary_seed"])      # the primary seed never works
        with mock.patch.object(A, "try_place", fake):
            arr, entry = A.generate_slot(ctx, slot, {})
        retry_max = ctx.config["slots"]["retry_max"]
        primary = [c for c in calls if c[0] == slot["primary_seed"]]
        self.assertEqual([c[1] for c in primary], list(range(retry_max + 1)))                # attempt 0, then retries 1..20, no more
        forced = [t for t in entry["failed_tries"] if t["seed"] == slot["primary_seed"]]
        self.assertEqual([t["attempt"] for t in forced], list(range(retry_max + 1)))
        self.assertEqual(entry["failed_tries"][:retry_max + 1], forced)                      # the primary seed's tries come first
        self.assertEqual({c[2] for c in calls}, {"S8"})                                      # never another family
        self.assertEqual({c[3] for c in calls}, {None})
        reserve = ctx.config["slots"]["dev"]["reserve"]["S8"]
        self.assertEqual(entry["replacements"], [{"from_seed": slot["primary_seed"], "to_seed": reserve[0], "reserve_rank": 0,
                                                  "after_failed_tries": retry_max + 1}])
        self.assertEqual((arr["seed"], arr["occluder_family"], arr["slot"]["used_seed"], arr["slot"]["reserve_rank"],
                          arr["slot"]["primary_seed"], arr["arrangement_id"]), (reserve[0], "S8", reserve[0], 0, slot["primary_seed"], A.arrangement_id(reserve[0])))
        self.assertEqual(arr["slot"]["failed_tries"][0], {"seed": slot["primary_seed"], "attempt": 0, "reason": "occluder", "target": "forced"})

    def test_a_retry_within_the_seed_is_used_before_the_reserve(self):
        ctx = A.load_context(_tiny_config())
        slot = next(sl for sl in A.slot_list(ctx, "dev") if sl["family"] == "white_box")
        fake, calls = _forcing(lambda seed, attempt, fam: attempt < 3)
        with mock.patch.object(A, "try_place", fake):
            arr, entry = A.generate_slot(ctx, slot, {})
        self.assertEqual((arr["seed"], arr["slot"]["used_attempt"], arr["slot"]["reserve_rank"], len(entry["failed_tries"])),
                         (slot["primary_seed"], 3, None, 3))
        self.assertEqual(entry["replacements"], [])
        self.assertFalse(entry["replaced"])

    def test_a_failure_never_changes_the_family(self):
        """The old generator turned a failed M16 draw into IND28. Whatever fails, the family is the slot's."""
        ctx = A.load_context(_tiny_config())
        slots = [sl for sl in A.slot_list(ctx, "dev") if sl["family"] in ("S8", "white_box")]
        fake, calls = _forcing(lambda seed, attempt, fam: seed % 2 == 0 and attempt < 7)
        with mock.patch.object(A, "try_place", fake):
            for sl in slots:
                arr, _ = A.generate_slot(ctx, sl, {})
                self.assertEqual(arr["occluder_family"], sl["family"])
                self.assertEqual(arr["objects"]["occluder"]["family"], sl["family"])
        self.assertTrue(all(c[2] == next(s["family"] for s in slots if s["primary_seed"] == c[0] or c[0] in
                                         ctx.config["slots"]["dev"]["reserve"][s["family"]]) for c in calls))

    def test_reserve_list_is_consumed_in_slot_order_and_can_run_out(self):
        cfg = _tiny_config()
        cfg["slots"]["dev"]["reserve"]["S8"] = cfg["slots"]["dev"]["reserve"]["S8"][:2]
        ctx = A.load_context(cfg)
        s8 = [sl for sl in A.slot_list(ctx, "dev") if sl["family"] == "S8"]
        primaries = {sl["primary_seed"] for sl in s8}
        fake, calls = _forcing(lambda seed, attempt, fam: seed in primaries)
        used: dict = {}
        with mock.patch.object(A, "try_place", fake):
            arrs = [A.generate_slot(ctx, sl, used)[0] for sl in s8[:2]]
            self.assertEqual([a["slot"]["reserve_rank"] for a in arrs], [0, 1])
            self.assertEqual([a["seed"] for a in arrs], cfg["slots"]["dev"]["reserve"]["S8"])
            with self.assertRaises(A.ReserveExhausted):
                A.generate_slot(ctx, s8[2], used)

    def test_a_subset_reproduces_the_full_set(self):
        ctx = A.load_context(_tiny_config())
        slots = A.slot_list(ctx, "dev")
        s8 = [sl for sl in slots if sl["family"] == "S8"]
        fake, _ = _forcing(lambda seed, attempt, fam: seed == s8[0]["primary_seed"])      # slot 0 of the family uses reserve 0
        with mock.patch.object(A, "try_place", fake):
            full, full_log = A.generate_set(ctx, "dev")
            sub, sub_log = A.generate_set(ctx, "dev", [s8[1]["index"], s8[2]["index"]])
        by_id = {a["arrangement_id"]: a for a in full}
        self.assertEqual(len(full), 6)
        self.assertEqual(len(sub), 2)
        for a in sub:
            self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(by_id[a["arrangement_id"]], sort_keys=True))
        self.assertEqual(full_log["reserve_used"], {"S8": 1})
        self.assertEqual(len(full_log["replacements"]), 1)
        c = full_log["counts_by_class"]["S8"]
        self.assertEqual((c["slots"], c["replaced"]), (3, 1))
        self.assertGreaterEqual(c["failed_tries"], 21)
        self.assertEqual(sub_log["n_selected"], 2)
        self.assertFalse(sub_log["slots"][0]["selected"])                                  # slot 0 was resolved to advance the cursor only
        # every retry and replacement is in the set-level log
        self.assertEqual(len(full_log["retries"]), sum(e["n_failed_tries"] for e in full_log["slots"]))
        self.assertGreaterEqual(len([r for r in full_log["retries"] if r["seed"] == s8[0]["primary_seed"]]), 21)
        self.assertTrue(all({"slot", "family", "m16_side", "seed", "attempt", "reason"} <= set(r) for r in full_log["retries"]))

    def test_cli_writes_the_index_in_a_subfolder(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "ARR"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(A.main(["--slots", "2,10", "--design", "--out", str(out)]), 0)
            top = sorted(p.name for p in out.glob("*.json"))
            self.assertEqual(top, ["DES0.json", "DES1.json", "DES2.json", "DES3.json", "SD1002.json", "SD1010.json"])
            for name in top:
                self.assertEqual(json.loads((out / name).read_text())["schema"], A.ARRANGEMENT_SCHEMA)       # the glob is only arrangements
            idx = out / "index"
            self.assertEqual(sorted(p.name for p in idx.glob("*.json")), ["generation_log.json", "index.json", "slots.json"])
            self.assertEqual(json.loads((idx / "index.json").read_text())["schema"], A.INDEX_SCHEMA)
            rows = {r["arrangement_id"]: r for r in json.loads((idx / "index.json").read_text())["arrangements"]}
            self.assertEqual((rows["SD1002"]["occluder_family"], rows["SD1002"]["design_case"], rows["SD1002"]["slot"]), ("S8", False, 2))
            self.assertEqual((rows["DES1"]["occluder_family"], rows["DES1"]["m16_side"], rows["DES1"]["design_case"]), ("M16", "R", True))
            slots = json.loads((idx / "slots.json").read_text())
            self.assertEqual((slots["schema"], len(slots["slots"]), slots["split"]), (A.SLOTS_SCHEMA, 60, "dev"))
            log = json.loads((idx / "generation_log.json").read_text())
            self.assertEqual((log["schema"], log["n_selected"]), (A.GENERATION_LOG_SCHEMA, 2))


class TestHashProvenance(unittest.TestCase):
    def test_hash_ignores_config_spelling_and_python(self):
        a = A.generate(1002, "S8", ctx=_ctx())                                         # absolute default path
        b = A.generate(1002, "S8", ctx=A.load_context(CFG_REL))                        # repo-relative, the CLI's spelling
        c = A.generate(1002, "S8", ctx=A.load_context(json.loads((REPO / CFG_REL).read_text())))   # in-memory dict
        self.assertEqual(a["arrangement_sha256"], b["arrangement_sha256"])
        self.assertEqual(a["arrangement_sha256"], c["arrangement_sha256"])
        self.assertNotEqual(a["unhashed"]["config_path"], b["unhashed"]["config_path"])
        self.assertIsNone(c["unhashed"]["config_path"])
        d = copy.deepcopy(a)
        d["unhashed"]["python"] = "3.99.0"
        d["unhashed"]["config_path"] = "/elsewhere/checkout/arrange_shelf_v1.json"
        self.assertEqual(A.arrangement_hash(d), a["arrangement_sha256"])
        self.assertEqual(A.arrangement_hash(a), a["arrangement_sha256"])
        text = json.dumps(a["provenance"])
        self.assertNotIn("python", text)
        self.assertNotIn("config_path", text)
        self.assertNotIn("arrange_shelf_v1.json", json.dumps({k: v for k, v in a.items() if k != "unhashed"}))

    def test_hash_changes_with_content(self):
        a = A.generate(1002, "S8", ctx=_ctx())
        self.assertNotEqual(a["arrangement_sha256"], A.generate(1010, "white_box", ctx=_ctx())["arrangement_sha256"])
        cfg = json.loads((REPO / CFG_REL).read_text())
        cfg["clearances_m"]["wall"] = 0.0101
        self.assertNotEqual(A.load_config(cfg)[2], _ctx().provenance["config"]["sha256"])
        d = copy.deepcopy(a)
        d["objects"]["target"]["pose"]["yaw_deg"] += 1
        self.assertNotEqual(A.arrangement_hash(d), a["arrangement_sha256"])

    def test_dict_order_cannot_change_the_arrangement(self):
        cfg = json.loads((REPO / CFG_REL).read_text())
        rev = {k: cfg[k] for k in reversed(list(cfg))}
        rev["occluder"] = {k: rev["occluder"][k] for k in reversed(list(rev["occluder"]))}
        rev["occluder"]["families"] = {k: v for k, v in reversed(list(rev["occluder"]["families"].items()))}
        a, b = A.generate(1002, "S8", ctx=_ctx()), A.generate(1002, "S8", ctx=A.load_context(rev))
        self.assertEqual(a["arrangement_sha256"], b["arrangement_sha256"])
        self.assertEqual(json.dumps(a["objects"], sort_keys=True), json.dumps(b["objects"], sort_keys=True))

    def test_cli_and_library_agree(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "ARR"
            with contextlib.redirect_stdout(io.StringIO()):
                A.main(["--slots", "2", "--out", str(out)])
            written = json.loads((out / "SD1002.json").read_text())
        slot_arr, _ = A.generate_slot(_ctx(), A.slot_list(_ctx(), "dev")[2], {})
        strip = lambda a: json.dumps({k: v for k, v in a.items() if k != "unhashed"}, sort_keys=True)     # noqa: E731
        self.assertEqual(strip(written), strip(slot_arr))
        self.assertEqual(written["arrangement_sha256"], slot_arr["arrangement_sha256"])
        # the slot record is part of the hashed arrangement: the same try without it has another hash
        self.assertNotEqual(written["arrangement_sha256"], A.generate(1002, "S8", ctx=_ctx())["arrangement_sha256"])


class TestDeterminism(unittest.TestCase):
    def test_same_try_same_arrangement(self):
        a = _gen(KNOWN[0])
        b = A.generate(1002, "S8", ctx=A.load_context())             # fresh context, same files
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_attempts_and_seeds_differ(self):
        sigs = {json.dumps(_gen(k)["objects"], sort_keys=True) for k in KNOWN}
        self.assertEqual(len(sigs), len(KNOWN))
        # the retry stream of (seed, attempt) differs from the plain seed's
        self.assertNotEqual(A.Draw([1002]).u(), A.Draw([1002, 1]).u())
        self.assertNotEqual(A.Draw([1002, 1]).u(), A.Draw([1002, 2]).u())

    def test_provenance_hashes(self):
        a = _gen(KNOWN[0])
        p = a["provenance"]
        self.assertEqual(p["generator"]["sha256"], A.sha256_file(REPO / "sim" / "arrange.py"))
        self.assertEqual(p["config"]["sha256"], A.sha256_json(json.loads(A.DEFAULT_CONFIG_PATH.read_text())))
        self.assertEqual(a["unhashed"]["config_file_sha256"], A.sha256_file(A.DEFAULT_CONFIG_PATH))
        self.assertEqual(p["positions"]["sha256"], A.sha256_file(POS))
        self.assertEqual((a["seed"], a["split"], a["slot"], a["draws"]["family"]), (1002, "dev", None, "S8"))


class TestPlacements(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.cfg = self.ctx.config
        self.cl = self.cfg["clearances_m"]

    def _inside(self, body, comp, front=None):
        c = self.ctx.comps[comp]
        fp = body.footprint()
        front = self.cl["front"] if front is None else front
        tol = 1e-6
        self.assertAlmostEqual(body.base[2], c["z"][0], places=6)
        self.assertLessEqual(body.top, c["z"][1] - self.cl["ceiling"] + tol)
        self.assertGreaterEqual(fp[:, 0].min(), c["x"][0] + self.cl["wall"] - tol)
        self.assertLessEqual(fp[:, 0].max(), c["x"][1] - self.cl["wall"] + tol)
        self.assertGreaterEqual(fp[:, 1].min(), c["y"][0] + front - tol)
        self.assertLessEqual(fp[:, 1].max(), c["y"][1] - self.cl["back"] + tol)

    def test_every_slot_class(self):
        cov_min = self.cfg["occluder"]["p3_coverage_min"]
        for key in KNOWN:
            with self.subTest(slot=key):
                a = _gen(key)
                o = a["objects"]
                t, occ, dis = o["target"], o["occluder"], o["distractors"]
                self.assertEqual((a["occluder_family"], occ["family"], a["m16_side"]), (key[1], key[1], key[2]))
                tb = A.body_of(t, ctx=self.ctx)
                # target: a configured object, fits #3, seen whole from every stop
                self.assertIn(t["obj_id"], self.cfg["target"]["objects"])
                self.assertLessEqual(t["size_m"][2], min(self.cfg["target"]["max_height_m"], 0.165 - self.cl["ceiling"]))
                self._inside(tb, "c3")
                self.assertTrue(all(A.seen_whole(self.ctx, c, tb) for c in self.ctx.cams))
                # occluder: placed in #3 in front of the target, covering it at the start stop
                pb, kb = A.body_of(occ, "placed", self.ctx), A.body_of(occ, "parked", self.ctx)
                self._inside(pb, "c3", front=self.cl["occluder_front"])
                self.assertLess(pb.footprint()[:, 1].max(), tb.footprint()[:, 1].min())
                self.assertTrue(A.separated(pb.footprint(), tb.footprint()))
                self.assertGreaterEqual(A.coverage(self.ctx.start_cam, tb, pb), cov_min)
                covs = {c.name: A.coverage(c, tb, pb) for c in self.ctx.cams}
                oc = self.cfg["occluder"]
                if occ["family"] == "IND28":                    # predeclared indistinguishable: every stop covered
                    self.assertTrue(a["predeclared_indistinguishable"])
                    self.assertGreaterEqual(min(covs.values()), cov_min)
                else:                                           # resolvable: >= 2 stops where the hulls do not touch
                    self.assertFalse(a["predeclared_indistinguishable"])
                    self.assertGreaterEqual(sum(v <= oc["clear_coverage_max"] for v in covs.values()), oc["min_clear_stops"])
                # distractors: 2-5 distinct other models, inside, clear of each other / the target / the placed occluder
                lo, hi = self.cfg["distractors"]["count"]
                self.assertTrue(lo <= len(dis) <= hi)
                ids = [d["obj_id"] for d in dis]
                self.assertEqual(len(set(ids)), len(ids))
                self.assertNotIn(t["obj_id"], ids)
                if occ["kind"] == "ycbv":
                    self.assertNotIn(occ["obj_id"], ids)
                bodies = []
                for d in dis:
                    comp = d["pose"]["compartment"]
                    db = A.body_of(d, ctx=self.ctx)
                    self._inside(db, comp)
                    self.assertIn(d["obj_id"], self.cfg["distractors"][f"{comp}_objects"])
                    if comp == "c3":
                        self.assertTrue(A.separated(db.footprint(), tb.footprint(), self.cl["between_objects"]))
                        self.assertTrue(A.separated(db.footprint(), pb.footprint(), self.cl["between_objects"]))
                        self.assertFalse(any(A.may_occlude(c, db, tb) for c in self.ctx.cams))
                    for comp2, b2 in bodies:
                        if comp2 == comp:
                            self.assertTrue(A.separated(db.footprint(), b2.footprint(), self.cl["between_objects"]))
                    bodies.append((comp, db))
                # parked: occludes nothing from any stop, outside the target's context window
                comp = occ["poses"]["parked"]["compartment"]
                self._inside(kb, comp)
                for c in self.ctx.cams:
                    self.assertFalse(A.may_occlude(c, kb, tb))
                    self.assertFalse(any(A.may_occlude(c, kb, b) for _, b in bodies))
                    self.assertLess(A.overlap_area(A.projected_hull(c, kb.corners()), A.target_window(c, tb, 1.0)), 1.0)

    def test_targets_fit_compartment_3(self):
        clear = self.ctx.comps["c3"]["z"][1] - self.cl["ceiling"]
        for obj in self.cfg["target"]["objects"]:
            ext = A.rest_extents(self.ctx.item(obj), self.ctx.rest("upright"))
            self.assertLessEqual(ext[2], clear, self.ctx.item(obj)["name"])
        for obj in self.cfg["distractors"]["c3_objects"]:
            self.assertLessEqual(A.rest_extents(self.ctx.item(obj), self.ctx.rest("upright"))[2], clear)

    def test_rest_poses_are_rotations(self):
        for name, m in self.cfg["rest_poses"].items():
            if name == "note":
                continue
            m = np.asarray(m, float)
            self.assertTrue(np.allclose(m @ m.T, np.eye(3)), name)
            self.assertAlmostEqual(float(np.linalg.det(m)), 1.0, msg=name)
        # sugar box on its long side: 17.6 wide, 5.0 deep, 9.4 tall
        w, d, h = A.rest_extents(self.ctx.item(3), self.ctx.rest("side"))
        self.assertEqual((round(w, 3), round(d, 3), round(h, 3)), (0.176, 0.05, 0.094))


class TestM16Side(unittest.TestCase):
    """M16 is asymmetric: SIDE R = the board's centre is displaced toward +x (image right at the start stop) from the sight
    line start stop -> target centre, measured on the board's plane; L = toward -x."""

    def test_side_rule_in_the_config(self):
        self.assertEqual(_ctx().config["occluder"]["families"]["M16"]["side_rule"], {"opposite_near_max": 0.10, "adjacent_min": 0.5})

    def test_side_sign(self):
        self.assertEqual((A.side_sign("R"), A.side_sign("L")), (1.0, -1.0))
        with self.assertRaises(ValueError):
            A.side_sign(None)

    def test_side_rule_on_synthetic_coverages(self):
        rule = {"opposite_near_max": 0.5, "adjacent_min": 0.5}
        ok = A.side_rule_ok
        # arc order P1..P5, start 2 (P3)
        self.assertTrue(ok([0, 0, 1, 1, 0], 2, "R", rule, 0.001))            # CCHHC
        self.assertTrue(ok([0, 1, 1, 0, 0], 2, "L", rule, 0.001))            # CHHCC
        self.assertTrue(ok([0, 0.3, 1, 0.9, 0.4], 2, "R", rule, 0.001))     # a partial far-side neighbour is allowed
        self.assertFalse(ok([0, 0, 1, 1, 0], 2, "L", rule, 0.001))           # the mirror image is not an L board
        self.assertFalse(ok([0, 1, 1, 0, 0], 2, "R", rule, 0.001))
        self.assertFalse(ok([0, 0.7, 1, 0.7, 0], 2, "R", rule, 0.001))       # centred: both neighbours covered
        self.assertFalse(ok([0.2, 0, 1, 1, 0], 2, "R", rule, 0.001))         # the far opposite stop is touched
        self.assertFalse(ok([0, 0, 1, 0.2, 0], 2, "R", rule, 0.001))         # the board does not reach the adjacent stop
        self.assertFalse(ok([0, 0, 1], 2, "R", rule, 0.001))                 # no stop on the board's side
        self.assertTrue(ok([0, 0.1, 1, 1, 0.9], 2, "R", rule, 0.001))

    def test_board_shift_follows_the_side(self):
        ctx = _ctx()
        spec = ctx.config["occluder"]["families"]["M16"]
        lo, hi = spec["side_shift_m"]
        tb = A.Body((0.06, 0.06, 0.05), (0.0, 0.2, 0.0), 0.0)
        for side, sign in (("R", 1.0), ("L", -1.0)):
            for k in range(20):
                b = A._board(ctx, A.Draw([k, 1]), spec, tb, side)
                self.assertIsNotNone(b)
                self.assertEqual(np.sign(b["shift_m"]), sign)
                self.assertTrue(lo <= abs(b["shift_m"]) <= hi)
                sl = A.sightline_x(ctx.start_cam, tb.centre(), b["body"].base[1])
                self.assertEqual(np.sign(b["body"].base[0] - sl), sign)         # the board centre is on that side of the sight line

    def test_hidden_stops_fall_on_the_slots_side(self):
        names = [c.name for c in _ctx().cams]
        start = _ctx().start_index
        for key in KNOWN[-2:]:
            seed, fam, side, _ = key
            a = _gen(key)
            self.assertEqual((a["m16_side"], a["objects"]["occluder"]["side"], a["slot"]), (side, side, None))
            cov = a["analytic"]["placed_coverage"]
            covered = [n for n in names if cov[n] > _ctx().config["occluder"]["clear_coverage_max"]]
            hidden = [n for n in names if cov[n] >= _ctx().config["occluder"]["p3_coverage_min"]]
            idx = [names.index(n) for n in covered]
            if side == "R":
                self.assertTrue(all(i >= start - 1 for i in idx), (covered, cov))        # nothing beyond the nearest left stop
                self.assertTrue(all(names.index(n) >= start for n in hidden), hidden)
                self.assertEqual(cov["P1"], 0.0)
                self.assertLessEqual(cov["P2"], 0.10)               # the near opposite stop renders as a C view
                self.assertGreaterEqual(cov["P4"], 0.5)
            else:
                self.assertTrue(all(i <= start + 1 for i in idx), (covered, cov))
                self.assertTrue(all(names.index(n) <= start for n in hidden), hidden)
                self.assertEqual(cov["P5"], 0.0)
                self.assertLessEqual(cov["P4"], 0.10)
                self.assertGreaterEqual(cov["P2"], 0.5)
            self.assertIn("P3", hidden)

    def test_l_and_r_are_mirror_images_in_pattern(self):
        l, r = _gen(KNOWN[4])["analytic"]["placed_coverage"], _gen(KNOWN[5])["analytic"]["placed_coverage"]
        self.assertGreater(l["P2"], l["P4"])
        self.assertGreater(r["P4"], r["P2"])

    def test_design_cases_carry_the_sides(self):
        d1, d2 = A.design_arrangement(1, ctx=_ctx()), A.design_arrangement(2, ctx=_ctx())
        self.assertEqual((d1["occluder_family"], d1["m16_side"], d2["occluder_family"], d2["m16_side"]), ("M16", "R", "M16", "L"))
        self.assertGreater(d1["objects"]["occluder"]["poses"]["placed"]["base"][0], 0)       # pick_M16+3: toward +x
        self.assertLess(d2["objects"]["occluder"]["poses"]["placed"]["base"][0], 0)          # pick_M16-5: toward -x
        self.assertEqual(d1["analytic"]["placed_coverage"]["P1"], 0.0)
        self.assertEqual(d2["analytic"]["placed_coverage"]["P5"], 0.0)


class TestRevisitNoise(unittest.TestCase):
    def setUp(self):
        self.ctx = _ctx()
        self.rc = self.ctx.config["revisit"]

    def test_ranges_in_the_config(self):
        self.assertEqual((self.rc["translation_m"], self.rc["rotation_deg"]), (0.010, 0.5))

    def test_deterministic_per_seed_and_stop(self):
        a, _ = A.revisit_cameras(self.ctx, self.ctx.rig, 1002)
        b, _ = A.revisit_cameras(A.load_context(), A.load_context().rig, 1002)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        c, _ = A.revisit_cameras(self.ctx, self.ctx.rig, 1003)
        self.assertNotEqual(a["stops"]["P3"]["delta"], c["stops"]["P3"]["delta"])
        self.assertNotEqual(a["stops"]["P2"]["delta"], a["stops"]["P3"]["delta"])             # one draw per stop, not one for all
        self.assertEqual(A.revisit_draw(self.rc, [7], 2, 0), A.revisit_draw(self.rc, [7], 2, 0))
        self.assertNotEqual(A.revisit_draw(self.rc, [7], 2, 0), A.revisit_draw(self.rc, [7], 2, 1))
        self.assertNotEqual(A.revisit_draw(self.rc, [7], 2, 0), A.revisit_draw(self.rc, [7], 3, 0))
        self.assertEqual(A.revisit_entropy("design-2"), [2 ** 31 + 2])
        self.assertNotEqual(A.revisit_entropy("design-0"), A.revisit_entropy(0))
        with self.assertRaises(A.SeedError):
            A.revisit_entropy("design-x")

    def test_draws_are_bounded_uniform(self):
        n = 3000
        d = [A.revisit_draw(self.rc, [s], s % 5, s % 7) for s in range(n)]
        for k, bound in (("dx_m", 0.010), ("dy_m", 0.010), ("dz_m", 0.010), ("dyaw_deg", 0.5), ("dpitch_deg", 0.5), ("droll_deg", 0.5)):
            v = np.array([x[k] for x in d])
            self.assertLessEqual(float(np.abs(v).max()), bound, k)
            self.assertGreater(float(np.abs(v).max()), 0.97 * bound, k)                     # the range is used
            self.assertAlmostEqual(float(v.mean()) / bound, 0.0, delta=0.04)
            self.assertAlmostEqual(float(v.std()) / bound, 1 / math.sqrt(3), delta=0.03)    # uniform on [-b, b]
        self.assertEqual(set(d[0]), {"dx_m", "dy_m", "dz_m", "dyaw_deg", "dpitch_deg", "droll_deg"})

    def test_perturb_camera_axes(self):
        cam = self.ctx.rig.cameras[2]
        zero = {"dx_m": 0, "dy_m": 0, "dz_m": 0, "dyaw_deg": 0, "dpitch_deg": 0, "droll_deg": 0}
        self.assertEqual(A._cam_pose(A.perturb_camera(cam, zero)), A._cam_pose(cam))
        p = A.perturb_camera(cam, {**zero, "dx_m": 0.01, "dz_m": -0.005})
        self.assertAlmostEqual(p.position[0] - cam.position[0], 0.01, places=8)
        self.assertAlmostEqual(p.position[2] - cam.position[2], -0.005, places=8)
        self.assertAlmostEqual(p.position[1], cam.position[1], places=8)
        self.assertEqual((p.heading_deg, p.pitch_down_deg, p.roll_deg), (cam.heading_deg, cam.pitch_down_deg, cam.roll_deg))
        q = A.perturb_camera(cam, {**zero, "dyaw_deg": 0.5, "dpitch_deg": -0.25, "droll_deg": 0.1})
        self.assertEqual((q.heading_deg, q.pitch_down_deg, q.roll_deg), (round(cam.heading_deg + 0.5, 9), round(cam.pitch_down_deg - 0.25, 9),
                                                                          round(cam.roll_deg + 0.1, 9)))
        self.assertTrue(np.allclose(q.position, cam.position, atol=1e-8))
        # yaw moves the image content sideways, pitch up/down: heading + = turned right, so a point moves LEFT in the image
        pt = np.array([[0.0, 0.2, 0.05]])
        u0 = cam.project(pt)[0]
        self.assertLess(A.perturb_camera(cam, {**zero, "dyaw_deg": 0.5}).project(pt)[0][0], u0[0])
        self.assertLess(abs((A.perturb_camera(cam, {**zero, "dpitch_deg": 0.5}).project(pt)[0] - u0)[0]), 5)

    def test_every_pose_passes_the_nominal_frame_check(self):
        for seed in (1002, 1005, 1010, 7, 8):
            rec, cams = A.revisit_cameras(self.ctx, self.ctx.rig, seed)
            for cam, (name, st) in zip(cams, rec["stops"].items()):
                with self.subTest(seed=seed, stop=name):
                    m = R.frame_margins(self.ctx.rig, cam, distort=True)
                    self.assertGreaterEqual(m["min"], self.rc["min_margin_px"])
                    self.assertAlmostEqual(m["min"], st["margin_px"]["min"], delta=0.0006)
                    self.assertEqual(st["pose"], A._cam_pose(cam))
                    self.assertEqual(st["attempt"], st["n_rejected"])
                    self.assertEqual([x["attempt"] for x in st["rejected"]], list(range(st["n_rejected"])))

    def test_failed_draws_are_redrawn_and_logged(self):
        # P1's pitch window is only 7.35-8.10 deg (nominal 7.74), so some draws must be rejected over enough seeds
        rejected = []
        for seed in range(40):
            rec, cams = A.revisit_cameras(self.ctx, self.ctx.rig, seed)
            for name, st in rec["stops"].items():
                for rj in st["rejected"]:
                    pc = A.perturb_camera(self.ctx.rig.camera(name), rj["delta"])
                    m = R.frame_margins(self.ctx.rig, pc, distort=True)["min"]
                    self.assertLess(m, self.rc["min_margin_px"])                    # the logged draw really fails the check
                    self.assertAlmostEqual(m, rj["min_margin_px"], delta=0.0006)
                    rejected.append((name, rj["attempt"]))
                accepted = A.revisit_draw(self.rc, rec["entropy"], list(rec["stops"]).index(name), st["attempt"])
                self.assertEqual(accepted, st["delta"])                              # the used draw is draw number `attempt`
        self.assertTrue(rejected)
        self.assertTrue({n for n, _ in rejected} <= {"P1", "P5"}, {n for n, _ in rejected})    # the narrow-window stops

    def test_impossible_margin_raises_after_max_redraws(self):
        cfg = json.loads((REPO / CFG_REL).read_text())
        cfg["revisit"]["min_margin_px"] = 1000.0
        cfg["revisit"]["max_redraws"] = 3
        ctx = A.load_context(cfg)
        with self.assertRaisesRegex(RuntimeError, "no revisit pose"):
            A.revisit_cameras(ctx, ctx.rig, 1)

    def test_larger_noise_means_more_redraws_but_the_same_check(self):
        cfg = json.loads((REPO / CFG_REL).read_text())
        cfg["revisit"]["translation_m"] = 0.04
        cfg["revisit"]["rotation_deg"] = 2.0
        ctx = A.load_context(cfg)
        rec, cams = A.revisit_cameras(ctx, ctx.rig, 3)
        self.assertGreater(sum(st["n_rejected"] for st in rec["stops"].values()), 0)
        for cam in cams:
            self.assertGreaterEqual(R.frame_margins(ctx.rig, cam, distort=True)["min"], 0.0)

    def test_one_pose_per_stop_shared_by_all_query_captures(self):
        a = _gen(KNOWN[0])
        cams = {c["name"]: c for c in a["scene"]["cameras"]}
        rcams = {c["name"]: c for c in a["scene"]["revisit_cameras"]}
        self.assertEqual(list(cams), list(rcams))
        self.assertEqual(list(cams), a["stops"])
        self.assertEqual(len(a["scene"]["revisit_cameras"]), len(a["stops"]))                 # exactly one revisit camera per stop
        # every query phase (visible / occluded / removed, both lightings) is rendered with that one camera set
        self.assertEqual({p["phase"]: p["camera"] for p in a["phases"]},
                         {"reference": "nominal", "visible": "revisit", "occluded": "revisit", "removed": "revisit"})
        self.assertEqual(A.PHASE_CAMERA, {p["phase"]: p["camera"] for p in a["phases"]})
        self.assertTrue(all(p["lightings"] == ["L1", "L2"] for p in a["phases"]))
        # a policy gets the same arrangement, so the same pose; a second generation gives the same record
        self.assertEqual(a["revisit"], A.generate(1002, "S8", ctx=_ctx())["revisit"])

    def test_references_are_unperturbed(self):
        a = _gen(KNOWN[0])
        nominal = R.export_scene_spec(self.ctx.rig)["cameras"]
        self.assertEqual(a["scene"]["cameras"], nominal)
        moved = 0
        for c, r in zip(a["scene"]["cameras"], a["scene"]["revisit_cameras"]):
            moved += c["position"] != r["position"] or c["forward"] != r["forward"]
            self.assertEqual((c["f_px"], c["width"], c["height"]), (r["f_px"], r["width"], r["height"]))
        self.assertEqual(moved, len(a["stops"]))
        for name, st in a["revisit"]["stops"].items():
            self.assertEqual(st["nominal"], A._cam_pose(self.ctx.rig.camera(name)))
            self.assertNotEqual(st["pose"], st["nominal"])
            for k, bound in (("dx_m", 0.01), ("dy_m", 0.01), ("dz_m", 0.01), ("dyaw_deg", 0.5), ("dpitch_deg", 0.5), ("droll_deg", 0.5)):
                self.assertLessEqual(abs(st["delta"][k]), bound)

    def test_revisit_is_part_of_the_hashed_record(self):
        a = copy.deepcopy(_gen(KNOWN[0]))
        a["revisit"]["stops"]["P3"]["delta"]["dx_m"] += 0.001
        self.assertNotEqual(A.arrangement_hash(a), _gen(KNOWN[0])["arrangement_sha256"])

    def test_design_cases_have_revisit_poses_too(self):
        d0, d1 = A.design_arrangement(0, ctx=_ctx()), A.design_arrangement(1, ctx=_ctx())
        self.assertNotEqual(d0["revisit"]["entropy"], d1["revisit"]["entropy"])
        self.assertNotEqual(d0["revisit"]["stops"]["P3"]["delta"], d1["revisit"]["stops"]["P3"]["delta"])
        self.assertEqual(len(d0["scene"]["revisit_cameras"]), 5)
        self.assertEqual(d0["revisit"], A.design_arrangement(0, ctx=_ctx())["revisit"])


class TestMatchedLayout(unittest.TestCase):
    def test_phases_mirror_the_physical_capture(self):
        spec = importlib.util.spec_from_file_location("run_arrangement", REPO / "capture" / "run_arrangement.py")
        ra = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ra)
        self.assertEqual(A.PHASES, ra.PHASES)
        a = _gen(KNOWN[0])
        ph = {p["phase"]: p for p in a["phases"]}
        self.assertEqual([p["phase"] for p in a["phases"]], [p for p, _, _ in ra.PHASES])
        self.assertEqual((ph["reference"]["target_present"], ph["reference"]["occluder_pose"]), (True, "parked"))
        self.assertEqual((ph["visible"]["target_present"], ph["visible"]["occluder_pose"]), (True, "parked"))
        self.assertEqual((ph["occluded"]["target_present"], ph["occluded"]["occluder_pose"]), (True, "placed"))
        self.assertEqual((ph["removed"]["target_present"], ph["removed"]["occluder_pose"]), (False, "placed"))
        for p in a["phases"]:
            self.assertEqual(p["lightings"], ["L1", "L2"])

    def test_removed_is_occluded_minus_the_target(self):
        for key in KNOWN[:3]:
            a = _gen(key)
            ph = {p["phase"]: p for p in a["phases"]}
            occ, rem = dict(ph["occluded"]), dict(ph["removed"])
            self.assertEqual(occ["occluder_pose"], rem["occluder_pose"])
            self.assertEqual(occ["camera"], rem["camera"])
            for k in ("phase", "condition", "target_present"):
                occ.pop(k), rem.pop(k)
            self.assertEqual(occ, rem)
            # one occluder object with one placed pose serves both phases
            self.assertEqual(set(a["objects"]["occluder"]["poses"]), {"placed", "parked"})


class TestDesignCases(unittest.TestCase):
    def test_four_option_a_cases(self):
        ctx = _ctx()
        g = ctx.geometry
        want = {"DES0": ("pick_S8", "CCHCC", "S8", None), "DES1": ("pick_M16+3", "CCHHC", "M16", "R"),
                "DES2": ("pick_M16-5", "CHHCC", "M16", "L"), "DES3": ("pick_IND28", "HHHHH", "IND28", None)}
        self.assertEqual(A.n_design_cases(), 4)
        for i in range(4):
            a = A.design_arrangement(i, ctx=ctx)
            name, pat, fam, side = want[a["arrangement_id"]]
            o = a["objects"]["occluder"]
            self.assertEqual((a["seed"], a["kind"], a["split"]), (f"design-{i}", "design", "dev"))
            self.assertEqual((o["geometry_occluder"], o["intended_pattern"]), (name, pat))
            self.assertEqual((a["occluder_family"], a["m16_side"], a["design_case"], o["family"]), (fam, side, True, fam))
            box = R.occluder_spec(g, name).box(0.0)
            self.assertTrue(np.allclose(np.asarray(o["size_m"]), 2 * np.asarray(box.half_size), atol=1e-6))
            pb = A.body_of(o, "placed")
            self.assertTrue(np.allclose(pb.centre(), box.centre, atol=1e-6))
            self.assertEqual(a["objects"]["target"]["kind"], "mesh")
            self.assertEqual(a["objects"]["distractors"], [])
            self.assertEqual({m["name"] for m in a["scene"]["meshes"]}, {"mug_body", "mug_handle"})
            self.assertEqual(a["scene"]["mug"]["axis_xy"], [0.0, 0.24])
            self.assertFalse(any(b["role"] == "occluder" for b in a["scene"]["boxes"]))
            self.assertEqual(a["predeclared_indistinguishable"], name == "pick_IND28")
            self.assertEqual(a["target_id"], "study_mug")
            self.assertIsNone(a.get("slot"))

    def test_random_arrangements_are_not_design_cases(self):
        self.assertFalse(_gen(KNOWN[0])["design_case"])


class TestPositionsFile(unittest.TestCase):
    def test_per_stop_pitch_loads_and_margins_are_positive(self):
        pos = json.loads(POS.read_text())
        self.assertNotIn("tilt_deg", pos)                    # no silent single-tilt fallback
        rig = R.load_rig(R.DEFAULT_GEOMETRY_PATH, POS)
        self.assertEqual([c.name for c in rig.cameras], ["P1", "P2", "P3", "P4", "P5"])
        for s, cam in zip(pos["stops"], rig.cameras):
            self.assertAlmostEqual(cam.pitch_down_deg, s["tilt_deg"])
            m = R.frame_margins(rig, cam, distort=True)
            self.assertGreater(m["min"], 0.0, cam.name)
            self.assertGreater(R.frame_margins(rig, cam, distort=False)["min"], 0.0, cam.name)
            self.assertAlmostEqual(m["min"], pos["frame_check"][cam.name]["k1_min_margin_px"], delta=0.06)

    def test_same_lens_positions_as_option_a(self):
        draft = R.load_rig(R.DEFAULT_GEOMETRY_PATH, DRAFT_A)
        sim = R.load_rig(R.DEFAULT_GEOMETRY_PATH, POS)
        for a, b in zip(draft.cameras, sim.cameras):
            self.assertTrue(np.allclose(a.position, b.position))
            self.assertAlmostEqual(a.heading_deg, b.heading_deg)
            self.assertAlmostEqual(a.roll_deg, b.roll_deg)
        # the draft's P1 tag 0 was outside the frame (frame check); per-stop pitch brings it in
        self.assertLess(R.frame_margins(draft, draft.cameras[0])["min"], 0.0)

    def test_verify_reads_it(self):
        from verify.schema import Positions

        p = Positions.load(POS)
        self.assertEqual((p.n_stops, p.name(p.start_index), p.f_order), (5, "P3", (0, 1, 3, 4)))
        self.assertAlmostEqual(p.travel_cm(2, 3), math.radians(22.5) * 126.8)

    def test_tilt_fallback_and_override(self):
        d = json.loads(DRAFT_A.read_text())
        g = R.load_geometry()
        self.assertEqual({c.pitch_down_deg for c in R.cameras_from_positions(d, g)}, {6.83})
        d["stops"][0]["tilt_deg"] = 7.5
        self.assertEqual([c.pitch_down_deg for c in R.cameras_from_positions(d, g)], [7.5, 6.83, 6.83, 6.83, 6.83])
        del d["tilt_deg"]
        with self.assertRaises(ValueError):
            R.cameras_from_positions(d, g)


class TestGeometryPrimitives(unittest.TestCase):
    def test_polygons(self):
        sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
        sh = sq + [1, 1]
        self.assertAlmostEqual(A.polygon_area(A.convex_hull(np.vstack([sq, [[1, 1]]]))), 4.0)
        self.assertAlmostEqual(A.overlap_area(sq, sh), 1.0)
        self.assertAlmostEqual(A.overlap_area(sq, sh[::-1]), 1.0)           # either orientation
        self.assertAlmostEqual(A.overlap_area(sq, sq + [3, 0]), 0.0)
        self.assertTrue(A.separated(sq, sq + [2.5, 0], 0.4))
        self.assertFalse(A.separated(sq, sq + [2.5, 0], 0.6))
        self.assertFalse(A.separated(sq, sh))

    def test_body_and_coverage(self):
        b = A.Body((0.1, 0.06, 0.08), (0.0, 0.2, 0.0), 90.0)
        fp = b.footprint()
        self.assertTrue(np.allclose(np.ptp(fp, axis=0), [0.06, 0.1]))
        self.assertEqual(b.corners().shape, (8, 3))
        cam = _ctx().start_cam
        self.assertAlmostEqual(A.coverage(cam, b, b), 1.0)
        front = A.Body((0.3, 0.005, 0.15), (0.0, 0.1, 0.0))
        self.assertAlmostEqual(A.coverage(cam, b, front), 1.0)
        self.assertTrue(A.may_occlude(cam, front, b))
        self.assertFalse(A.may_occlude(cam, b, front) and A.coverage(cam, front, b) == 0.0)
        # hull points: the projected hull of the support points is the projected hull of all the vertices
        rng = np.random.default_rng(0)
        box = A.Body((0.1, 0.06, 0.08), (0.02, 0.2, 0.0), 20.0)
        w = rng.random((300, 8)) + 0.05
        v = np.vstack([box.corners(), (w / w.sum(1, keepdims=True)) @ box.corners()])        # corners + strictly inner points
        sp = A.support_points(v)
        self.assertEqual(sorted(map(tuple, np.round(sp, 9))), sorted(map(tuple, np.round(box.corners(), 9))))
        self.assertAlmostEqual(A.polygon_area(A.projected_hull(cam, sp)), A.polygon_area(A.projected_hull(cam, v)))
        self.assertTrue(np.allclose(A.project(cam, v), cam.project(v)))


if __name__ == "__main__":
    unittest.main()
