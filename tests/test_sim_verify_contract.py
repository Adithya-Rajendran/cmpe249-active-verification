"""Contract between sim/episodes.py (the writer) and verify/ (the readers).

Why: checkpoint 2 built the two sides in parallel. Each passed its own tests against its own fixtures, yet the view
labels schema had drifted (sim wrote v1, verify read v0). Here the files come from the real writer
(sim.episodes.build on a small rendered set) and go through the real readers: verify.manifest's loaders and
verify.select.sim_dev_data, up to a short calibration run. If either side changes a key, a schema string or a stratum
name, this fails.
"""
import json
import tempfile
import unittest
from pathlib import Path

from sim import episodes as E
from tests.test_sim_batch import POS, _write_set
from verify import select as sel
from verify.manifest import (
    SIM_VIEW_LABELS_SCHEMA,
    SIMULATION_LABEL,
    load_geometric_labels,
    load_manifest,
)
from verify.schema import EvidenceRecord, EvidenceStore, RunContext

# one arrangement per quota slot class, plus two design cases (they go to the separate design files)
ITEMS = [("SD1002", 1002, "random", "mug", False, {"family": "S8"}),
         ("SD1000", 1000, "random", "foam_brick", False, {"family": "M16", "side": "L"}),
         ("SD1007", 1007, "random", "tuna_fish_can", False, {"family": "M16", "side": "R"}),
         ("SD1001", 1001, "random", "mug", True, {"family": "IND28"}),
         ("SD1010", 1010, "random", "mug", False, {"family": "white_box"}),
         ("SD1005", 1005, "random", "mug", False, {"family": "ycb_box"}),
         ("DES0", "design-0", "design", "study_mug", False, {"family": "S8", "design": True}),
         ("DES1", "design-1", "design", "study_mug", False, {"family": "M16", "side": "R", "design": True})]
RANDOM = ["SD1000", "SD1001", "SD1002", "SD1005", "SD1007", "SD1010"]
FAST = sel.SelectionSettings(cap_steps=(2,), r_seeds=2, bootstrap_B=20, k_sensitivity=(), lam_sensitivity=(),
                             tau_q_grid=(0.5, 0.6), tau_occ_grid=(0.3,), tau_match_grid=(0.5, 0.6))


def stand_in_evidence(root: Path, jobs_file: str = "jobs_refL1.json") -> Path:
    """evidence.jsonl for every job's (episode, stop), shaped like perception/evidence.py output: a view the labels
    call diagnostic gets a clear match (present) or a clear miss (removed); a non-diagnostic one an obstruction cue."""
    labels = json.loads((root / "view_labels.json").read_text())["episodes"]
    store = EvidenceStore(5, synthetic=False)
    for job in json.loads((root / jobs_file).read_text())["jobs"]:
        eid, st = job["episode_id"], int(job["stop"])
        ep = labels[eid]
        diag = ep["stops"][st]["diagnostic"]
        match = (0.9 if ep["truth"] == "present" else 0.05) if diag else 0.1
        store.put(eid, EvidenceRecord(st, match, 0.9, 0.05 if diag else 0.9))
    out = root / "evidence.jsonl"
    store.to_jsonl(out)
    return out


class SimToVerifyContract(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "set"
        _write_set(self.root, ITEMS)
        E.build(self.root, positions=POS)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_view_labels_written_by_sim_are_read_by_verify(self) -> None:
        doc = json.loads((self.root / "view_labels.json").read_text())
        self.assertEqual(doc["schema"], E.VIEW_LABELS_SCHEMA)
        self.assertEqual(E.VIEW_LABELS_SCHEMA, SIM_VIEW_LABELS_SCHEMA)
        geo = load_geometric_labels(self.root / "view_labels.json", n_stops=5)
        for eid, ep in doc["episodes"].items():
            self.assertEqual(geo.episodes[eid], tuple(s["diagnostic"] for s in ep["stops"]), eid)
        load_geometric_labels(self.root / "view_labels_design.json", n_stops=5)

    def test_manifest_written_by_sim_is_read_by_verify(self) -> None:
        specs, truth = load_manifest(self.root / "manifest.json")
        self.assertEqual(sorted({s.arrangement_id for s in specs}), RANDOM)
        self.assertEqual(len(specs), 6 * len(RANDOM))
        self.assertEqual(len(load_manifest(self.root / "manifest_design.json")[0]), 6 * 2)

    def test_sim_dev_data_and_a_short_calibration_run(self) -> None:
        stand_in_evidence(self.root)
        dev = sel.sim_dev_data(self.root / "manifest.json", self.root / "evidence.jsonl",
                               self.root / "view_labels.json", self.root / "positions_shelf_A.json")
        self.assertEqual(list(dev.arrangements), RANDOM)
        self.assertTrue(dev.simulation)
        self.assertFalse(dev.synthetic)
        self.assertEqual(dev.label, SIMULATION_LABEL)
        # the stratum of every quota slot class, as the stratified bootstrap uses it
        self.assertEqual(dev.strata, {"SD1000": "M16-L", "SD1001": "IND28", "SD1002": "S8", "SD1005": "ycb_box",
                                      "SD1007": "M16-R", "SD1010": "white_box"})
        res = sel.run_selection(dev, FAST, RunContext(synthetic=False))
        self.assertIn("calibration", res)


if __name__ == "__main__":
    unittest.main()
