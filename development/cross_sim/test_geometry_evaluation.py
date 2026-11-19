import unittest

from development.cross_sim.geometry_evaluation import compare, decision, evaluate_criteria, load, PROTOCOL


def decision_fixture():
    rows = []
    for engine, n_correct in (("mujoco", 16), ("isaac", 17)):
        for i in range(25):
            answer = "REMOVED" if i < n_correct else "UNKNOWN"
            rows.append({"id": f"{engine}{i}", "engine": engine, "environment": "tabletop", "level": "D6",
                         "state": "clear_removed", "base_id": f"table{i // 5}", "stop": f"P{i % 5 + 1}",
                         "physical_state": "REMOVED", "baseline": answer, "candidate": answer})
    pairs = []
    for i in range(40):
        pair = []
        for truth in ("PRESENT", "REMOVED"):
            ident = f"hidden{i}{truth}"
            pair.append(ident)
            rows.append({"id": ident, "pair_id": str(i), "engine": "mujoco", "environment": "shelf", "level": "D4",
                         "state": "placed_" + truth.lower(), "base_id": f"shelf{i // 10}", "stop": "P1",
                         "physical_state": truth, "baseline": "UNKNOWN", "candidate": "UNKNOWN"})
        pairs.append(pair)
    return rows, pairs


class EvaluationTests(unittest.TestCase):
    def test_primary_gains_must_span_arrangements(self):
        rows, pairs = decision_fixture()
        rule = load(PROTOCOL)["decision"]
        rows[16]["candidate"] = rows[17]["candidate"] = "REMOVED"
        one_group = evaluate_criteria(rows, pairs, rule)
        self.assertFalse(one_group["criteria"]["more_clear_mujoco_removals_across_arrangements"])
        rows[20]["candidate"] = "REMOVED"
        self.assertEqual(evaluate_criteria(rows, pairs, rule)["decision"], "replicate_on_fresh_arrangements")
        rows[25]["candidate"] = "UNKNOWN"
        self.assertFalse(evaluate_criteria(rows, pairs, rule)["criteria"]["retain_clear_isaac_removals"])

    def test_certified_pair_requires_two_unknowns(self):
        rows, pairs = decision_fixture()
        rows[-1]["candidate"] = "REMOVED"
        result = evaluate_criteria(rows, pairs, load(PROTOCOL)["decision"])
        self.assertEqual(result["candidate_both_unknown"], 39)
        self.assertTrue(result["criteria"]["no_new_physical_errors"])
        self.assertFalse(result["criteria"]["all_certified_pairs_both_unknown"])

    def test_new_error_cannot_be_offset_by_fixing_another(self):
        metrics = compare([{"baseline": "REMOVED", "physical_state": "PRESENT", "candidate": "UNKNOWN"},
                           {"baseline": "UNKNOWN", "physical_state": "REMOVED", "candidate": "PRESENT"}])
        self.assertEqual(metrics["newly_wrong"], 1)
        self.assertEqual(metrics["prior_errors_avoided"], 1)

    def test_thresholds_and_refusal_take_priority(self):
        thresholds = load(PROTOCOL)["geometry"]["thresholds"]
        raw = {"diagnostics": {"align_ok": True}, "failure": False, "refusal": "accepted",
               "evidence": {"quality": 0.3, "match_score": 0.6, "obstruction": 0.4}}
        self.assertEqual(decision(raw, thresholds), ("REMOVED", "accepted"))
        raw["evidence"]["match_score"] = 0.7
        self.assertEqual(decision(raw, thresholds), ("PRESENT", "accepted"))
        raw["diagnostics"]["align_ok"] = False
        raw["refusal"] = "target_valid_fraction_below_threshold"
        self.assertEqual(decision(raw, thresholds), ("UNKNOWN", raw["refusal"]))
