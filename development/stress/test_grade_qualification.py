"""Check that aggregate scores cannot hide failed label or environment gates."""
import copy
import unittest
from development.stress.grade_qualification import grade_counts, read


class GradingTests(unittest.TestCase):
    def setUp(self):
        self.criteria = read('development/stress/label_qualification_v1.json')['qualification']
        self.rows = [{'id': f'{env}-{label}-{i}', 'environment': env, 'reference_label': label, 'label': label}
                     for env in ('shelf', 'tabletop') for label in ('PRESENT', 'REMOVED', 'UNKNOWN') for i in range(8)]

    def test_removal_failure_despite_high_total(self):
        for row in self.rows:
            if row['id'] in ('shelf-REMOVED-0', 'tabletop-REMOVED-0'):
                row['label'] = 'UNKNOWN'
        score = grade_counts(self.rows, self.criteria)
        self.assertEqual(score['correct'], 46)
        self.assertFalse(score['passed'])
        self.assertFalse(score['correct_by_label']['REMOVED']['passed'])

    def test_one_opposite_error_fails_even_if_class_counts_pass(self):
        self.rows[0]['label'] = 'REMOVED'
        score = grade_counts(self.rows, self.criteria)
        self.assertTrue(score['correct_by_label']['PRESENT']['passed'])
        self.assertFalse(score['passed'])
        self.assertEqual(len(score['opposite_definite_error_ids']), 1)

    def test_environment_gate_applies_independently(self):
        criteria = copy.deepcopy(self.criteria)
        criteria['minimum_correct_by_label']['PRESENT'] = 14
        self.rows[0]['label'] = self.rows[1]['label'] = 'DISPUTED'
        score = grade_counts(self.rows, criteria)
        self.assertTrue(score['correct_by_label']['PRESENT']['passed'])
        self.assertFalse(score['correct_by_environment_and_label']['shelf']['PRESENT']['passed'])
        self.assertFalse(score['passed'])

    def test_incomplete_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reference class counts'):
            grade_counts(self.rows[:-1], self.criteria)


if __name__ == '__main__':
    unittest.main()
