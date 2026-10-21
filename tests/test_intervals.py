"""Tests for analysis/intervals.py against closed forms and independently known values."""
from __future__ import annotations

import math
import unittest

from analysis.intervals import (
    beta_ppf,
    betainc,
    binom_cdf,
    bounds_summary,
    clopper_pearson,
    clopper_pearson_lower,
    clopper_pearson_upper,
    clopper_pearson_upper_bisect,
    jeffreys,
    jeffreys_lower,
    jeffreys_upper,
)


class BetaIncTest(unittest.TestCase):
    def test_arcsine_closed_form(self) -> None:
        # Beta(1/2, 1/2) cdf = (2/pi) asin(sqrt(x))
        for x in (1e-6, 0.01, 0.2, 0.5, 0.77, 0.999):
            self.assertAlmostEqual(betainc(0.5, 0.5, x), 2 / math.pi * math.asin(math.sqrt(x)), places=12)

    def test_power_closed_forms(self) -> None:
        # Beta(1, b) cdf = 1 - (1-x)^b ; Beta(a, 1) cdf = x^a
        for x in (0.01, 0.3, 0.9):
            for b in (1, 3.5, 40):
                self.assertAlmostEqual(betainc(1, b, x), 1 - (1 - x) ** b, places=12)
                self.assertAlmostEqual(betainc(b, 1, x), x ** b, places=12)

    def test_binomial_identity(self) -> None:
        # P(X <= x; n, p) = 1 - I_p(x+1, n-x)
        for n, x, p in ((10, 0, 0.1), (40, 1, 0.07), (20, 16, 0.6), (80, 3, 0.094)):
            self.assertAlmostEqual(binom_cdf(x, n, p), 1 - betainc(x + 1, n - x, p), places=12)

    def test_edges_and_ppf_inverse(self) -> None:
        self.assertEqual(betainc(2, 3, 0.0), 0.0)
        self.assertEqual(betainc(2, 3, 1.0), 1.0)
        for a, b, q in ((0.5, 10.5, 0.95), (2, 39, 0.05), (16.5, 4.5, 0.5)):
            self.assertAlmostEqual(betainc(a, b, beta_ppf(q, a, b)), q, places=10)
        with self.assertRaises(ValueError):
            betainc(0, 1, 0.5)


class ClopperPearsonTest(unittest.TestCase):
    def test_reference_values(self) -> None:
        # One-sided 95%. Values checked two ways (beta quantile and binomial-cdf bisection).
        # NOTE: 1/40 is 0.1132 (not 0.1129): at p = 0.1129, P(X <= 1) = 0.0505 > 0.05.
        cases = {(0, 40): 0.0722, (1, 40): 0.1132, (2, 40): 0.1492, (0, 10): 0.2589,
                 (0, 80): 0.0368, (2, 80): 0.0766, (3, 80): 0.0941, (4, 80): 0.1108}
        for (x, n), want in cases.items():
            with self.subTest(x=x, n=n):
                self.assertAlmostEqual(clopper_pearson_upper(x, n), want, places=4)
                self.assertAlmostEqual(clopper_pearson_upper_bisect(x, n), clopper_pearson_upper(x, n), places=10)

    def test_zero_successes_closed_form(self) -> None:
        for n in (1, 10, 40, 80):
            self.assertAlmostEqual(clopper_pearson_upper(0, n), 1 - 0.05 ** (1 / n), places=12)
            self.assertAlmostEqual(clopper_pearson_lower(n, n), 0.05 ** (1 / n), places=12)

    def test_recall_lower(self) -> None:
        self.assertAlmostEqual(clopper_pearson_lower(16, 20), 0.5990, places=4)

    def test_two_sided_textbook(self) -> None:
        lo, hi = clopper_pearson(5, 20)       # standard table value (0.0866, 0.4910)
        self.assertAlmostEqual(lo, 0.0866, places=4)
        self.assertAlmostEqual(hi, 0.4910, places=4)

    def test_extremes_and_symmetry(self) -> None:
        self.assertEqual(clopper_pearson_lower(0, 10), 0.0)
        self.assertEqual(clopper_pearson_upper(10, 10), 1.0)
        for x, n in ((3, 17), (0, 9), (9, 9)):
            lo, hi = clopper_pearson(x, n)
            lo2, hi2 = clopper_pearson(n - x, n)
            self.assertAlmostEqual(lo, 1 - hi2, places=12)
            self.assertAlmostEqual(hi, 1 - lo2, places=12)

    def test_bounds_bracket_rate(self) -> None:
        for x, n in ((0, 5), (1, 40), (20, 20), (7, 13)):
            self.assertLessEqual(clopper_pearson_lower(x, n), x / n)
            self.assertGreaterEqual(clopper_pearson_upper(x, n), x / n)


class JeffreysTest(unittest.TestCase):
    def test_definition(self) -> None:
        # One-sided Jeffreys upper is the conf quantile of Beta(x+1/2, n-x+1/2).
        for x, n in ((0, 10), (1, 40), (16, 20)):
            self.assertAlmostEqual(betainc(x + 0.5, n - x + 0.5, jeffreys_upper(x, n)), 0.95, places=10)
            if x:
                self.assertAlmostEqual(betainc(x + 0.5, n - x + 0.5, jeffreys_lower(x, n)), 0.05, places=10)

    def test_conventions(self) -> None:
        self.assertEqual(jeffreys_lower(0, 10), 0.0)
        self.assertEqual(jeffreys_upper(10, 10), 1.0)
        lo, hi = jeffreys(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, jeffreys_upper(0, 10))  # two-sided uses alpha/2

    def test_narrower_than_cp(self) -> None:
        for x, n in ((0, 40), (1, 40), (3, 10), (16, 20)):
            self.assertLess(jeffreys_upper(x, n), clopper_pearson_upper(x, n))
            if x:
                self.assertGreater(jeffreys_lower(x, n), clopper_pearson_lower(x, n))


class ValidationTest(unittest.TestCase):
    def test_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            clopper_pearson_upper(1, 0)
        with self.assertRaises(ValueError):
            clopper_pearson_upper(5, 4)
        with self.assertRaises(TypeError):
            clopper_pearson_upper(0.5, 4)  # fractional (seed-averaged) counts are refused
        with self.assertRaises(TypeError):
            jeffreys_upper(True, 4)
        with self.assertRaises(ValueError):
            clopper_pearson_upper(0, 4, conf=1.0)

    def test_summary(self) -> None:
        s = bounds_summary(0, 10)
        self.assertEqual((s["x"], s["n"], s["rate"]), (0, 10, 0.0))
        self.assertAlmostEqual(s["cp_upper_1s"], 0.2589, places=4)


if __name__ == "__main__":
    unittest.main()
