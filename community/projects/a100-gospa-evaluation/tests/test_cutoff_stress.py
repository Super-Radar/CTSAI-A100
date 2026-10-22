import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutoff_stress import analytic_case
from study import gospa


def partial_matching_cost(a, b, cutoff):
    # Enumerate all partial matchings, including the choice of matching nothing;
    # unlike the implementation, this does not pad/clip a full cost matrix.
    best = float("inf")
    for size in range(min(len(a), len(b)) + 1):
        for rows in itertools.combinations(range(len(a)), size):
            for cols in itertools.permutations(range(len(b)), size):
                cost = sum(
                    float(np.dot(a[i] - b[j], a[i] - b[j])) for i, j in zip(rows, cols)
                )
                cost += (len(a) + len(b) - 2 * size) * cutoff**2 / 2
                best = min(best, cost)
    return best


class CutoffTests(unittest.TestCase):
    def test_closed_form_ranking_reversal(self):
        cases = analytic_case()
        self.assertEqual(cases[0]["preferred"], "one_missing")
        self.assertEqual(cases[-1]["preferred"], "all_shifted")

    def test_all_partial_matchings_across_cutoffs_and_cardinalities(self):
        rng = np.random.default_rng(20261022)
        for n in range(4):
            for m in range(4):
                a, b = rng.normal(size=(n, 2)), rng.normal(size=(m, 2))
                for cutoff in (0.1, 0.5, 2.0, 4.0):
                    result = gospa(a, b, cutoff)
                    expected = partial_matching_cost(a, b, cutoff)
                    self.assertAlmostEqual(result["distance"] ** 2, expected)
                    components = sum(
                        result[k]
                        for k in (
                            "localization_squared",
                            "missed_squared",
                            "false_squared",
                        )
                    )
                    self.assertAlmostEqual(components, expected)

    def test_cutoff_tie_is_reported_as_miss_and_false(self):
        result = gospa([[0, 0]], [[0.5, 0]], 0.5)
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["localization_squared"], 0)
        self.assertEqual(result["missed_squared"], 0.125)
        self.assertEqual(result["false_squared"], 0.125)
