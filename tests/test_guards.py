from __future__ import annotations

import unittest

from alpha.guards import SubmissionPolicy, evaluate_submission_readiness


PASS_CHECKS = {
    "LOW_SHARPE": {"status": "PASS", "value": 2.0},
    "LOW_FITNESS": {"status": "PASS", "value": 1.1},
    "LOW_TURNOVER": {"status": "PASS", "value": 0.2},
    "HIGH_TURNOVER": {"status": "PASS", "value": 0.2},
    "CONCENTRATED_WEIGHT": {"status": "PASS"},
    "LOW_SUB_UNIVERSE_SHARPE": {"status": "PASS", "value": 1.2},
    "IS_LADDER_SHARPE": {"status": "PASS", "value": 2.5},
    "SELF_CORRELATION": {"status": "PASS", "value": 0.2},
    "PROD_CORRELATION": {"status": "PASS", "value": 0.3},
    "DATA_DIVERSITY": {"status": "PASS"},
    "REGULAR_SUBMISSION": {"status": "PASS"},
}


class SubmissionGuardTests(unittest.TestCase):
    def test_submission_guard_allows_clean_candidate(self):
        result = evaluate_submission_readiness(
            metrics={"sharpe": 2.0, "fitness": 1.1, "turnover": 0.2},
            checks=PASS_CHECKS,
            policy=SubmissionPolicy(),
            submitted_this_round=0,
        )

        self.assertTrue(result.ready)
        self.assertEqual(result.errors, [])

    def test_submission_guard_blocks_pending_mandatory_checks(self):
        checks = dict(PASS_CHECKS)
        checks["PROD_CORRELATION"] = {"status": "PENDING"}

        result = evaluate_submission_readiness(
            metrics={"sharpe": 2.0, "fitness": 1.1, "turnover": 0.2},
            checks=checks,
            policy=SubmissionPolicy(),
            submitted_this_round=0,
        )

        self.assertFalse(result.ready)
        self.assertIn("PROD_CORRELATION:PENDING", result.errors)

    def test_submission_guard_treats_regular_submission_quota_as_temporary_wait(self):
        checks = dict(PASS_CHECKS)
        checks["REGULAR_SUBMISSION"] = {"status": "FAIL", "value": 4, "limit": 4}

        result = evaluate_submission_readiness(
            metrics={"sharpe": 2.0, "fitness": 1.1, "turnover": 0.2},
            checks=checks,
            policy=SubmissionPolicy(),
            submitted_this_round=0,
        )

        self.assertFalse(result.ready)
        self.assertEqual(result.errors, ["REGULAR_SUBMISSION:QUOTA_FULL"])

    def test_submission_guard_blocks_correlation_above_limit_even_when_pass(self):
        checks = dict(PASS_CHECKS)
        checks["SELF_CORRELATION"] = {"status": "PASS", "value": 0.71}

        result = evaluate_submission_readiness(
            metrics={"sharpe": 2.0, "fitness": 1.1, "turnover": 0.2},
            checks=checks,
            policy=SubmissionPolicy(max_correlation=0.7),
            submitted_this_round=0,
        )

        self.assertFalse(result.ready)
        self.assertIn("SELF_CORRELATION:0.710>0.7", result.errors)

    def test_submission_guard_blocks_fifth_submit_in_round(self):
        result = evaluate_submission_readiness(
            metrics={"sharpe": 2.0, "fitness": 1.1, "turnover": 0.2},
            checks=PASS_CHECKS,
            policy=SubmissionPolicy(max_final_submits_per_round=4),
            submitted_this_round=4,
        )

        self.assertFalse(result.ready)
        self.assertIn("ROUND_SUBMIT_LIMIT_REACHED:4/4", result.errors)


if __name__ == "__main__":
    unittest.main()
