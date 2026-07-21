import unittest

from alpha.models import CandidateSpec
from alpha.profile_compliance import profile_compliance_errors


class ProfileComplianceTests(unittest.TestCase):
    def test_rejects_cooldown_primary_field_without_profile_guidance(self):
        candidate = CandidateSpec(expression="rank(ts_rank(crowded_signal,63))")
        context = {
            "research_context": {
                "datafields": {
                    "field_ids": ["crowded_signal", "fresh_signal"],
                    "field_types": {"crowded_signal": "MATRIX", "fresh_signal": "MATRIX"},
                },
                "experiment_plan": {
                    "field_exposure_control": {"cooldown_fields": ["crowded_signal"]},
                },
            }
        }

        errors = profile_compliance_errors(candidate, context)

        self.assertEqual(errors, ["PROFILE_COOLDOWN_FIELD:crowded_signal"])

    def test_other_profile_accepts_oth_prefixed_fields_when_metadata_is_compacted(self):
        candidate = CandidateSpec(
            expression="rank(vec_avg(oth384_presentation_posnum))",
            metadata={"profile_guidance": {"field_family": "OTHER primary only from surfaced fields"}},
        )
        context = {
            "research_context": {
                "datafields": {
                    "field_ids": ["oth384_presentation_posnum"],
                    "field_types": {"oth384_presentation_posnum": "VECTOR"},
                }
            }
        }

        errors = profile_compliance_errors(candidate, context)

        self.assertNotIn("PROFILE_REQUIRED_FIELD_FAMILY:OTHER", errors)

    def test_analyst_profile_ignores_group_identifier_arguments(self):
        candidate = CandidateSpec(
            expression="group_zscore(anl83_numwordoperqa,industry)",
            metadata={"profile_guidance": {"field_family": "ANALYST primary only from surfaced fields"}},
        )
        context = {
            "research_context": {
                "datafields": {
                    "field_ids": ["anl83_numwordoperqa", "industry"],
                    "field_types": {"anl83_numwordoperqa": "MATRIX"},
                    "fields": [
                        {
                            "id": "anl83_numwordoperqa",
                            "type": "MATRIX",
                            "category": "Analyst",
                            "dataset_id": "analyst83",
                        },
                        {"id": "industry", "type": "GROUP", "category": "Group"},
                    ],
                }
            }
        }

        errors = profile_compliance_errors(candidate, context)

        self.assertNotIn("PROFILE_REQUIRED_FIELD_FAMILY:ANALYST", errors)


if __name__ == "__main__":
    unittest.main()
