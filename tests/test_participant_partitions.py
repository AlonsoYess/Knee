"""Synthetic adversarial tests for the prespecified participant partitions.

No private labels, pixels, participant lists, or model artifacts are accessed.
"""
from collections import Counter
from copy import deepcopy
import unittest

from knee import participant_partitions as subject


def synthetic_inputs(*, routine=60, development=10):
    members, labels, reservations = [], [], ["synthetic_reserved_excluded"]
    for index in range(routine + development):
        sid = f"synthetic_{index:04d}"
        role = "routine" if index < routine else "development"
        if role == "development":
            reservations.append(sid)
        clinical = {"AGEYEARS": 50 + index % 25,
                    "SEX": "F" if index % 3 else "M",
                    "BMI": None if index == 4 else 24.0 + index % 8}
        for side in ("RIGHT", "LEFT"):
            members.append({"src_subject_id": sid, "side": side, "role": role,
                            "clinical": deepcopy(clinical)})
            # Half of participants have discordant knees. Their grouping stratum
            # must therefore be derived from max(y), not whichever knee is first.
            labels.append({"src_subject_id": sid, "side": side,
                           "PROGRESION_48M": int(index % 2 == 1 and side == "RIGHT")})
    return {"rows": members, "development_reservations": reservations}, {"rows": labels}


class ParticipantPartitionTests(unittest.TestCase):
    def setUp(self):
        self.membership, self.labels = synthetic_inputs()

    def build(self, *, policy=None):
        if policy is None:
            return subject.build_partitions(self.membership, self.labels)
        return subject.build_partitions(self.membership, self.labels, policy=policy)

    def reject(self, *, policy=None):
        with self.assertRaises((ValueError, TypeError, KeyError)):
            self.build(policy=policy)

    @staticmethod
    def by_id(result):
        return {r["src_subject_id"]: r for r in result["participants"]}

    def participant_strata(self):
        result = {}
        for row in self.labels["rows"]:
            sid = row["src_subject_id"]
            result[sid] = max(result.get(sid, 0), row["PROGRESION_48M"])
        return result

    def test_returns_one_assignment_per_participant_and_all_five_outer_blocks(self):
        result = self.build()
        rows = result["participants"]
        self.assertEqual(len(rows), 70)
        self.assertEqual(len(self.by_id(result)), 70)
        self.assertEqual({r["outer_block"] for r in rows}, set(range(5)))
        self.assertEqual(result["policy"], subject.POLICY)
        for row in rows:
            self.assertIs(type(row["outer_block"]), int)
            if row["outer_block"] == 0:
                self.assertIsNone(row["cv_fold"])
            else:
                self.assertIs(type(row["cv_fold"]), int)
                self.assertIn(row["cv_fold"], range(5))

    def test_repeated_build_is_exactly_deterministic(self):
        self.assertEqual(self.build(), self.build())

    def test_input_order_and_reservation_order_do_not_change_assignment(self):
        expected = self.build()
        self.membership["rows"].reverse()
        self.membership["development_reservations"].reverse()
        self.labels["rows"] = self.labels["rows"][3:] + self.labels["rows"][:3]
        self.assertEqual(self.build(), expected)

    def test_build_does_not_mutate_inputs_or_policy(self):
        before = deepcopy((self.membership, self.labels, subject.POLICY))
        self.build()
        self.assertEqual((self.membership, self.labels, subject.POLICY), before)

    def test_both_knees_map_to_same_participant_assignment(self):
        result = self.by_id(self.build())
        pairs = Counter(row["src_subject_id"] for row in self.membership["rows"])
        self.assertTrue(all(n == 2 for n in pairs.values()))
        self.assertEqual(set(result), set(pairs))
        self.assertEqual(len(result), len(self.membership["rows"]) // 2)

    def test_all_development_participants_stay_out_of_final_test(self):
        assigned = self.by_id(self.build())
        for sid in self.membership["development_reservations"]:
            if sid in assigned:
                self.assertEqual(assigned[sid]["preprocessing_role"], "development")
                self.assertNotEqual(assigned[sid]["outer_block"], 0)

    def test_excluded_reservations_are_retained_without_creating_members(self):
        result = self.build()
        self.assertNotIn("synthetic_reserved_excluded", self.by_id(result))
        self.assertEqual(result["development_reservations"],
                         sorted(self.membership["development_reservations"]))
        self.assertEqual(result["summary"]["reservations_without_accepted_knees"], 1)

    def test_outer_block_quotas_are_balanced_by_participant_and_stratum(self):
        rows = self.build()["participants"]
        totals = Counter(r["outer_block"] for r in rows)
        self.assertEqual(totals[0], 14)
        self.assertLessEqual(max(totals.values()) - min(totals.values()), 1)
        strata = self.participant_strata()
        for y in (0, 1):
            counts = Counter(r["outer_block"] for r in rows if strata[r["src_subject_id"]] == y)
            self.assertEqual(set(counts), set(range(5)))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_inner_validation_folds_are_balanced_and_test_has_no_cv_assignment(self):
        result = self.build()
        development = [r for r in result["participants"] if r["outer_block"] != 0]
        counts = Counter(r["cv_fold"] for r in development)
        self.assertEqual(set(counts), set(range(5)))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertTrue(all(r["cv_fold"] is None for r in result["participants"] if r["outer_block"] == 0))

    def test_nondivisible_group_totals_keep_all_outer_and_cv_quota_differences_at_most_one(self):
        for routine, development in ((61, 10), (64, 9), (73, 11), (95, 12)):
            with self.subTest(routine=routine, development=development):
                self.membership, self.labels = synthetic_inputs(routine=routine, development=development)
                rows = self.build()["participants"]
                for field, eligible in (("outer_block", rows),
                                        ("cv_fold", [r for r in rows if r["outer_block"] != 0])):
                    counts = Counter(row[field] for row in eligible)
                    self.assertEqual(set(counts), set(range(5)))
                    self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_class_imbalance_is_preserved_without_sampling_or_duplicate_participants(self):
        self.membership, self.labels = synthetic_inputs(routine=80, development=10)
        for row in self.labels["rows"]:
            index = int(row["src_subject_id"].rsplit("_", 1)[1])
            row["PROGRESION_48M"] = int(index % 10 == 1 and row["side"] == "RIGHT")
        result = self.build()
        self.assertEqual(len(result["participants"]), 90)
        self.assertEqual(result["summary"]["total"]["knees"], 180)
        self.assertEqual(result["summary"]["total"]["positive_knees"], 9)
        self.assertEqual(result["summary"]["total"]["negative_knees"], 171)
        self.assertEqual(len(self.by_id(result)), 90)

    def test_each_outer_and_cv_training_validation_partition_has_both_classes(self):
        rows = self.build()["participants"]
        strata = self.participant_strata()
        for block in range(5):
            self.assertEqual({strata[r["src_subject_id"]] for r in rows if r["outer_block"] == block}, {0, 1})
        for fold in range(5):
            valid = [r for r in rows if r["outer_block"] != 0 and r["cv_fold"] == fold]
            train = [r for r in rows if r["outer_block"] != 0 and r["cv_fold"] != fold]
            self.assertEqual({strata[r["src_subject_id"]] for r in valid}, {0, 1})
            self.assertEqual({strata[r["src_subject_id"]] for r in train}, {0, 1})

    def test_discordant_knee_order_does_not_change_participant_stratum(self):
        expected = self.build()
        self.labels["rows"].reverse()
        self.assertEqual(self.build(), expected)

    def test_identical_explicit_policy_is_accepted(self):
        self.assertEqual(self.build(policy=deepcopy(subject.POLICY)), self.build())

    def test_unknown_policy_field_is_rejected(self):
        policy = deepcopy(subject.POLICY)
        policy["unapproved_retry_count"] = 10
        self.reject(policy=policy)

    def test_any_altered_policy_is_rejected(self):
        policy = deepcopy(subject.POLICY)
        key = next(iter(policy))
        policy[key] = "unapproved_replacement"
        self.reject(policy=policy)

    def test_missing_membership_label_is_rejected(self):
        self.labels["rows"].pop()
        self.reject()

    def test_extra_nonmember_label_is_rejected(self):
        self.labels["rows"].append({"src_subject_id": "synthetic_foreign", "side": "RIGHT", "PROGRESION_48M": 0})
        self.reject()

    def test_duplicate_member_side_is_rejected(self):
        self.membership["rows"].append(deepcopy(self.membership["rows"][0]))
        self.reject()

    def test_duplicate_label_side_is_rejected(self):
        self.labels["rows"].append(deepcopy(self.labels["rows"][0]))
        self.reject()

    def test_missing_or_invalid_side_is_rejected(self):
        for invalid in (None, "", "right", "BOTH", 1, True):
            with self.subTest(side=invalid):
                original = self.membership["rows"][0]["side"]
                self.membership["rows"][0]["side"] = invalid
                self.reject()
                self.membership["rows"][0]["side"] = original

    def test_invalid_identity_values_are_rejected(self):
        for invalid in (None, "", " synthetic", "synthetic ", 123, True):
            with self.subTest(identity=invalid):
                original = self.membership["rows"][0]["src_subject_id"]
                self.membership["rows"][0]["src_subject_id"] = invalid
                self.reject()
                self.membership["rows"][0]["src_subject_id"] = original

    def test_label_must_be_strict_integer_in_binary_domain(self):
        for invalid in (True, False, 0.0, 1.0, "0", "1", None, -1, 2):
            with self.subTest(label=invalid):
                original = self.labels["rows"][0]["PROGRESION_48M"]
                self.labels["rows"][0]["PROGRESION_48M"] = invalid
                self.reject()
                self.labels["rows"][0]["PROGRESION_48M"] = original

    def test_development_member_must_have_permanent_reservation(self):
        sid = self.membership["rows"][-1]["src_subject_id"]
        self.membership["development_reservations"].remove(sid)
        self.reject()

    def test_routine_member_cannot_have_development_reservation(self):
        self.membership["development_reservations"].append(self.membership["rows"][0]["src_subject_id"])
        self.reject()

    def test_roles_must_agree_for_both_knees(self):
        self.membership["rows"][0]["role"] = "development"
        self.reject()

    def test_unknown_role_is_rejected(self):
        self.membership["rows"][0]["role"] = "training"
        self.reject()

    def test_duplicate_development_reservation_is_rejected(self):
        self.membership["development_reservations"].append(self.membership["development_reservations"][0])
        self.reject()

    def test_invalid_reservation_identifier_is_rejected(self):
        self.membership["development_reservations"].append(True)
        self.reject()

    def test_clinical_fields_must_agree_across_both_knees(self):
        for key, replacement in (("AGEYEARS", 99), ("SEX", "F"), ("BMI", 48.5)):
            with self.subTest(clinical=key):
                original = self.membership["rows"][1]["clinical"][key]
                self.membership["rows"][1]["clinical"][key] = replacement
                self.reject()
                self.membership["rows"][1]["clinical"][key] = original

    def test_invalid_clinical_types_or_domains_are_rejected(self):
        for key, invalid in (("AGEYEARS", True), ("AGEYEARS", 50.0),
                             ("AGEYEARS", 0), ("SEX", "Unknown"),
                             ("BMI", True), ("BMI", 0), ("BMI", -2.0),
                             ("BMI", float("nan")), ("BMI", float("inf"))):
            with self.subTest(clinical=key, invalid=invalid):
                originals = [self.membership["rows"][i]["clinical"][key] for i in (0, 1)]
                for index in (0, 1):
                    self.membership["rows"][index]["clinical"][key] = invalid
                self.reject()
                for index, original in zip((0, 1), originals):
                    self.membership["rows"][index]["clinical"][key] = original

    def test_future_or_extra_clinical_predictor_is_rejected(self):
        for index in (0, 1):
            self.membership["rows"][index]["clinical"]["KL_48"] = 3
        self.reject()

    def test_missing_bmi_is_preserved_without_imputation(self):
        before = deepcopy(self.membership)
        self.build()
        missing = [r for r in self.membership["rows"] if r["clinical"]["BMI"] is None]
        self.assertEqual(len(missing), 2)
        self.assertEqual(self.membership, before)

    def test_single_class_dataset_fails_without_retry_or_dropping_members(self):
        for row in self.labels["rows"]:
            row["PROGRESION_48M"] = 0
        before = deepcopy((self.membership, self.labels))
        self.reject()
        self.assertEqual((self.membership, self.labels), before)

    def test_no_routine_positive_class_capacity_for_test_fails_closed(self):
        for member in self.membership["rows"]:
            if member["role"] == "routine":
                for label in self.labels["rows"]:
                    if label["src_subject_id"] == member["src_subject_id"]:
                        label["PROGRESION_48M"] = 0
        before = deepcopy((self.membership, self.labels))
        self.reject()
        self.assertEqual((self.membership, self.labels), before)

    def test_too_few_class_groups_for_five_cv_folds_fails_closed(self):
        self.membership, self.labels = synthetic_inputs(routine=10, development=0)
        self.reject()


if __name__ == "__main__":
    unittest.main()
