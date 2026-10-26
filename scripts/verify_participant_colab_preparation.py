"""Independent audit of participant splits and portable data, without fitting.

Only stdlib and NumPy are used. Reserved-test archive metadata and raw bytes are
checked for packaging integrity; reserved tensors are never decoded or evaluated.
"""
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import sys
import zipfile

import numpy as np

sys.dont_write_bytecode = True


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def key(row):
    return row["src_subject_id"], row["side"]


def unique_rows(rows):
    result = {key(row): row for row in rows}
    assert len(result) == len(rows), "duplicate_knee"
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pinned(spec):
    assert sha_file(spec["path"]) == spec["sha256"], "source_pin_changed"
    return read_json(spec["path"])


def rank(seed, stage, label, sid):
    return hashlib.sha256(canonical([seed, stage, label, sid])).hexdigest(), sid


def quotas(counts):
    result, offset = {}, 0
    for label in (0, 1):
        base, remainder = divmod(counts[label], 5)
        values = [base] * 5
        for extra in range(remainder):
            values[(offset + extra) % 5] += 1
        result[label] = values
        offset = (offset + counts[label]) % 5
    return result


def reconstruct(members, labels, reservations, policy):
    assert policy["seed"] == 2026 and policy["outer_blocks"] == policy["development_cv_folds"] == 5
    assert policy["reserved_test_block"] == 0 and policy["group_unit"] == "participant"
    assert policy["stratum"] == "any_accepted_knee_with_PROGRESION_48M_1_per_participant"
    assert policy["algorithm"] == "seeded_sha256_rank_with_continuous_round_robin_class_quotas"
    assert policy["retry_other_seeds_or_assignments"] is False
    assert policy["historical_processing_block_reused_as_partition"] is False
    assert policy["fit_preprocessing"] is False
    assert policy["same_assignment_for"] == ["clinical", "radiographic", "multimodal"]
    strata, roles, clinical = {}, {}, {}
    for pair, member in members.items():
        sid = pair[0]
        y = labels[pair]["PROGRESION_48M"]
        assert type(y) is int and y in {0, 1}
        strata[sid] = max(y, strata.get(sid, 0))
        role = "development" if sid in reservations else "routine"
        assert member["role"] == role
        roles[sid] = role
        assert set(member["clinical"]) == {"AGEYEARS", "SEX", "BMI"}
        assert sid not in clinical or clinical[sid] == member["clinical"]
        clinical[sid] = member["clinical"]
    groups = {label: [sid for sid in strata if strata[sid] == label] for label in (0, 1)}
    outer_quota = quotas({label: len(ids) for label, ids in groups.items()})
    assignment = {}
    for label in (0, 1):
        ranked_test = sorted((sid for sid in groups[label] if sid not in reservations),
                             key=lambda sid: rank(policy["seed"], "outer_test", label, sid))
        assert len(ranked_test) >= outer_quota[label][0]
        test = set(ranked_test[:outer_quota[label][0]])
        assignment.update({sid: (0, None) for sid in test})
        remaining = sorted(set(groups[label]) - test,
                           key=lambda sid: rank(policy["seed"], "outer_development", label, sid))
        assert len(remaining) == sum(outer_quota[label][1:])
        boundaries = np.cumsum(outer_quota[label][1:]).tolist()
        for position, sid in enumerate(remaining):
            assignment[sid] = (next(block for block, stop in enumerate(boundaries, 1)
                                    if position < stop), None)
    dev_groups = {label: [sid for sid in groups[label] if assignment[sid][0] != 0] for label in (0, 1)}
    inner_quota = quotas({label: len(ids) for label, ids in dev_groups.items()})
    for label in (0, 1):
        ranked = sorted(dev_groups[label], key=lambda sid: rank(policy["seed"], "inner_cv", label, sid))
        boundaries = np.cumsum(inner_quota[label]).tolist()
        for position, sid in enumerate(ranked):
            fold = next(fold for fold, stop in enumerate(boundaries) if position < stop)
            assignment[sid] = (assignment[sid][0], fold)
    assert set(assignment) == set(strata)
    assert not {sid for sid in reservations if sid in assignment and assignment[sid][0] == 0}
    return assignment, roles


def summarize(ids, members, labels):
    pairs = [pair for pair in members if pair[0] in ids]
    positive = [pair for pair in pairs if labels[pair]["PROGRESION_48M"] == 1]
    return {"participants": len(ids), "knees": len(pairs), "positive_knees": len(positive),
            "negative_knees": len(pairs) - len(positive),
            "participants_with_positive_knee": len({pair[0] for pair in positive})}


def safe_name(value):
    assert isinstance(value, str) and value and not value.startswith("/")
    assert not any(c in value for c in ("\\", ":", "\0"))
    assert all(part not in {"", ".", ".."} for part in value.split("/"))
    return value


def verify(config_path):
    config_path = Path(config_path)
    config_sha = sha_file(config_path)
    config = read_json(config_path)
    source = {name: pinned(spec) for name, spec in config["sources"].items()}
    output = Path(config["output_root"])
    closure_path = output / "closure_summary.json"
    closure, handoff = read_json(closure_path), read_json(output / "handoff.json")
    partition_path = output / "participant_partitions.json"
    partitions = read_json(partition_path)
    assert sha_file(partition_path) == closure["participant_partitions_sha256"] == handoff["partition_sha256"]
    assert sha_file(output / "handoff.json") == closure["handoff_sha256"]
    assert handoff == closure["packages"]
    policy = source["split_policy"]
    assert read_json(output / "adopted_split_policy.json") == policy == partitions["policy"]
    assert handoff["split_policy_sha256"] == digest(policy)
    members, labels = unique_rows(source["membership"]["rows"]), unique_rows(source["labels"]["rows"])
    assert set(members) == set(labels)
    assert source["labels"]["membership_sha256"] == digest(source["membership"])
    assert source["cohort_closure"]["files_sha256"]["membership_without_outcomes.json"] == config["sources"]["membership"]["sha256"]
    assert source["cohort_closure"]["files_sha256"]["outcomes.json"] == config["sources"]["labels"]["sha256"]
    reservations = source["membership"]["development_reservations"]
    assert len(reservations) == len(set(reservations))
    reserved = set(reservations)
    assert sorted(reserved) == partitions["development_reservations"]
    expected, roles = reconstruct(members, labels, reserved, policy)
    actual = {row["src_subject_id"]: (row["outer_block"], row["cv_fold"]) for row in partitions["participants"]}
    assert len(actual) == len(partitions["participants"]) and actual == expected
    assert all(row["preprocessing_role"] == roles[row["src_subject_id"]] for row in partitions["participants"])
    people, dev_ids = set(expected), {sid for sid, value in expected.items() if value[0] != 0}
    test_ids = people - dev_ids
    assert not dev_ids & test_ids and dev_ids | test_ids == people and not test_ids & reserved
    total = summarize(people, members, labels)
    outer = {str(block): summarize({sid for sid in people if expected[sid][0] == block}, members, labels)
             for block in range(5)}
    inner, covered_validation = {}, set()
    for fold in range(5):
        valid = {sid for sid in dev_ids if expected[sid][1] == fold}
        train = dev_ids - valid
        assert not valid & train and valid | train == dev_ids and not valid & test_ids
        assert not covered_validation & valid
        covered_validation.update(valid)
        inner[str(fold)] = {"training": summarize(train, members, labels),
                            "validation": summarize(valid, members, labels)}
    assert covered_validation == dev_ids
    summary = partitions["summary"]
    assert summary["total"] == total and summary["outer_blocks"] == outer and summary["development_cv"] == inner
    assert summary["development"] == summarize(dev_ids, members, labels)
    assert summary["permanent_development_reservations"] == len(reserved)
    assert summary["accepted_preprocessing_development_participants"] == len(people & reserved)
    assert summary["reservations_without_accepted_knees"] == len(reserved - people)
    assert summary["test_participant_fraction"] == len(test_ids) / len(people)
    assert all(row["positive_knees"] > 0 and row["negative_knees"] > 0
               for row in [outer["0"]] + [data[role] for data in inner.values() for role in ("training", "validation")])
    tensor_records = {}
    for prefix in ("development", "routine"):
        rows = source[prefix + "_manifest"]["rows"]
        assert len(rows) == len(unique_rows(rows))
        assert source[prefix + "_closure"]["manifest_sha256"] == config["sources"][prefix + "_manifest"]["sha256"]
        for row in rows:
            pair = key(row)
            assert pair not in tensor_records and row["role"] == prefix == members[pair]["role"]
            tensor_records[pair] = row["tensor"]
    assert set(tensor_records) == set(members)
    assert source["development_manifest"]["tensor_contract_sha256"] == source["routine_manifest"]["tensor_contract_sha256"]
    contract_sha = source["development_manifest"]["tensor_contract_sha256"]
    packaged_keys, package_counts, decoded_dev = set(), {}, 0
    expected_paths = {pair: f"tensors/knee_{index:06d}.npy" for index, pair in enumerate(sorted(members), 1)}
    row_fields = {"src_subject_id", "side", "preprocessing_role", "outer_block", "cv_fold", "AGEYEARS", "SEX", "BMI", "PROGRESION_48M", "tensor"}
    for kind in ("development", "reserved_test"):
        descriptor = handoff[kind]
        assert descriptor["file"] == kind + "_package.zip"
        path = output / descriptor["file"]
        assert sha_file(path) == descriptor["sha256"] and path.stat().st_size == descriptor["bytes"]
        ids = dev_ids if kind == "development" else test_ids
        expected_keys = {pair for pair in members if pair[0] in ids}
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [safe_name(entry.filename) for entry in entries]
            assert len(names) == len(set(name.casefold() for name in names))
            assert set(names) == {"manifest.json"} | {expected_paths[pair] for pair in expected_keys}
            for entry in entries:
                mode = entry.external_attr >> 16
                assert not entry.is_dir() and not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in {0, stat.S_IFREG}
                assert not entry.flag_bits & 1 and entry.compress_type == zipfile.ZIP_STORED
            manifest = json.loads(archive.read("manifest.json"))
            assert set(manifest) == {"version", "package_kind", "bindings", "rows"}
            assert manifest["version"] == "colab_data_package_v1.0" and manifest["package_kind"] == kind
            assert manifest["bindings"]["partition_sha256"] == sha_file(partition_path)
            assert manifest["bindings"]["split_policy_sha256"] == digest(policy)
            assert manifest["bindings"]["cohort_closure_sha256"] == config["sources"]["cohort_closure"]["sha256"]
            assert manifest["bindings"]["tensor_contract_sha256"] == contract_sha
            package_rows = unique_rows(manifest["rows"])
            assert set(package_rows) == expected_keys and not packaged_keys & expected_keys
            packaged_keys.update(expected_keys)
            for pair, row in package_rows.items():
                member, spec = members[pair], tensor_records[pair]
                assert set(row) == row_fields
                assert row["preprocessing_role"] == member["role"] == roles[pair[0]]
                assert (row["outer_block"], row["cv_fold"]) == expected[pair[0]]
                assert {name: row[name] for name in ("AGEYEARS", "SEX", "BMI")} == member["clinical"]
                assert row["PROGRESION_48M"] == labels[pair]["PROGRESION_48M"]
                assert row["tensor"] == dict(spec, relative_path=expected_paths[pair])
                relative = row["tensor"]["relative_path"]
                if kind == "reserved_test":
                    # Packaging integrity only: no NumPy parse of reserved bytes.
                    with archive.open(relative) as stream:
                        assert hashlib.file_digest(stream, "sha256").hexdigest() == spec["sha256"]
                else:
                    raw = archive.read(relative)
                    assert hashlib.sha256(raw).hexdigest() == spec["sha256"]
                    buffer = io.BytesIO(raw)
                    tensor = np.load(buffer, allow_pickle=False, max_header_size=4096)
                    assert buffer.tell() == len(raw) and 0 < len(raw) - tensor.nbytes <= 4096
                    assert tensor.shape == (1, 224, 224) and tensor.dtype == np.float32 and tensor.flags.c_contiguous
                    assert np.isfinite(tensor).all() and 0 <= tensor.min() <= tensor.max() <= 1
                    assert hashlib.sha256(tensor.tobytes(order="C")).hexdigest() == spec["pixel_sha256"]
                    decoded_dev += 1
        assert sha_file(path) == descriptor["sha256"]
        assert descriptor["participants"] == len(ids) and descriptor["knees"] == len(expected_keys)
        package_counts[kind] = {"participants": len(ids), "knees": len(expected_keys)}
    assert packaged_keys == set(members)
    code = handoff["code"]
    code_path = output / code["file"]
    assert sha_file(code_path) == code["sha256"]
    with zipfile.ZipFile(code_path) as archive:
        assert len(archive.infolist()) == len(code["files_sha256"])
        assert set(archive.namelist()) == set(code["files_sha256"])
        for relative, sha in code["files_sha256"].items():
            assert safe_name(relative) == relative
            assert sha == config["component_pins"][relative]
            assert hashlib.sha256(archive.read(relative)).hexdigest() == sha
    for flag in ("training_executed", "imputation_performed", "model_specific_normalization_applied",
                 "reserved_test_predictive_evaluation_opened", "reserved_test_used_for_model_development",
                 "originals_modified", "cohort_membership_changed"):
        assert closure[flag] is False
    assert closure["partition_summary"] == summary and closure["same_cohort_for_three_main_models"] is True
    assert closure["external_pretraining_disjointness_verified"] is False
    assert handoff["training_executed"] is False and handoff["reserved_test_predictive_evaluation_opened"] is False
    assert sha_file(config_path) == config_sha
    for spec in config["sources"].values():
        assert sha_file(spec["path"]) == spec["sha256"]
    for relative, sha in config["component_pins"].items():
        assert sha_file(Path(config["project_root"]) / relative) == sha
    return {"version": "participant_colab_independent_verification_v1.0", "status": "passed",
        "common_participants_checked": len(people), "common_knees_checked": len(members),
        "assignment_and_fold_reconstruction_exact": True, "packages": package_counts,
        "permanent_development_reservations_checked": len(reserved),
        "accepted_reserved_participants_in_development": len(people & reserved),
        "reservations_without_accepted_knees": len(reserved - people),
        "no_reserved_participant_in_test": True, "both_classes_in_all_required_splits": True,
        "all_knee_keys_covered_once": True, "development_tensors_decoded_and_verified": decoded_dev,
        "reserved_test_archive_metadata_checked": True, "reserved_tensor_bytes_sha256_checked": package_counts["reserved_test"]["knees"],
        "reserved_test_tensors_decoded": 0, "reserved_test_predictive_evaluation_opened": False,
        "outcome_access_scope": "fixed_partition_reconstruction_and_package_metadata_integrity_only",
        "source_files_unchanged": True, "imputation_performed": False,
        "model_specific_normalization_applied": False, "training_executed": False,
        "external_pretraining_disjointness_verified": False,
        "participant_partitions_sha256": sha_file(partition_path),
        "closure_summary_sha256": sha_file(closure_path), "verifier_sha256": sha_file(__file__)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.record.exists():
        assert args.record.read_text(encoding="utf-8") == payload, "existing_independent_verification_changed"
    else:
        with args.record.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(payload, end="")
