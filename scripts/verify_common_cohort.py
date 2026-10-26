"""Independently reconcile a frozen private cohort against its pinned CSV/QC.

This checks the resulting data, without calling the construction or label
validation functions, fitting anything, or projecting excluded outcomes.
"""
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knee.roi_global_retention_sources import collect, sha_file, digest


def verify(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_root"])
    summary = json.loads((output / "closure_summary.json").read_text(encoding="utf-8"))
    for name, sha in summary["files_sha256"].items():
        assert sha_file(output / name) == sha, "cohort_output_hash_changed"
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_files_sha256"][str(config_path.resolve())] == sha_file(config_path)
    # The constructor checked all full payload pins twice. Here recheck metadata
    # and tabular pins, independently avoiding another mass pixel payload scan.
    for path, sha in provenance["source_files_sha256"].items():
        if Path(path).suffix.lower() != ".npy":
            assert sha_file(path) == sha, "nonpixel_source_pin_changed"
    monitor_config = json.loads(Path(config["monitor_config"]["path"]).read_text(encoding="utf-8"))
    ledger = collect(monitor_config["sources"])
    members = json.loads((output / "membership_without_outcomes.json").read_text(encoding="utf-8"))
    outcomes = json.loads((output / "outcomes.json").read_text(encoding="utf-8"))
    key = lambda row: (row["src_subject_id"], row["side"])
    by_key = {key(row): row for row in members["rows"]}
    labels = {key(row): row for row in outcomes["rows"]}
    qc = {key(row): row for row in ledger["rows"]}
    included = {pair for pair, row in qc.items() if row["state"] == "accepted"}
    assert len(by_key) == len(members["rows"]) == len(included) == 2161
    assert len(labels) == len(outcomes["rows"]) == len(included)
    assert set(by_key) == set(labels) == included
    assert {key(row) for row in members["exclusions"]} == set(qc) - included
    assert len(members["exclusions"]) == 617
    assert digest(members) == outcomes["membership_sha256"]
    assert members["development_reservations"] == ledger["roles"]["development_ids"]
    assert len(set(members["development_reservations"])) == 90
    seen, classes = set(), Counter()
    with Path(monitor_config["baseline"]["path"]).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        columns = next(reader)
        indices = {name: columns.index(name) for name in (
            "SRC_SUBJECT_ID", "SIDE", "KL_BASE", "AGEYEARS", "SEX", "BMI", "BARCODE_BASE",
            "IMAGE_FILE", "KL_48", "PROGRESION_48M", "FECHA_V00", "FECHA_V06")}
        for tokens in reader:
            pair = (tokens[indices["SRC_SUBJECT_ID"]], {"1": "RIGHT", "2": "LEFT"}[tokens[indices["SIDE"]]])
            if pair not in included:
                continue
            assert pair not in seen
            seen.add(pair)
            row = {name: tokens[index] for name, index in indices.items()}
            member, label = by_key[pair], labels[pair]
            assert member["clinical"] == {"AGEYEARS": int(row["AGEYEARS"]), "SEX": row["SEX"],
                "BMI": None if row["BMI"] == "" else float(row["BMI"])}
            assert member["image_file"] == row["IMAGE_FILE"]
            assert member["barcode_base"] == row["BARCODE_BASE"]
            assert member["KL_BASE"] == label["KL_BASE"] == int(row["KL_BASE"])
            assert label["KL_48"] == int(row["KL_48"])
            assert label["PROGRESION_48M"] == int(row["PROGRESION_48M"]) == int(int(row["KL_48"]) > int(row["KL_BASE"]))
            dates = [datetime.strptime(row[name], "%d/%m/%Y").date() for name in ("FECHA_V00", "FECHA_V06")]
            assert dates[0] < dates[1]
            assert [label[name] for name in ("FECHA_V00", "FECHA_V06")] == [date.isoformat() for date in dates]
            assert member["role"] == qc[pair]["role"]
            assert member["block"] == qc[pair]["block"]
            classes[(member["role"], label["PROGRESION_48M"])] += 1
    assert seen == included
    missing = [r for r in by_key.values() if r["clinical"]["BMI"] is None]
    assert len(missing) == 3 and len({r["src_subject_id"] for r in missing}) == 2
    assert all(math.isfinite(r["clinical"]["BMI"]) for r in by_key.values() if r["clinical"]["BMI"] is not None)
    for role in ("routine", "development"):
        assert outcomes["summary"]["role_class_counts"][role] == {
            "positive": classes[(role, 1)], "negative": classes[(role, 0)]}
    assert not summary["partitions_created"] and not summary["training_executed"] and not summary["reserved_test_opened"]
    assert not members["outcomes_used_for_membership"] and not outcomes["used_for_qc_or_membership_selection"]
    return {"version": "common_cohort_independent_verification_v1.0", "status": "passed",
        "accepted_records_independently_compared_to_csv": len(seen),
        "eligible_qc_decisions_independently_reconciled": len(qc),
        "permanent_development_reservations_checked": 90,
        "missing_bmi_preserved": True, "native_payload_rehash_repeated_by_this_verifier": False,
        "closure_summary_sha256": sha_file(output / "closure_summary.json"),
        "verifier_sha256": sha_file(__file__), "partitions_created": False, "training_executed": False}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config)
    data = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.record.exists():
        assert args.record.read_text(encoding="utf-8") == data, "existing_verification_changed"
    else:
        with args.record.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
    print(data, end="")
