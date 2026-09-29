"""Validate methodological change records before any proposal is applied."""

import argparse
import json
import re
from pathlib import Path
from typing import Any


ID_PATTERN = re.compile(r"^MCR-\d{4}-\d{3}$")
ALLOWED_STATUSES = {
    "borrador",
    "en_analisis",
    "esperando_aprobacion",
    "aprobada_no_aplicada",
    "rechazada",
    "retirada",
    "implementada",
    "implementada_y_verificada",
}
IMPACT_CLASSES = {"M1", "M2", "M3", "M4"}
IMPLEMENTED_STATUSES = {"implementada", "implementada_y_verificada"}


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load a JSON change registry."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_registry(registry: dict[str, Any]) -> list[str]:
    """Return deterministic validation errors; an empty list means valid."""
    errors: list[str] = []
    required_top = {
        "schema_version",
        "registry_version",
        "status",
        "mechanism_approval",
        "researcher_approval_required",
        "training_authorized",
        "allowed_statuses",
        "impact_classes",
        "proposals",
    }
    missing_top = sorted(required_top - registry.keys())
    if missing_top:
        errors.append(f"registry missing fields: {', '.join(missing_top)}")
        return errors

    if registry["researcher_approval_required"] is not True:
        errors.append("researcher approval must remain required")
    if registry["training_authorized"] is not False:
        errors.append("the change registry cannot authorize training")
    if registry["status"] != "approved_control_mechanism":
        errors.append("the change-control mechanism is not approved")
    mechanism_approval = registry["mechanism_approval"]
    if (
        mechanism_approval.get("decision") != "approved"
        or mechanism_approval.get("approved_by") != "researcher"
        or not mechanism_approval.get("date")
    ):
        errors.append("the mechanism lacks complete researcher approval")
    if set(registry["allowed_statuses"]) != ALLOWED_STATUSES:
        errors.append("allowed statuses do not match the controlled workflow")
    if set(registry["impact_classes"]) != IMPACT_CLASSES:
        errors.append("impact classes do not match M1-M4")

    seen: set[str] = set()
    for index, proposal in enumerate(registry["proposals"]):
        prefix = proposal.get("id", f"proposal[{index}]")
        required = {
            "id",
            "title",
            "record_kind",
            "registered_at",
            "proponent",
            "impact_class",
            "status",
            "affected_rules",
            "problem",
            "proposed_change",
            "evidence",
            "alternatives",
            "impact",
            "test_block_consulted_before_decision",
            "approval",
            "implementation",
        }
        missing = sorted(required - proposal.keys())
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")
            continue
        if not ID_PATTERN.fullmatch(proposal["id"]):
            errors.append(f"{prefix} has an invalid id")
        if proposal["id"] in seen:
            errors.append(f"{prefix} is duplicated")
        seen.add(proposal["id"])
        if proposal["status"] not in ALLOWED_STATUSES:
            errors.append(f"{prefix} has an invalid status")
        if proposal["impact_class"] not in IMPACT_CLASSES:
            errors.append(f"{prefix} has an invalid impact class")
        if proposal["test_block_consulted_before_decision"] is not False:
            errors.append(f"{prefix} consulted the test block and is blocked")
        if not proposal["evidence"]:
            errors.append(f"{prefix} has no evidence")
        if not proposal["alternatives"]:
            errors.append(f"{prefix} has no alternatives")
        if not proposal["affected_rules"]:
            errors.append(f"{prefix} identifies no affected rule")

        approval = proposal["approval"]
        if proposal["status"] in {
            "aprobada_no_aplicada",
            "implementada",
            "implementada_y_verificada",
        }:
            if approval.get("decision") != "aprobada":
                errors.append(f"{prefix} lacks an approved decision")
            if approval.get("approved_by") != "researcher":
                errors.append(f"{prefix} lacks researcher approval")
            if not approval.get("date") or not approval.get("approved_scope"):
                errors.append(f"{prefix} has incomplete approval metadata")
        if proposal["status"] == "rechazada" and approval.get("decision") != "rechazada":
            errors.append(f"{prefix} rejection is inconsistent")

        implementation = proposal["implementation"]
        if proposal["status"] == "aprobada_no_aplicada" and implementation.get("applied"):
            errors.append(f"{prefix} is marked applied before implementation state")
        if proposal["status"] in IMPLEMENTED_STATUSES:
            if implementation.get("applied") is not True:
                errors.append(f"{prefix} implementation is not recorded")
            if not implementation.get("applied_at"):
                errors.append(f"{prefix} lacks implementation date")
            if not implementation.get("artifacts"):
                errors.append(f"{prefix} lacks implementation artifacts")
        if proposal["status"] == "implementada_y_verificada" and not implementation.get("verification"):
            errors.append(f"{prefix} lacks verification evidence")
    return errors


def proposal_is_authorized(registry: dict[str, Any], proposal_id: str) -> bool:
    """Allow implementation only from the explicit pre-implementation state."""
    if validate_registry(registry):
        return False
    proposal = next(
        (item for item in registry["proposals"] if item["id"] == proposal_id),
        None,
    )
    return bool(
        proposal
        and proposal["status"] == "aprobada_no_aplicada"
        and proposal["approval"]["decision"] == "aprobada"
        and proposal["approval"]["approved_by"] == "researcher"
        and proposal["test_block_consulted_before_decision"] is False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--proposal-id")
    args = parser.parse_args()
    registry = load_registry(args.registry)
    errors = validate_registry(registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    if args.proposal_id:
        authorized = proposal_is_authorized(registry, args.proposal_id)
        print("AUTHORIZED" if authorized else "BLOCKED")
        raise SystemExit(0 if authorized else 2)
    print("VALID")


if __name__ == "__main__":
    main()
