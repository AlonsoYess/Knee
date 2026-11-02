"""Checks for the controlled registry of thesis and OAI source files."""

import json
import re
import unittest
from pathlib import Path


REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "governance"
    / "source_registry.json"
)
PATHS_CONFIG = REGISTRY_PATH.parents[1] / "paths.example.json"


class SourceRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        cls.files = cls.registry["files"]
        cls.by_id = {item["artifact_id"]: item for item in cls.files}

    def test_registry_has_twelve_unique_canonical_files(self):
        self.assertEqual(len(self.files), 12)
        self.assertEqual(len(self.by_id), 12)
        for item in self.files:
            self.assertTrue(item["private_metadata_verified"])
            self.assertTrue(item["status"].startswith("available_private"))
            for forbidden_key in (
                "source_name",
                "drive_path",
                "size_bytes",
                "sha256",
                "drive_id",
                "drive_url",
            ):
                self.assertNotIn(forbidden_key, item)

    def test_academic_authority_is_explicit(self):
        self.assertEqual(
            self.by_id["SRC-THESIS-001"]["authority"],
            "primary_academic_scope",
        )
        self.assertEqual(
            self.by_id["SRC-THESIS-002"]["authority"],
            "primary_methodological_scope",
        )
        self.assertEqual(
            self.by_id["SRC-CONTEXT-002"]["authority"],
            "supporting_not_governing",
        )

    def test_runtime_configuration_preserves_private_output_boundary(self):
        paths = json.loads(PATHS_CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(paths["audit_json"].startswith("outputs/auditorias/"))
        self.assertTrue(paths["run_record_json"].startswith("outputs/auditorias/"))
        self.assertEqual(
            self.by_id["DATA-COHORT-001"]["sensitivity"],
            "restricted_oai_data",
        )

    def test_exact_duplicate_is_not_uploaded_twice(self):
        duplicates = self.registry["duplicates"]
        self.assertEqual(len(duplicates), 1)
        duplicate = duplicates[0]
        self.assertEqual(duplicate["canonical_artifact_id"], "DATA-MANIFEST-001")
        self.assertEqual(duplicate["action"], "no_second_upload_exact_duplicate")
        self.assertTrue(duplicate["verified_exact_duplicate"])
        self.assertNotIn("sha256", duplicate)

    def test_registry_contains_no_local_paths_or_drive_ids(self):
        serialized = json.dumps(self.registry)
        self.assertNotIn("C:\\\\", serialized)
        self.assertNotIn("drive.google.com", serialized)
        self.assertNotIn("docs.google.com", serialized)
        self.assertIsNone(re.search(r"[A-Fa-f0-9]{64}", serialized))
        self.assertTrue(self.registry["public_view"])
        self.assertFalse(self.registry["training_authorized"])


if __name__ == "__main__":
    unittest.main()
