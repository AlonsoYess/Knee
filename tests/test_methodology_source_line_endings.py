"""LF/CRLF portability against existing source pins, never data normalization."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee.aligned_training import data
from knee import training_data as historical
from knee.colab_data import canonical_json
from knee.model_inputs_colab import SCIENTIFIC_FILES
from test_training_data import reference_fixture

ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class SourceLineEndingTests(unittest.TestCase):
    def test_exact_lf_crlf_and_cross_platform_pins_for_all_registered_suffixes(self):
        payloads = {".py": b"def f():\n    return 'baseline'\n", ".json": b'{"value":1}\n',
                    ".txt": b"numpy==1.26.4\ntorch==2.6.0\n"}
        with tempfile.TemporaryDirectory() as directory:
            for relative in SCIENTIFIC_FILES.values():
                path = Path(directory) / Path(relative).name
                lf = payloads[path.suffix]
                for actual, expected in ((lf, lf), (lf, lf.replace(b"\n", b"\r\n")),
                                         (lf.replace(b"\n", b"\r\n"), lf),
                                         (lf.replace(b"\n", b"\r\n"), lf.replace(b"\n", b"\r\n"))):
                    path.write_bytes(actual)
                    with self.subTest(relative=relative, actual=actual, expected=expected):
                        proof = data.pinned_text_source_proof(path, digest(expected), relative_name=relative)
                        self.assertIsNotNone(proof)
                        self.assertEqual(proof["actual_sha256"], digest(actual))
                        self.assertEqual(proof["expected_sha256"], digest(expected))
                        self.assertEqual(proof["line_endings_only_equivalence"], actual != expected)
                        self.assertTrue(data.verified_text_source_matches(path, digest(expected), relative_name=relative))

    def test_utf8_multibyte_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clinical_fold_inputs.py"
            lf = "# Cirugía y evaluación\nvalue = 'rodilla'\n".encode("utf-8")
            path.write_bytes(lf)
            self.assertTrue(data.verified_text_source_matches(path, digest(lf.replace(b"\n", b"\r\n")),
                relative_name="src/knee/clinical_fold_inputs.py"))

    def test_content_whitespace_bom_and_json_semantic_edits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            relative = "configs/model_registry.json"
            path = Path(directory) / "model_registry.json"
            original = b'{"value":1}\n'
            for changed in (b'{"value":2}\n', b'{"value": 1}\n', b'{"value":1}\n ',
                            b' {"value":1}\n', b'\xef\xbb\xbf' + original,
                            b'{"value":1}', b'{"value":1}\n\n'):
                path.write_bytes(changed)
                with self.subTest(changed=changed):
                    self.assertFalse(data.verified_text_source_matches(path, digest(original), relative_name=relative))

    def test_invalid_utf8_and_bare_cr_rejected_even_with_exact_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clinical_fold_inputs.py"
            for raw in (b"# invalid\xff\n", b"line1\rline2\n", b"line1\r\nline2\rline3\n"):
                path.write_bytes(raw)
                with self.subTest(raw=raw):
                    self.assertFalse(data.verified_text_source_matches(path, digest(raw),
                        relative_name="src/knee/clinical_fold_inputs.py"))

    def test_unpinned_mixed_lf_crlf_not_accepted_as_canonical_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clinical_fold_inputs.py"
            mixed = b"line1\r\nline2\n"
            path.write_bytes(mixed)
            self.assertFalse(data.verified_text_source_matches(path, digest(b"line1\nline2\n"),
                relative_name="src/knee/clinical_fold_inputs.py"))
            # Exact hash authentication is still valid for a previously pinned
            # mixed-EOL source with no bare CR; no normalization is involved.
            self.assertTrue(data.verified_text_source_matches(path, digest(mixed),
                relative_name="src/knee/clinical_fold_inputs.py"))

    def test_unregistered_or_private_files_never_receive_portable_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handoff.json"
            path.write_bytes(b'{"value":1}\n')
            for relative in ("handoff.json", "private/development_baseline_projection_v1.json",
                             "development_package.zip", "src/knee/unregistered.py"):
                with self.subTest(relative=relative):
                    self.assertFalse(data.verified_text_source_matches(path, digest(path.read_bytes()), relative_name=relative))
            self.assertFalse(data.verified_text_source_matches(path, digest(path.read_bytes()),
                relative_name="src/knee/clinical_fold_inputs.py"))

    def test_empty_oversized_files_and_invalid_expected_digest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clinical_fold_inputs.py"
            for raw in (b"", b"a" * (data.MAX_REFERENCE_TEXT_BYTES + 1)):
                path.write_bytes(raw)
                with self.subTest(size=len(raw)):
                    self.assertFalse(data.verified_text_source_matches(path, digest(raw),
                        relative_name="src/knee/clinical_fold_inputs.py"))
            path.write_bytes(b"pass\n")
            self.assertFalse(data.verified_text_source_matches(path, "not-a-digest",
                relative_name="src/knee/clinical_fold_inputs.py"))

    def opposite_eol_reference(self, fixture):
        reference = deepcopy(fixture["reference"])
        for name, relative in SCIENTIFIC_FILES.items():
            raw = (ROOT / relative).read_bytes()
            lf = raw.replace(b"\r\n", b"\n")
            alternate = lf if b"\r\n" in raw else lf.replace(b"\n", b"\r\n")
            reference["implementation_sha256"][name] = digest(alternate)
        reference["sources_sha256"]["policy"] = reference["implementation_sha256"]["model_inputs_preflight_v1.json"]
        reference["sources_sha256"]["registry"] = reference["implementation_sha256"]["model_registry.json"]
        fixture["reference_path"].write_bytes(canonical_json(reference))
        return reference

    def test_actual_development_loader_replays_all_five_folds_under_alternate_source_eol_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = reference_fixture(directory)
            before = {relative: (ROOT / relative).read_bytes() for relative in SCIENTIFIC_FILES.values()}
            reference = self.opposite_eol_reference(fixture)
            original_verify = data.verify_clinical_reference
            with patch.object(data, "verify_clinical_reference", wraps=original_verify) as replay:
                loaded = data.load_sealed_inputs(fixture["package"], fixture["reference_path"], ROOT, fixture["package_sha256"])
            replay.assert_called_once()
            self.assertEqual(loaded["manifest"]["rows"], fixture["rows"])
            self.assertEqual(len(loaded["tensors"]), 30)
            self.assertEqual(len(loaded["source_text_verification"]), len(SCIENTIFIC_FILES) + 2)
            self.assertTrue(any(p["line_endings_only_equivalence"] for p in loaded["source_text_verification"]))
            self.assertTrue(all((ROOT / relative).read_bytes() == raw for relative, raw in before.items()))
            self.assertEqual(reference["clinical_artifacts_sha256"], fixture["reference"]["clinical_artifacts_sha256"])

    def test_historical_loader_still_rejects_alternate_hash_and_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = reference_fixture(directory)
            historical_path = ROOT / "src/knee/training_data.py"
            before = historical_path.read_bytes()
            self.opposite_eol_reference(fixture)
            with patch.object(historical, "load_development_package") as loader:
                with self.assertRaisesRegex(ValueError, "verified_clinical_or_input_implementation_changed"):
                    historical.load_sealed_inputs(fixture["package"], fixture["reference_path"], ROOT, fixture["package_sha256"])
            loader.assert_not_called()
            self.assertEqual(historical_path.read_bytes(), before)

    def test_code_content_pin_change_rejected_before_package_open(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = reference_fixture(directory)
            reference = deepcopy(fixture["reference"])
            reference["implementation_sha256"]["clinical_fold_inputs.py"] = digest(
                (ROOT / "src/knee/clinical_fold_inputs.py").read_bytes() + b"# real source change\n")
            fixture["reference_path"].write_bytes(canonical_json(reference))
            with patch.object(data, "load_development_package") as loader:
                with self.assertRaisesRegex(ValueError, "verified_clinical_or_input_implementation_changed"):
                    data.load_sealed_inputs(fixture["package"], fixture["reference_path"], ROOT, fixture["package_sha256"])
            loader.assert_not_called()

    def test_equal_eol_source_match_does_not_replace_clinical_artifact_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = reference_fixture(directory)
            reference = self.opposite_eol_reference(fixture)
            name = next(iter(reference["clinical_artifacts_sha256"]))
            reference["clinical_artifacts_sha256"][name] = "a" * 64
            fixture["reference_path"].write_bytes(canonical_json(reference))
            with self.assertRaisesRegex(ValueError, "previously_verified_clinical_artifact_bytes_changed"):
                data.load_sealed_inputs(fixture["package"], fixture["reference_path"], ROOT, fixture["package_sha256"])

    def test_handoff_data_hash_remains_exact_even_for_eol_only_change(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = reference_fixture(directory)
            self.opposite_eol_reference(fixture)
            handoff = fixture["handoff_path"].read_bytes()
            fixture["handoff_path"].write_bytes(handoff.replace(b"\n", b"\r\n"))
            with patch.object(data, "load_development_package") as loader:
                with self.assertRaisesRegex(ValueError, "sealed_handoff_changed"):
                    data.load_sealed_inputs(fixture["package"], fixture["reference_path"], ROOT, fixture["package_sha256"])
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
