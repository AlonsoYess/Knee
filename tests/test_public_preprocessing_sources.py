"""Integrity and dependency checks for the published preparation source inventory."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicPreprocessingSourceTests(unittest.TestCase):
    def test_all_listed_sources_are_present_and_match_the_reviewed_text(self):
        inventory = json.loads((ROOT / "configs/public_preprocessing_sources_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(inventory["version"], "public_preprocessing_source_inventory_v1.0")
        self.assertFalse(inventory["training_executed"])
        self.assertFalse(inventory["reserved_test_opened"])
        paths = [row["path"] for row in inventory["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        for row in inventory["files"]:
            with self.subTest(path=row["path"]):
                path = ROOT / row["path"]
                self.assertTrue(path.is_file() and not path.is_symlink())
                self.assertTrue(path.resolve().is_relative_to(ROOT))
                raw = path.read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["canonical_lf_sha256"])

    def test_three_preparation_entrypoints_have_a_complete_local_import_closure(self):
        pending = ["common_cohort", "model_input_tensor", "participant_partitions"]
        visited = set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            path = ROOT / "src/knee" / (name + ".py")
            self.assertTrue(path.is_file(), name)
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("knee."):
                            pending.append(alias.name.split(".")[1])
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "knee":
                        pending.extend(alias.name for alias in node.names)
                    elif node.module and node.module.startswith("knee."):
                        pending.append(node.module.split(".")[1])

    def test_historical_package_check_notebooks_are_clean_sources(self):
        for name in ("31_verificar_paquete_desarrollo_sin_entrenar.ipynb",
                     "31_verificar_paquete_desarrollo_sin_entrenar_v1_1.ipynb"):
            notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8-sig"))
            for cell in notebook["cells"]:
                self.assertFalse(cell.get("outputs"))
                self.assertIsNone(cell.get("execution_count"))
                if cell["cell_type"] == "code":
                    source = cell["source"]
                    ast.parse("".join(source) if isinstance(source, list) else source)


if __name__ == "__main__":
    unittest.main()
