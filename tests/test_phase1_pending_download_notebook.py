"""Contract tests for the automatic pending-batch download notebook."""

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04_descarga_lotes_pendientes.ipynb"


class Phase1PendingDownloadNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_all_code_cells_are_valid_python(self):
        ast.parse(self.code)

    def test_resume_point_comes_from_current_inventory(self):
        self.assertIn("summary['remaining_batch_numbers']", self.code)
        self.assertIn("summary['first_pending_batch']", self.code)
        self.assertIn("batch_number = int(remaining_batches[0])", self.code)
        self.assertNotIn("range(3, 21)", self.code)

    def test_each_batch_is_promoted_and_reinventoried_before_continuing(self):
        loop_at = self.code.index("while True:")
        prepare_at = self.code.index("'prepare'", loop_at)
        download_at = self.code.index("completed = subprocess.run(", prepare_at)
        promote_at = self.code.index("'promote'", download_at)
        inventory_at = self.code.index("'knee.acquisition_inventory'", promote_at)
        close_at = self.code.index("completed_batches.append(batch_number)", inventory_at)
        self.assertLess(prepare_at, download_at)
        self.assertLess(download_at, promote_at)
        self.assertLess(promote_at, inventory_at)
        self.assertLess(inventory_at, close_at)

    def test_execution_stops_on_first_error_and_is_recoverable(self):
        self.assertIn("except Exception as exc:", self.code)
        self.assertIn("caught_error = exc", self.code)
        self.assertIn("if caught_error is not None:\n    raise caught_error", self.code)
        self.assertIn("'failed_batch': failed_batch", self.code)
        self.assertIn("ejecucion_automatica_{run_id}.json", self.code)

    def test_credentials_are_requested_once_and_always_removed(self):
        self.assertEqual(self.code.count("getpass('Contraseña de la cuenta NDA: ')"), 1)
        self.assertIn("keyring.set_password", self.code)
        self.assertIn("keyring.delete_password", self.code)
        self.assertIn("shutil.rmtree(keyring_root, ignore_errors=True)", self.code)
        self.assertNotIn("NDA_PASSWORD", self.code)

    def test_failures_are_sanitized(self):
        self.assertIn("sanitize_diagnostic", self.code)
        self.assertIn("'<NDA_USERNAME>'", self.code)
        self.assertIn("'<NDA_PACKAGE_ID>'", self.code)
        self.assertIn("'s3://<RUTA_PRIVADA>'", self.code)
        self.assertIn("'<PAQUETE_PRIVADO>'", self.code)

    def test_notebook_does_not_train_or_open_reserved_test(self):
        forbidden = ("model.fit(", "torch.optim", "tensorflow", "test_loader")
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.code)
        self.assertIn("'training_executed': False", self.code)
        self.assertIn("'reserved_test_opened': False", self.code)

    def test_notebook_has_no_saved_execution_state(self):
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])


if __name__ == "__main__":
    unittest.main()
