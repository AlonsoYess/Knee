"""Contract tests for the controlled NDA selective-download notebook."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "03_descarga_selectiva_lote.ipynb"


class Phase1SelectiveDownloadNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_uses_official_pinned_client_and_exact_batch_file(self):
        self.assertIn("nda-tools==0.7.0", self.code)
        self.assertIn("'knee.selective_download'", self.code)
        self.assertIn("'prepare'", self.code)
        self.assertIn("'promote'", self.code)
        self.assertIn("'-t', str(batch_txt)", self.code)

    def test_credentials_are_not_hard_coded_or_persisted_to_drive(self):
        self.assertIn("userdata.get('NDA_USERNAME')", self.code)
        self.assertIn("userdata.get('NDA_PACKAGE_ID')", self.code)
        self.assertIn("getpass('Contraseña de la cuenta NDA: ')", self.code)
        self.assertIn("keyring.delete_password", self.code)
        self.assertIn("shutil.rmtree(keyring_root", self.code)
        self.assertNotIn("NDA_PASSWORD", self.code)

    def test_batch_is_audited_before_drive_promotion(self):
        prepare_at = self.code.index("'prepare'")
        download_at = self.code.index("downloadcmd =")
        promote_at = self.code.index("'promote'")
        inventory_at = self.code.index("'knee.acquisition_inventory'")
        self.assertLess(prepare_at, download_at)
        self.assertLess(download_at, promote_at)
        self.assertLess(promote_at, inventory_at)

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
