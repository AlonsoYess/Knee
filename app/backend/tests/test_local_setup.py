from contextlib import closing
import importlib.util
from pathlib import Path

import pytest

from knee_app.config import Settings
from knee_app.db import Database


@pytest.fixture
def setup_module(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_local_demo.py"
    spec = importlib.util.spec_from_file_location("local_setup_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = Settings(database_path=tmp_path / "application.sqlite3", environment="test", secret_key=b"x" * 32)
    monkeypatch.setattr(module.Settings, "from_env", lambda: settings)
    return module, settings


def test_creation_preserves_account_if_private_file_is_later_removed(setup_module):
    module, settings = setup_module
    module.main()
    target = settings.database_path.parent / "ACCESO_LOCAL.txt"
    assert "Contraseña:" in target.read_text(encoding="utf-8")
    db = Database(settings.database_path)
    with closing(db.connect()) as connection:
        original = tuple(connection.execute("SELECT id,password_hash FROM users").fetchone())
    target.unlink()
    module.main()
    with closing(db.connect()) as connection:
        assert tuple(connection.execute("SELECT id,password_hash FROM users").fetchone()) == original
        assert connection.execute("SELECT count(*) FROM users").fetchone()[0] == 1
    assert not target.exists()


def test_failed_private_file_write_never_creates_an_unrecoverable_account(setup_module, monkeypatch):
    module, settings = setup_module
    def fail(_descriptor):
        raise OSError("simulated full disk")
    monkeypatch.setattr(module.os, "fsync", fail)
    with pytest.raises(OSError, match="simulated full disk"):
        module.main()
    with closing(Database(settings.database_path).connect()) as connection:
        assert connection.execute("SELECT count(*) FROM users").fetchone()[0] == 0
    assert not (settings.database_path.parent / "ACCESO_LOCAL.txt").exists()


def test_preexisting_orphan_access_file_is_not_overwritten(setup_module):
    module, settings = setup_module
    target = settings.database_path.parent / "ACCESO_LOCAL.txt"
    target.write_text("previous private content", encoding="utf-8")
    with pytest.raises(ValueError, match="sin su cuenta"):
        module.main()
    assert target.read_text(encoding="utf-8") == "previous private content"
