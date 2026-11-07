"""Loopback-only browser-test server. It never opens the user's application database."""
from pathlib import Path
import os
import secrets
import sys
import time
import uuid

BACKEND = Path(__file__).resolve().parents[3] / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "tests"))

import uvicorn
from knee_app.api import create_app
from knee_app.config import Settings
from knee_app.db import create_user
from fake_imaging import SyntheticProcessor
from synthetic_dicoms import dicom_bytes

if __name__ == "__main__":
    port = int(os.environ.get("E2E_API_PORT", "8011"))
    frontend_port = int(os.environ.get("E2E_PORT", "5174"))
    directory = BACKEND / "runtime" / ("browser-test-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    (directory / "synthetic-only.dcm").write_bytes(dicom_bytes())
    app = create_app(Settings(
        database_path=directory / "cases.sqlite3",
        secret_key=secrets.token_bytes(32),
        allowed_origins=(f"http://127.0.0.1:{frontend_port}",),
        secure_cookies=False,
        environment="test",
        login_limit=50,
    ), processor=SyntheticProcessor())
    # Public testing credentials exist exclusively in this throwaway test database.
    for email, display_name in [("browser-a@kneeoai.invalid", "Médico A de pruebas"), ("browser-b@kneeoai.invalid", "Médico B de pruebas")]:
        create_user(app.state.db, email=email, password="BrowserTestsOnly-2026!", display_name=display_name, role="doctor", now=time.time())
    uvicorn.run(app, host="127.0.0.1", port=port, access_log=False)
