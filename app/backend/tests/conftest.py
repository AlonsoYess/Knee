"""Isolated phase-2 API checks. All accounts and cases below are fictional."""

from __future__ import annotations

import builtins
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker


BACKEND_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = BACKEND_ROOT.parent / "contracts" / "openapi.json"
sys.path.insert(0, str(BACKEND_ROOT))

ORIGIN = "http://testserver"
COOKIE = "kneeoai_session"
FIRST_EMAIL = "doctor.one@example.invalid"
SECOND_EMAIL = "doctor.two@example.invalid"
PASSWORD = "Fictional-Only-Password-2026!"
SECRET = b"fictional-test-key-32-characters!!"


class MutableClock:
    def __init__(self):
        self.value = time.time()

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@pytest.fixture
def scientific_import_guard(monkeypatch):
    """Fail if the web-only phase tries to load the scientific package."""
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "knee" or name.startswith("knee.") or name == "src.knee" or name.startswith("src.knee."):
            raise AssertionError("Phase-2 API must not import the scientific package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    yield


class ApiHarness:
    def __init__(self, tmp_path, contract):
        from knee_app.api import create_app
        from knee_app.db import create_user

        self.contract = contract
        self.clock = MutableClock()
        self.path = tmp_path / "fictional.sqlite3"
        self.settings = self.settings_for(self.path)
        self.app = create_app(self.settings, clock=self.clock)
        self.stack = ExitStack()
        self.tokens = {}
        self.user_ids = {}
        for email, name in [(FIRST_EMAIL, "Fictional Doctor One"), (SECOND_EMAIL, "Fictional Doctor Two")]:
            self.user_ids[email] = create_user(
                self.app.state.db, email=email, password=PASSWORD,
                display_name=name, role="doctor", now=self.clock(),
            )

    @staticmethod
    def settings_for(path, **overrides):
        from knee_app.config import Settings

        values = {
            "database_path": path, "secret_key": SECRET, "allowed_origins": (ORIGIN,),
            "secure_cookies": False, "environment": "test", "session_seconds": 3600,
            "anonymous_session_seconds": 1200, "login_limit": 5, "login_window_seconds": 900,
        }
        values.update(overrides)
        return Settings(**values)

    def client(self, app=None):
        return self.stack.enter_context(TestClient(app or self.app, base_url=ORIGIN, raise_server_exceptions=False))

    def close(self):
        self.stack.close()

    def check(self, response, method, path, status):
        assert response.status_code == status, response.text
        assert response.headers.get("x-request-id")
        assert "no-store" in response.headers.get("cache-control", "").lower()
        operation = self.contract.get("paths", {}).get(path, {}).get(method.lower(), {})
        response_spec = operation.get("responses", {}).get(str(status), operation.get("responses", {}).get("default"))
        while response_spec and "$ref" in response_spec:
            cursor = self.contract
            for part in response_spec["$ref"].removeprefix("#/").split("/"):
                cursor = cursor[part.replace("~1", "/").replace("~0", "~")]
            response_spec = cursor
        if status == 204:
            assert response.content == b""
            return None
        assert response.headers.get("content-type", "").startswith("application/json")
        payload = response.json()
        schema = (response_spec or {}).get("content", {}).get("application/json", {}).get("schema")
        if schema is None:
            assert status >= 400, "A successful endpoint must have a contract schema"
            schema = {"$ref": "#/components/schemas/ErrorResponse"}
        root_schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "components": self.contract["components"], **schema}
        Draft202012Validator(root_schema, format_checker=FormatChecker()).validate(payload)
        if status >= 400:
            assert payload["error"]["requestId"] == response.headers["x-request-id"]
        return payload

    def bootstrap(self, client):
        response = client.get("/api/v1/session")
        payload = self.check(response, "get", "/session", 200)
        self.tokens[client] = payload["csrfToken"]
        return payload

    def headers(self, client, **extra):
        return {"Origin": ORIGIN, "X-CSRF-Token": self.tokens[client], **extra}

    def login(self, client, email=FIRST_EMAIL):
        self.bootstrap(client)
        response = client.post("/api/v1/session", headers=self.headers(client), json={"email": email, "password": PASSWORD})
        payload = self.check(response, "post", "/session", 200)
        self.tokens[client] = payload["csrfToken"]
        return payload

    def create(self, client, clinical=None):
        if clinical is None:
            clinical = {"age": 62, "sex": "F", "bmi": None, "side": "right"}
        response = client.post("/api/v1/evaluations", headers=self.headers(client), json={"clinical": clinical})
        self.check(response, "post", "/evaluations", 201)
        return response

    def revision_count(self, evaluation_id):
        connection = self.app.state.db.connect()
        try:
            return connection.execute("SELECT COUNT(*) FROM evaluation_revisions WHERE evaluation_id = ?", (evaluation_id,)).fetchone()[0]
        finally:
            connection.close()


@pytest.fixture(scope="session")
def contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def api(tmp_path, contract, scientific_import_guard):
    harness = ApiHarness(tmp_path, contract)
    try:
        yield harness
    finally:
        harness.close()
