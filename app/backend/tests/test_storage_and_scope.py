from __future__ import annotations

import sqlite3
import sys
from http.cookies import SimpleCookie

import pytest
from fastapi.testclient import TestClient

from conftest import COOKIE, SECOND_EMAIL


def test_same_database_survives_new_application_instance(api):
    from knee_app.api import create_app

    first = api.client()
    api.login(first)
    created = api.create(first)
    identifier = created.json()["id"]
    restarted_app = create_app(api.settings_for(api.path), clock=api.clock)
    restarted = api.client(restarted_app)
    api.login(restarted)
    response = restarted.get(f"/api/v1/evaluations/{identifier}")
    assert api.check(response, "get", "/evaluations/{evaluationId}", 200) == created.json()
    assert response.headers["etag"] == created.headers["etag"]
    assert api.revision_count(identifier) == 1


def test_failed_patch_rolls_back_inserted_revision_and_preserves_case_and_etag(api, monkeypatch):
    client = api.client()
    api.login(client)
    created = api.create(client)
    identifier = created.json()["id"]
    route = f"/api/v1/evaluations/{identifier}"
    original = api.app.state.db.record_revision

    def fail_after_insert(connection, **kwargs):
        original(connection, **kwargs)
        raise sqlite3.OperationalError("Fictional write failure must not reach the response")

    with monkeypatch.context() as local:
        local.setattr(api.app.state.db, "record_revision", fail_after_insert)
        failed = client.patch(route, headers=api.headers(client, **{"If-Match": created.headers["etag"]}), json={"clinical": {"age": 63}})
        assert failed.status_code in {500, 503}
        api.check(failed, "patch", "/evaluations/{evaluationId}", failed.status_code)
        assert "Fictional write failure" not in failed.text
        assert "sqlite" not in failed.text.lower() and "traceback" not in failed.text.lower()
    after = client.get(route)
    assert api.check(after, "get", "/evaluations/{evaluationId}", 200) == created.json()
    assert after.headers["etag"] == created.headers["etag"]
    assert api.revision_count(identifier) == 1
    retried = client.patch(route, headers=api.headers(client, **{"If-Match": created.headers["etag"]}), json={"clinical": {"age": 63}})
    assert api.check(retried, "patch", "/evaluations/{evaluationId}", 200)["clinical"]["age"] == 63
    assert api.revision_count(identifier) == 2


def test_failed_idempotent_creation_does_not_leave_case_or_poison_retry_key(api, monkeypatch):
    client = api.client()
    api.login(client)
    body = {"clinical": {"age": 62, "sex": "F", "bmi": None, "side": "right"}}
    headers = api.headers(client, **{"Idempotency-Key": "fictional-key-after-write-failure"})
    original = api.app.state.db.record_revision

    def fail_after_insert(connection, **kwargs):
        original(connection, **kwargs)
        raise sqlite3.OperationalError("Fictional creation failure")

    with monkeypatch.context() as local:
        local.setattr(api.app.state.db, "record_revision", fail_after_insert)
        response = client.post("/api/v1/evaluations", headers=headers, json=body)
        assert response.status_code in {500, 503}
        api.check(response, "post", "/evaluations", response.status_code)
    assert api.check(client.get("/api/v1/evaluations"), "get", "/evaluations", 200)["items"] == []
    retry = client.post("/api/v1/evaluations", headers=headers, json=body)
    case = api.check(retry, "post", "/evaluations", 201)
    replay = client.post("/api/v1/evaluations", headers=headers, json=body)
    assert api.check(replay, "post", "/evaluations", 200) == case
    assert api.check(client.get("/api/v1/evaluations"), "get", "/evaluations", 200)["items"] == [case]
    assert api.revision_count(case["id"]) == 1


def test_backup_restore_keeps_users_and_cases_but_revokes_all_copied_sessions(api, tmp_path):
    from knee_app.api import create_app
    from knee_app.db import backup_database, restore_database

    owner = api.client()
    api.login(owner)
    created = api.create(owner)
    identifier = created.json()["id"]
    old_cookie = owner.cookies.get(COOKIE)
    backup = tmp_path / "backup" / "fictional-copy.sqlite3"
    backup_database(api.path, backup)
    changed = owner.patch(f"/api/v1/evaluations/{identifier}", headers=api.headers(owner, **{"If-Match": created.headers["etag"]}), json={"clinical": {"age": 70}})
    api.check(changed, "patch", "/evaluations/{evaluationId}", 200)

    restored_path = tmp_path / "restored" / "fictional.sqlite3"
    restore_database(backup, restored_path)
    restored_app = create_app(api.settings_for(restored_path), clock=api.clock)
    restored = api.client(restored_app)
    restored.cookies.set(COOKIE, old_cookie, path="/api/v1")
    api.check(restored.get("/api/v1/evaluations"), "get", "/evaluations", 401)
    api.login(restored)
    response = restored.get(f"/api/v1/evaluations/{identifier}")
    assert api.check(response, "get", "/evaluations/{evaluationId}", 200) == created.json()
    assert response.headers["etag"] == created.headers["etag"]
    second = api.client(restored_app)
    assert api.login(second, SECOND_EMAIL)["authenticated"] is True
    assert api.check(second.get("/api/v1/evaluations"), "get", "/evaluations", 200)["items"] == []


def test_backup_and_restore_refuse_existing_destination_without_replacing_it(api, tmp_path):
    from knee_app.db import backup_database, restore_database

    destination = tmp_path / "existing-copy.sqlite3"
    original = b"fictional existing copy to preserve"
    destination.write_bytes(original)
    for operation in [backup_database, restore_database]:
        with pytest.raises(FileExistsError):
            operation(api.path, destination)
        assert destination.read_bytes() == original


@pytest.mark.parametrize("kind", ["future", "unknown_nonempty", "incomplete_known"])
def test_startup_backup_and_restore_reject_incompatible_sqlite_schema(api, tmp_path, kind):
    from knee_app.api import create_app
    from knee_app.db import backup_database, restore_database

    source = tmp_path / f"{kind}.sqlite3"
    with sqlite3.connect(source) as connection:
        if kind == "future":
            connection.execute("PRAGMA user_version = 999")
        elif kind == "unknown_nonempty":
            connection.execute("CREATE TABLE unrelated (id INTEGER)")
        else:
            connection.execute("PRAGMA user_version = 1")
    before = source.read_bytes()
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        create_app(api.settings_for(source), clock=api.clock)
    # Failure must not turn an unrecognized source into a newly initialized application DB.
    with sqlite3.connect(source) as connection:
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "users" not in names and "evaluations" not in names
    assert source.exists() and before
    for index, operation in enumerate([backup_database, restore_database]):
        destination = tmp_path / f"rejected-{kind}-{index}.sqlite3"
        with pytest.raises((ValueError, sqlite3.DatabaseError)):
            operation(source, destination)
        assert not destination.exists()


def test_secure_cookie_mode_emits_secure_httponly_cookie(api, tmp_path):
    from knee_app.api import create_app

    application = create_app(api.settings_for(tmp_path / "secure.sqlite3", allowed_origins=("https://testserver",), secure_cookies=True), clock=api.clock)
    client = api.stack.enter_context(TestClient(application, base_url="https://testserver", raise_server_exceptions=False))
    response = client.get("/api/v1/session")
    api.check(response, "get", "/session", 200)
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    assert cookie[COOKIE]["secure"] and cookie[COOKIE]["httponly"]
    assert cookie[COOKIE]["samesite"].lower() == "lax"
    assert cookie[COOKIE]["path"] == "/api/v1"


@pytest.mark.parametrize("method,path,body", [
    ("get", "/evaluations", None), ("get", "/evaluations/fictional-missing", None),
    ("patch", "/evaluations/fictional-missing", {"clinical": {"age": 63}}),
    ("post", "/evaluations", {"clinical": {"age": None, "sex": None, "bmi": None, "side": None}}),
    ("get", "/engine-status", None),
    ("post", "/evaluations/fictional-missing/executions", {}),
])
def test_protected_routes_do_not_accept_anonymous_or_client_side_demo_access(api, method, path, body):
    client = api.client()
    api.bootstrap(client)
    # This header intentionally cannot substitute for a server-authenticated session.
    headers = {**api.headers(client), "X-Demo-Session": "active"}
    kwargs = {"headers": headers}
    if body is not None:
        kwargs["json"] = body
    response = client.request(method, f"/api/v1{path}", **kwargs)
    operation = "/evaluations/{evaluationId}/executions" if path.endswith("/executions") else "/evaluations/{evaluationId}" if "fictional-missing" in path else path
    api.check(response, method, operation, 401)


def test_prediction_is_always_unavailable_after_auth_csrf_and_owner_checks(api):
    owner = api.client()
    other = api.client()
    api.login(owner)
    api.login(other, SECOND_EMAIL)
    created = api.create(owner)
    case = created.json()
    status = api.check(owner.get("/api/v1/engine-status"), "get", "/engine-status", 200)
    assert status["status"] == "not_configured" and status["package"] is None
    route = f'/api/v1/evaluations/{case["id"]}/executions'
    body = {"evaluationRevisionId": case["currentRevisionId"], "preparationId": "fictional-preparation", "preparationRevisionId": "fictional-preparation-revision", "reviewId": "fictional-review"}
    headers = api.headers(owner, **{"If-Match": created.headers["etag"], "Idempotency-Key": "fictional-key-001"})
    for _ in range(2):
        response = owner.post(route, headers=headers, json=body)
        api.check(response, "post", "/evaluations/{evaluationId}/executions", 503)
    api.check(owner.post(route, headers={"Origin": "http://testserver"}, json=body), "post", "/evaluations/{evaluationId}/executions", 403)
    api.check(other.post(route, headers=api.headers(other), json=body), "post", "/evaluations/{evaluationId}/executions", 404)
    read = owner.get(f'/api/v1/evaluations/{case["id"]}')
    assert api.check(read, "get", "/evaluations/{evaluationId}", 200) == case
    assert read.headers["etag"] == created.headers["etag"]
    assert api.revision_count(case["id"]) == 1
    assert not any(name == "knee" or name.startswith("knee.") or name == "src.knee" or name.startswith("src.knee.") for name in sys.modules)


@pytest.mark.parametrize("method,path", [
    ("post", "/evaluations/fictional-case/preparations"),
    ("get", "/preparations/fictional-preparation"),
    ("get", "/preparations/fictional-preparation/views/original"),
    ("post", "/preparations/fictional-preparation/reviews"),
    ("get", "/executions/fictional-execution"),
])
def test_out_of_phase_dicom_qc_and_execution_read_routes_are_not_implemented(api, method, path):
    client = api.client()
    api.login(client)
    response = client.request(method, f"/api/v1{path}", headers=api.headers(client))
    api.check(response, method, "/not-implemented-in-phase-2", 404)
