from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from conftest import FIRST_EMAIL, SECOND_EMAIL


def test_partial_case_creation_assigns_identity_and_preserves_explicit_nulls(api):
    client = api.client()
    api.login(client)
    clinical = {"age": None, "sex": None, "bmi": None, "side": None}
    created = api.create(client, clinical)
    case = created.json()
    assert case["clinical"] == clinical
    assert case["id"] and case["code"] and case["currentRevisionId"]
    assert case["preparationId"] is None and case["reviewId"] is None and case["currentExecutionId"] is None
    assert created.headers["etag"].startswith('"') and created.headers["etag"].endswith('"')
    assert created.headers["location"].endswith(f'/evaluations/{case["id"]}')
    read = client.get(f'/api/v1/evaluations/{case["id"]}')
    assert api.check(read, "get", "/evaluations/{evaluationId}", 200) == case
    assert read.headers["etag"] == created.headers["etag"]
    listing = api.check(client.get("/api/v1/evaluations"), "get", "/evaluations", 200)
    assert listing["items"] == [case]
    assert listing["nextCursor"] is None


def test_case_owner_is_enforced_for_listing_read_and_patch_without_existence_leak(api):
    owner = api.client()
    other = api.client()
    api.login(owner, FIRST_EMAIL)
    api.login(other, SECOND_EMAIL)
    created = api.create(owner)
    identifier = created.json()["id"]
    assert api.check(other.get("/api/v1/evaluations"), "get", "/evaluations", 200)["items"] == []
    for method in ["get", "patch"]:
        errors = []
        for case_id in [identifier, "missing-fictional-case"]:
            kwargs = {}
            if method == "patch":
                kwargs = {"headers": api.headers(other, **{"If-Match": created.headers["etag"]}), "json": {"clinical": {"age": 99}}}
            response = other.request(method, f"/api/v1/evaluations/{case_id}", **kwargs)
            errors.append(api.check(response, method, "/evaluations/{evaluationId}", 404)["error"])
        assert errors[0]["code"] == errors[1]["code"]
        assert errors[0]["message"] == errors[1]["message"]
    final = api.check(owner.get(f"/api/v1/evaluations/{identifier}"), "get", "/evaluations/{evaluationId}", 200)
    assert final == created.json()
    assert api.revision_count(identifier) == 1


def test_patch_requires_etag_and_two_sessions_cannot_silently_overwrite_each_other(api):
    first = api.client()
    second = api.client()
    api.login(first)
    api.login(second)
    created = api.create(first)
    identifier = created.json()["id"]
    initial_tag = created.headers["etag"]
    route = f"/api/v1/evaluations/{identifier}"
    api.check(first.patch(route, headers=api.headers(first), json={"clinical": {"age": 63}}), "patch", "/evaluations/{evaluationId}", 428)
    changed = first.patch(route, headers=api.headers(first, **{"If-Match": initial_tag}), json={"clinical": {"age": 63}})
    current = api.check(changed, "patch", "/evaluations/{evaluationId}", 200)
    assert current["clinical"]["age"] == 63
    assert current["currentRevisionId"] != created.json()["currentRevisionId"]
    assert changed.headers["etag"] != initial_tag
    stale = second.patch(route, headers=api.headers(second, **{"If-Match": initial_tag}), json={"clinical": {"age": 64}})
    api.check(stale, "patch", "/evaluations/{evaluationId}", 412)
    fresh = second.get(route)
    assert api.check(fresh, "get", "/evaluations/{evaluationId}", 200) == current
    retried = second.patch(route, headers=api.headers(second, **{"If-Match": fresh.headers["etag"]}), json={"clinical": {"bmi": 27.5, "side": "left"}})
    saved = api.check(retried, "patch", "/evaluations/{evaluationId}", 200)
    assert saved["clinical"] == {"age": 63, "sex": "F", "bmi": 27.5, "side": "left"}
    assert saved["preparationId"] is None and saved["reviewId"] is None and saved["currentExecutionId"] is None
    assert api.revision_count(identifier) == 3


def test_simultaneous_same_version_writes_have_one_winner_and_keep_original_revision(api):
    first, second = api.client(), api.client()
    api.login(first)
    api.login(second)
    created = api.create(first)
    original = created.json()
    route = f'/api/v1/evaluations/{original["id"]}'
    gate = Barrier(2)

    def update(client, age):
        gate.wait(timeout=5)
        return client.patch(route, headers=api.headers(client, **{"If-Match": created.headers["etag"]}), json={"clinical": {"age": age}})

    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(update, first, 63), executor.submit(update, second, 64)]
        responses = [job.result(timeout=15) for job in jobs]
    assert sorted(response.status_code for response in responses) == [200, 412]
    winner = None
    for response in responses:
        body = api.check(response, "patch", "/evaluations/{evaluationId}", response.status_code)
        if response.status_code == 200:
            winner = body
    assert winner is not None
    assert api.check(first.get(route), "get", "/evaluations/{evaluationId}", 200) == winner
    assert api.revision_count(original["id"]) == 2
    connection = api.app.state.db.connect()
    try:
        row = connection.execute("SELECT clinical_json FROM evaluation_revisions WHERE revision_id = ?", (original["currentRevisionId"],)).fetchone()
    finally:
        connection.close()
    assert json.loads(row[0]) == original["clinical"]


def test_create_idempotency_survives_restart_without_duplicate_or_reapplying_old_inputs(api):
    from knee_app.api import create_app

    client = api.client()
    api.login(client)
    body = {"clinical": {"age": 62, "sex": "F", "bmi": None, "side": "right"}}
    key = "fictional-create-key-001"
    created = client.post("/api/v1/evaluations", headers=api.headers(client, **{"Idempotency-Key": key}), json=body)
    initial = api.check(created, "post", "/evaluations", 201)
    updated = client.patch(f'/api/v1/evaluations/{initial["id"]}', headers=api.headers(client, **{"If-Match": created.headers["etag"]}), json={"clinical": {"age": 63}})
    current = api.check(updated, "patch", "/evaluations/{evaluationId}", 200)
    restarted = api.client(create_app(api.settings_for(api.path), clock=api.clock))
    api.login(restarted)
    # Object-key order is immaterial, and a replay must not reapply the original age.
    reordered_body = {"clinical": {"side": "right", "bmi": None, "sex": "F", "age": 62}}
    replay = restarted.post("/api/v1/evaluations", headers=api.headers(restarted, **{"Idempotency-Key": key}), json=reordered_body)
    assert api.check(replay, "post", "/evaluations", 200) == current
    assert replay.headers["etag"] == updated.headers["etag"]
    assert api.check(restarted.get("/api/v1/evaluations"), "get", "/evaluations", 200)["items"] == [current]
    assert api.revision_count(initial["id"]) == 2
    conflicting = restarted.post("/api/v1/evaluations", headers=api.headers(restarted, **{"Idempotency-Key": key}), json={"clinical": {**body["clinical"], "age": 99}})
    api.check(conflicting, "post", "/evaluations", 409)
    assert api.check(restarted.get(f'/api/v1/evaluations/{initial["id"]}'), "get", "/evaluations/{evaluationId}", 200) == current
    assert api.revision_count(initial["id"]) == 2


def test_create_idempotency_keys_are_scoped_to_owner(api):
    first, second = api.client(), api.client()
    api.login(first, FIRST_EMAIL)
    api.login(second, SECOND_EMAIL)
    key = "fictional-shared-key"
    results = []
    for client, age in [(first, 62), (second, 70)]:
        body = {"clinical": {"age": age, "sex": "F", "bmi": None, "side": "right"}}
        created = client.post("/api/v1/evaluations", headers=api.headers(client, **{"Idempotency-Key": key}), json=body)
        case = api.check(created, "post", "/evaluations", 201)
        replay = client.post("/api/v1/evaluations", headers=api.headers(client, **{"Idempotency-Key": key}), json=body)
        assert api.check(replay, "post", "/evaluations", 200) == case
        assert api.check(client.get("/api/v1/evaluations"), "get", "/evaluations", 200)["items"] == [case]
        results.append(case)
    assert results[0]["id"] != results[1]["id"]


@pytest.mark.parametrize("controlled", [
    {"ownerId": "fictional-other-owner"}, {"id": "client-chosen-id"},
    {"code": "CLIENT-CHOSEN"}, {"currentRevisionId": "client-chosen-revision"},
    {"preparationId": "client-chosen-preparation"}, {"reviewId": "client-chosen-review"},
    {"currentExecutionId": "client-chosen-execution"}, {"status": "ready"},
])
def test_clients_cannot_assign_server_controlled_fields(api, controlled):
    client = api.client()
    api.login(client)
    body = {"clinical": {"age": 62, "sex": "F", "bmi": None, "side": "right"}, **controlled}
    response = client.post("/api/v1/evaluations", headers=api.headers(client), json=body)
    error = api.check(response, "post", "/evaluations", 422)
    assert error["error"]["fields"]
    assert api.check(client.get("/api/v1/evaluations"), "get", "/evaluations", 200)["items"] == []
    created = api.create(client)
    before = created.json()
    response = client.patch(f'/api/v1/evaluations/{before["id"]}', headers=api.headers(client, **{"If-Match": created.headers["etag"]}), json={"clinical": {"age": 63}, **controlled})
    api.check(response, "patch", "/evaluations/{evaluationId}", 422)
    assert api.check(client.get(f'/api/v1/evaluations/{before["id"]}'), "get", "/evaluations/{evaluationId}", 200) == before
    assert api.revision_count(before["id"]) == 1


@pytest.mark.parametrize("field,value", [
    ("age", 0), ("age", 62.5), ("age", True), ("age", "62"),
    ("bmi", 0), ("bmi", True), ("bmi", "28.4"), ("sex", "X"), ("side", "both"),
])
def test_invalid_clinical_types_are_rejected_without_a_new_case(api, field, value):
    client = api.client()
    api.login(client)
    clinical = {"age": 62, "sex": "F", "bmi": 28.4, "side": "right", field: value}
    response = client.post("/api/v1/evaluations", headers=api.headers(client), json={"clinical": clinical})
    error = api.check(response, "post", "/evaluations", 422)
    assert error["error"]["fields"]
    assert api.check(client.get("/api/v1/evaluations"), "get", "/evaluations", 200)["items"] == []


def test_patch_null_is_distinct_from_omission_and_empty_patch_is_rejected(api):
    client = api.client()
    api.login(client)
    created = api.create(client, {"age": 62, "sex": "F", "bmi": 28.4, "side": "right"})
    route = f'/api/v1/evaluations/{created.json()["id"]}'
    response = client.patch(route, headers=api.headers(client, **{"If-Match": created.headers["etag"]}), json={"clinical": {"bmi": None}})
    changed = api.check(response, "patch", "/evaluations/{evaluationId}", 200)
    assert changed["clinical"] == {"age": 62, "sex": "F", "bmi": None, "side": "right"}
    empty = client.patch(route, headers=api.headers(client, **{"If-Match": response.headers["etag"]}), json={"clinical": {}})
    api.check(empty, "patch", "/evaluations/{evaluationId}", 422)
    assert api.check(client.get(route), "get", "/evaluations/{evaluationId}", 200) == changed


@pytest.mark.parametrize("csrf,origin", [(None, "http://testserver"), ("wrong-token", "http://testserver"), ("valid", "https://other.example.invalid")])
def test_case_mutation_requires_csrf_and_allowed_origin_even_with_session_cookie(api, csrf, origin):
    client = api.client()
    api.login(client)
    headers = {"Origin": origin}
    if csrf is not None:
        headers["X-CSRF-Token"] = api.tokens[client] if csrf == "valid" else csrf
    response = client.post("/api/v1/evaluations", headers=headers, json={"clinical": {"age": None, "sex": None, "bmi": None, "side": None}})
    api.check(response, "post", "/evaluations", 403)
    assert api.check(client.get("/api/v1/evaluations"), "get", "/evaluations", 200)["items"] == []
