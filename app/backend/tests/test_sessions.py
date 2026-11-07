from __future__ import annotations

from http.cookies import SimpleCookie

import pytest

from conftest import COOKIE, FIRST_EMAIL, ORIGIN, PASSWORD


def test_anonymous_bootstrap_cookie_and_authentication_rotation(api):
    client = api.client()
    initial = api.bootstrap(client)
    anonymous_cookie = client.cookies.get(COOKIE)
    anonymous_csrf = initial["csrfToken"]
    assert initial["authenticated"] is False
    assert initial["user"] is None and initial["expiresAt"] is None
    again = api.bootstrap(client)
    assert again["csrfToken"] == anonymous_csrf
    assert client.cookies.get(COOKIE) == anonymous_cookie

    response = client.post("/api/v1/session", headers=api.headers(client), json={"email": FIRST_EMAIL, "password": PASSWORD})
    logged_in = api.check(response, "post", "/session", 200)
    assert logged_in["authenticated"] is True
    assert logged_in["user"]["id"] == api.user_ids[FIRST_EMAIL]
    assert logged_in["user"]["role"] == "doctor"
    assert logged_in["csrfToken"] != anonymous_csrf
    assert client.cookies.get(COOKIE) != anonymous_cookie
    assert PASSWORD not in response.text and client.cookies.get(COOKIE) not in response.text
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    assert cookie[COOKIE]["httponly"]
    assert cookie[COOKIE]["samesite"].lower() == "lax"
    assert cookie[COOKIE]["path"] == "/api/v1"

    old_context = api.client()
    old_context.cookies.set(COOKIE, anonymous_cookie, path="/api/v1")
    api.check(old_context.get("/api/v1/evaluations"), "get", "/evaluations", 401)
    api.tokens[client] = anonymous_csrf
    api.check(client.post("/api/v1/evaluations", headers=api.headers(client), json={"clinical": {"age": None, "sex": None, "bmi": None, "side": None}}), "post", "/evaluations", 403)


def test_logout_revokes_server_session_not_only_browser_cookie(api):
    client = api.client()
    api.login(client)
    old_cookie = client.cookies.get(COOKIE)
    response = client.delete("/api/v1/session", headers=api.headers(client))
    api.check(response, "delete", "/session", 204)
    assert client.cookies.get(COOKIE) is None
    replay = api.client()
    replay.cookies.set(COOKIE, old_cookie, path="/api/v1")
    api.check(replay.get("/api/v1/evaluations"), "get", "/evaluations", 401)
    assert api.bootstrap(replay)["authenticated"] is False


def test_authenticated_and_anonymous_sessions_expire_with_server_clock(api):
    client = api.client()
    api.login(client)
    old_cookie = client.cookies.get(COOKIE)
    api.clock.advance(3601)
    api.check(client.get("/api/v1/evaluations"), "get", "/evaluations", 401)
    assert api.bootstrap(client)["authenticated"] is False
    assert client.cookies.get(COOKIE) != old_cookie
    anonymous_cookie = client.cookies.get(COOKIE)
    api.clock.advance(1201)
    assert api.bootstrap(client)["authenticated"] is False
    assert client.cookies.get(COOKIE) != anonymous_cookie


@pytest.mark.parametrize("headers", [
    {"Origin": ORIGIN},
    {"Origin": ORIGIN, "X-CSRF-Token": "wrong-fictional-token"},
    {"Origin": "https://untrusted.example.invalid"},
    {"Origin": "null"},
    {},
])
def test_login_rejects_missing_or_wrong_csrf_and_untrusted_or_missing_origin(api, headers):
    client = api.client()
    api.bootstrap(client)
    supplied = dict(headers)
    if supplied.get("Origin") != ORIGIN:
        supplied["X-CSRF-Token"] = api.tokens[client]
    response = client.post("/api/v1/session", headers=supplied, json={"email": FIRST_EMAIL, "password": PASSWORD})
    api.check(response, "post", "/session", 403)
    assert api.bootstrap(client)["authenticated"] is False


def test_login_failures_are_throttled_and_window_expiry_allows_retry(api):
    client = api.client()
    for _ in range(api.settings.login_limit):
        api.bootstrap(client)
        response = client.post("/api/v1/session", headers=api.headers(client), json={"email": FIRST_EMAIL, "password": "wrong-fictional-password"})
        api.check(response, "post", "/session", 401)
    api.bootstrap(client)
    response = client.post("/api/v1/session", headers=api.headers(client), json={"email": FIRST_EMAIL, "password": PASSWORD})
    api.check(response, "post", "/session", 429)
    assert int(response.headers["retry-after"]) > 0
    api.clock.advance(api.settings.login_window_seconds + 1)
    assert api.login(client)["authenticated"] is True


def test_unknown_account_and_wrong_password_use_same_public_error(api):
    client = api.client()
    bodies = []
    for email in [FIRST_EMAIL, "absent@example.invalid"]:
        api.bootstrap(client)
        response = client.post("/api/v1/session", headers=api.headers(client), json={"email": email, "password": "wrong-fictional-password"})
        bodies.append(api.check(response, "post", "/session", 401)["error"])
    assert bodies[0]["code"] == bodies[1]["code"]
    assert bodies[0]["message"] == bodies[1]["message"]


def test_database_contains_hashes_instead_of_passwords_or_session_tokens(api):
    client = api.client()
    api.login(client)
    cookie = client.cookies.get(COOKIE)
    connection = api.app.state.db.connect()
    try:
        hashes = [row[0] for row in connection.execute("SELECT password_hash FROM users").fetchall()]
        tokens = [row[0] for row in connection.execute("SELECT token_hash FROM sessions").fetchall()]
    finally:
        connection.close()
    assert hashes and all(value != PASSWORD and PASSWORD not in value for value in hashes)
    assert tokens and all(value != cookie and len(value) == 64 for value in tokens)
