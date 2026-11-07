"""Exercise an isolated, application-only HTTPS deployment using invented data.

Requires only Python's standard library. It creates no accounts and never reads a
DICOM, scientific checkout, model, existing application DB, or OS certificate store.
The orchestrator must first create the explicitly disposable fixture account below.

Example (use separate report names for each invocation):
  python smoke_https.py --mode seed --base-url https://localhost:18443 \
    --ca-file /private/local-ca.crt --state-file /private/smoke-state.json \
    --cookie-file /private/smoke-auth.json --report-file /private/seed-report.json

Run ``resume`` after a service restart and ``restored`` after package restoration.
This script alone does not establish whether that service ran in a container.
State contains identifiers/ETags and a synthetic fixture digest, never credentials.
The separate auth file contains TEST session/CSRF tokens: keep it private. POSIX
files are mode 0600; the Windows orchestrator must restrict its directory's ACL.
Reports/stdout contain check outcomes only, never response bodies, IDs or tokens.
This is transport/storage testing, not scientific or clinical validation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
from http.cookiejar import Cookie, CookieJar
import json
import os
from pathlib import Path
import re
import ssl
import struct
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    build_opener, HTTPCookieProcessor, HTTPRedirectHandler, HTTPSHandler,
    ProxyHandler, Request,
)
import uuid


FIXTURE_EMAIL = "browser-a@kneeoai.invalid"
FIXTURE_PASSWORD = "BrowserTestsOnly-2026!"
CLINICAL = {"age": 62, "sex": "F", "bmi": None, "side": "right"}
COOKIE_NAME = "kneeoai_session"
STATE_FORMAT = "kneeoai-https-smoke-state-v1"
AUTH_FORMAT = "kneeoai-https-smoke-auth-v1"
MAX_JSON = 128 * 1024


class SmokeFailure(Exception):
    """Only a fixed, non-sensitive diagnostic code may leave the runner."""

    def __init__(self, code: str, status: int | None = None):
        super().__init__(code)
        self.code, self.status = code, status


def require(condition: bool, code: str) -> None:
    if not condition:
        raise SmokeFailure(code)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def loopback_origin(value: str) -> str:
    """Reject arbitrary destinations before opening any network connection."""
    try:
        require(not any(ord(c) <= 32 or ord(c) == 127 for c in value), "INVALID_BASE_URL")
        parts = urlsplit(value)
        require(parts.scheme == "https" and parts.hostname in {"localhost", "127.0.0.1"}, "HTTPS_LOOPBACK_REQUIRED")
        require(parts.username is None and parts.password is None, "URL_CREDENTIALS_FORBIDDEN")
        require(parts.path in {"", "/"} and not parts.query and not parts.fragment, "ORIGIN_ONLY_REQUIRED")
        require(parts.port is not None and 18000 <= parts.port <= 18999, "DEMO_PORT_REQUIRED")
        return f"https://{parts.hostname}:{parts.port}"
    except ValueError:
        raise SmokeFailure("INVALID_BASE_URL") from None


def strict_json(raw: bytes) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    def nonfinite(_):
        raise SmokeFailure("INVALID_JSON")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                          parse_constant=nonfinite)
    except (UnicodeError, ValueError):
        raise SmokeFailure("INVALID_JSON") from None


def read_private(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), "PRIVATE_FILE_REQUIRED")
    require(not getattr(path, "is_junction", lambda: False)(), "LINK_FORBIDDEN")
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON + 1)
    require(len(raw) <= MAX_JSON, "PRIVATE_FILE_TOO_LARGE")
    value = strict_json(raw)
    require(isinstance(value, dict), "PRIVATE_FILE_INVALID")
    return value


def write_private(path: Path, value: dict, *, overwrite: bool) -> None:
    require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(), "LINK_FORBIDDEN")
    require(not path.exists() or path.is_file(), "PRIVATE_FILE_INVALID")
    require(overwrite or not path.exists(), "DESTINATION_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".smoke-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic publish without replacing an existing seed, including races.
            os.link(temporary, path)
        if os.name == "posix":
            path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def synthetic_dicom() -> bytes:
    """A deterministic explicit-VR little-endian DICOM, wholly invented here."""
    def element(group: int, tag: int, vr: str, data: bytes) -> bytes:
        if len(data) % 2:
            data += b"\0" if vr in {"UI", "OB", "OW"} else b" "
        prefix = struct.pack("<HH", group, tag) + vr.encode("ascii")
        if vr in {"OB", "OW"}:
            return prefix + b"\0\0" + struct.pack("<I", len(data)) + data
        return prefix + struct.pack("<H", len(data)) + data
    def text(group: int, tag: int, vr: str, value: str) -> bytes:
        return element(group, tag, vr, value.encode("ascii"))
    def us(tag: int, value: int) -> bytes:
        return element(0x0028, tag, "US", struct.pack("<H", value))
    sop_class = "1.2.840.10008.5.1.4.1.1.1"
    sop_instance = "2.25.289672323995613882055799828207041511101"
    meta = b"".join((
        element(0x0002, 0x0001, "OB", b"\0\1"),
        text(0x0002, 0x0002, "UI", sop_class),
        text(0x0002, 0x0003, "UI", sop_instance),
        text(0x0002, 0x0010, "UI", "1.2.840.10008.1.2.1"),
        text(0x0002, 0x0012, "UI", "2.25.289672323995613882055799828207041511102"),
    ))
    # Mirrored geometric bands, without anatomy, sourced from no external image.
    pixels = bytearray()
    for y in range(241):
        for x in range(802):
            value = 1700 if 70 <= x % 401 <= 330 and (y < 105 or y > 135) else 100
            pixels.extend(struct.pack("<H", value + (x % 17)))
    dataset = b"".join((
        text(0x0008, 0x0016, "UI", sop_class), text(0x0008, 0x0018, "UI", sop_instance),
        text(0x0008, 0x0020, "DA", "20260101"), text(0x0008, 0x0060, "CS", "CR"),
        text(0x0008, 0x1030, "LO", "WHOLLY SYNTHETIC SOFTWARE TEST"),
        text(0x0010, 0x0010, "PN", "SYNTHETIC^ONLY"),
        text(0x0010, 0x0020, "LO", "NOT-A-PATIENT"),
        text(0x0018, 0x1164, "DS", "0.5\\0.5"), us(0x0002, 1),
        text(0x0028, 0x0004, "CS", "MONOCHROME2"), text(0x0028, 0x0008, "IS", "1"),
        us(0x0010, 241), us(0x0011, 802), us(0x0100, 16), us(0x0101, 16),
        us(0x0102, 15), us(0x0103, 0), element(0x7FE0, 0x0010, "OW", bytes(pixels)),
    ))
    return b"\0" * 128 + b"DICM" + element(0x0002, 0x0000, "UL", struct.pack("<I", len(meta))) + meta + dataset


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


@dataclass
class Response:
    status: int
    headers: Any
    data: dict

    def expected(self, status: int, error_code: str | None = None) -> "Response":
        if self.status != status:
            raise SmokeFailure("UNEXPECTED_HTTP_STATUS", self.status)
        if error_code is not None:
            error = self.data.get("error")
            require(isinstance(error, dict) and error.get("code") == error_code, "UNEXPECTED_ERROR_CODE")
        return self


class Client:
    def __init__(self, origin: str, ca_file: Path, timeout: int):
        self.origin, self.timeout, self.csrf = origin, timeout, None
        # Unlike create_default_context(), this does NOT load the OS trust store.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.verify_mode, context.check_hostname = ssl.CERT_REQUIRED, True
        context.load_verify_locations(cafile=str(ca_file))
        self.jar = CookieJar()
        handlers = lambda: [ProxyHandler({}), HTTPSHandler(context=context), NoRedirect()]
        self.opener = build_opener(*handlers(), HTTPCookieProcessor(self.jar))
        self.anonymous_opener = build_opener(*handlers())

    def request(self, method: str, path: str, *, body: dict | None = None,
                raw: bytes | None = None, headers: dict | None = None,
                anonymous: bool = False) -> Response:
        require(bool(re.fullmatch(r"/api/v1/[A-Za-z0-9_/-]+", path)), "INVALID_API_PATH")
        require(body is None or raw is None, "AMBIGUOUS_BODY")
        sent_headers = {"Accept": "application/json", "Cache-Control": "no-store"}
        if body is not None:
            raw = json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")
            sent_headers["Content-Type"] = "application/json"
        if method in {"POST", "PATCH", "PUT", "DELETE"}:
            sent_headers["Origin"] = self.origin
            if self.csrf is not None:
                sent_headers["X-CSRF-Token"] = self.csrf
        for name, value in (headers or {}).items():
            if value is None:
                sent_headers.pop(name, None)
            else:
                sent_headers[name] = value
        request = Request(self.origin + path, data=raw, headers=sent_headers, method=method)
        try:
            try:
                response = (self.anonymous_opener if anonymous else self.opener).open(request, timeout=self.timeout)
            except HTTPError as error:
                response = error
            with response:
                status, received_headers = response.status, response.headers
                received = response.read(MAX_JSON + 1)
            if 300 <= status < 400:
                raise SmokeFailure("REDIRECT_FORBIDDEN", status)
            require(len(received) <= MAX_JSON, "RESPONSE_TOO_LARGE")
            require(received_headers.get_content_type() == "application/json", "JSON_RESPONSE_REQUIRED")
            require("no-store" in received_headers.get("Cache-Control", "").lower(), "API_NO_STORE_REQUIRED")
            require(received_headers.get("X-Content-Type-Options", "").lower() == "nosniff", "NOSNIFF_REQUIRED")
            data = strict_json(received)
            require(isinstance(data, dict), "JSON_OBJECT_REQUIRED")
            return Response(status, received_headers, data)
        except (ssl.SSLError, ssl.CertificateError):
            raise SmokeFailure("TLS_VERIFICATION_FAILED") from None
        except URLError as error:
            code = "TLS_VERIFICATION_FAILED" if isinstance(error.reason, ssl.SSLError) else "NETWORK_FAILED"
            raise SmokeFailure(code) from None
        except (OSError, TimeoutError):
            raise SmokeFailure("NETWORK_FAILED") from None

    def bootstrap(self, *, authenticated: bool) -> dict:
        response = self.request("GET", "/api/v1/session").expected(200)
        require(response.data.get("authenticated") is authenticated, "SESSION_STATE_MISMATCH")
        token = response.data.get("csrfToken")
        require(isinstance(token, str) and bool(re.fullmatch(r"[a-f0-9]{64}", token)), "CSRF_TOKEN_INVALID")
        self.csrf = token
        self.session_cookie()
        return response.data

    def session_cookie(self) -> Cookie:
        cookies = list(self.jar)
        require(len(cookies) == 1 and cookies[0].name == COOKIE_NAME, "SESSION_COOKIE_REQUIRED")
        cookie = cookies[0]
        require(cookie.secure and cookie.path == "/api/v1", "SECURE_COOKIE_REQUIRED")
        require(cookie.has_nonstandard_attr("HttpOnly"), "HTTPONLY_COOKIE_REQUIRED")
        require(str(cookie.get_nonstandard_attr("SameSite", "")).lower() == "lax", "SAMESITE_COOKIE_REQUIRED")
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]{43}", cookie.value)), "SESSION_COOKIE_INVALID")
        return cookie

    def login(self) -> None:
        self.bootstrap(authenticated=False)
        old_token = self.session_cookie().value
        response = self.request("POST", "/api/v1/session", body={"email": FIXTURE_EMAIL, "password": FIXTURE_PASSWORD}).expected(200)
        require(response.data.get("authenticated") is True, "FIXTURE_LOGIN_FAILED")
        require(self.session_cookie().value != old_token, "SESSION_ROTATION_REQUIRED")
        self.bootstrap(authenticated=True)

    def save_auth(self, path: Path, *, overwrite: bool) -> None:
        cookie = self.session_cookie()
        write_private(path, {"format": AUTH_FORMAT, "origin": self.origin, "csrfToken": self.csrf,
            "cookie": {"value": cookie.value, "domain": cookie.domain,
                       "domainSpecified": cookie.domain_specified, "expires": cookie.expires}}, overwrite=overwrite)

    def load_auth(self, path: Path) -> str:
        value = read_private(path)
        require(set(value) == {"format", "origin", "csrfToken", "cookie"} and value["format"] == AUTH_FORMAT and value["origin"] == self.origin, "AUTH_FILE_INVALID")
        saved, csrf = value["cookie"], value["csrfToken"]
        require(isinstance(saved, dict) and set(saved) == {"value", "domain", "domainSpecified", "expires"}, "AUTH_FILE_INVALID")
        require(isinstance(csrf, str) and bool(re.fullmatch(r"[a-f0-9]{64}", csrf)), "AUTH_FILE_INVALID")
        require(isinstance(saved["value"], str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{43}", saved["value"])), "AUTH_FILE_INVALID")
        host = urlsplit(self.origin).hostname
        require(saved["domain"] in ({"localhost", "localhost.local"} if host == "localhost" else {host}), "AUTH_COOKIE_ORIGIN_MISMATCH")
        require(type(saved["expires"]) is int and type(saved["domainSpecified"]) is bool, "AUTH_FILE_INVALID")
        self.jar.set_cookie(Cookie(0, COOKIE_NAME, saved["value"], None, False, saved["domain"],
            saved["domainSpecified"], False, "/api/v1", True, True, saved["expires"], False,
            None, None, {"HttpOnly": None, "SameSite": "lax"}, False))
        self.csrf = csrf
        return csrf


def identifier(value: Any, prefix: str) -> str:
    require(isinstance(value, str) and bool(re.fullmatch(re.escape(prefix) + r"_[A-Za-z0-9_-]{1,120}", value)), "RESOURCE_ID_INVALID")
    return value


def etag(response: Response) -> str:
    value = response.headers.get("ETag", "")
    require(bool(re.fullmatch(r'"[A-Za-z0-9_-]+-v[1-9][0-9]*"', value)), "STRONG_ETAG_REQUIRED")
    return value


class Runner:
    def __init__(self, client: Client, arguments):
        self.client, self.args = client, arguments
        self.report = {"format": "kneeoai-https-smoke-report-v1", "mode": arguments.mode,
                       "startedAt": utc_now(), "status": "running", "checks": []}
        self.current = "initialization"

    def check(self, name, action):
        self.current = name
        result = action()
        self.report["checks"].append({"name": name, "ok": True})
        return result

    def health(self):
        response = self.client.request("GET", "/api/v1/health", anonymous=True).expected(200)
        require(response.data.get("status") == "ok" and isinstance(response.data.get("version"), str), "HEALTH_INVALID")
        require(not response.headers.get_all("Set-Cookie"), "HEALTH_MUST_NOT_SET_COOKIE")

    def case(self, state=None):
        identifier_ = state["evaluationId"]
        response = self.client.request("GET", f"/api/v1/evaluations/{identifier_}").expected(200)
        require(response.data.get("id") == identifier_ and response.data.get("clinical") == CLINICAL, "CASE_SNAPSHOT_MISMATCH")
        require(response.data.get("currentRevisionId") == state["evaluationRevisionId"] and etag(response) == state["evaluationEtag"], "CASE_VERSION_MISMATCH")
        require(response.data.get("preparationId") == state["preparationId"], "PREPARATION_LINK_MISMATCH")
        require(response.data.get("reviewId") is None and response.data.get("currentExecutionId") is None, "UNEXPECTED_CLINICAL_OUTPUT")
        return response

    def preparation(self, state):
        response = self.client.request("GET", f'/api/v1/preparations/{state["preparationId"]}').expected(200)
        data = response.data
        require(data.get("id") == state["preparationId"] and data.get("evaluationId") == state["evaluationId"], "PREPARATION_LINK_MISMATCH")
        require(data.get("preparationRevisionId") == state["preparationRevisionId"] and etag(response) == state["preparationEtag"], "PREPARATION_VERSION_MISMATCH")
        require(data.get("image") == {"imageId": state["imageId"], "imageVersion": state["imageVersion"]}, "IMAGE_VERSION_MISMATCH")
        require(data.get("sourceEvaluationRevisionId") == state["evaluationRevisionId"] and data.get("side") == "right", "PREPARATION_SOURCE_MISMATCH")
        require(data.get("status") == "failed" and isinstance(data.get("error"), dict) and data["error"].get("code") == "PROCESSOR_UNAVAILABLE", "FAILED_PREPARATION_NOT_PRESERVED")
        require(all(data.get(name) is None for name in ("sourceContext", "sourceReference", "cropGeometry", "activeReview")), "UNEXPECTED_PREPARATION_OUTPUT")
        return response

    def seed(self):
        require(not self.args.state_file.exists() and not self.args.cookie_file.exists(), "SEED_FILES_ALREADY_EXIST")
        self.check("fixture_login_and_secure_rotated_cookie", self.client.login)
        body = {"clinical": dict(CLINICAL)}
        self.check("missing_csrf_rejected", lambda: self.client.request("POST", "/api/v1/evaluations", body=body,
            headers={"X-CSRF-Token": None}).expected(403, "ACCESS_DENIED"))
        self.check("foreign_origin_rejected", lambda: self.client.request("POST", "/api/v1/evaluations", body=body,
            headers={"Origin": "https://foreign.invalid"}).expected(403, "ACCESS_DENIED"))
        created = self.check("synthetic_case_created", lambda: self.client.request("POST", "/api/v1/evaluations", body=body,
            headers={"Idempotency-Key": uuid.uuid4().hex}).expected(201))
        case_id = identifier(created.data.get("id"), "ev")
        case_path = f"/api/v1/evaluations/{case_id}"
        require(created.headers.get("Location") == case_path and created.data.get("clinical") == CLINICAL, "CREATED_CASE_INVALID")
        old_etag = etag(created)
        patched = self.check("current_etag_patch_accepted", lambda: self.client.request("PATCH", case_path,
            body={"clinical": {"age": 62}}, headers={"If-Match": old_etag}).expected(200))
        require(etag(patched) != old_etag, "ETAG_NOT_ADVANCED")
        self.check("stale_etag_patch_rejected", lambda: self.client.request("PATCH", case_path,
            body={"clinical": {"age": 63}}, headers={"If-Match": old_etag}).expected(412, "VERSION_CONFLICT"))
        unchanged = self.client.request("GET", case_path).expected(200)
        require(unchanged.data == patched.data and etag(unchanged) == etag(patched), "REJECTED_PATCH_CHANGED_CASE")
        self.check("prediction_stays_blocked", lambda: self.client.request("POST", case_path + "/executions",
            body={}).expected(503, "ENGINE_NOT_AVAILABLE"))
        fixture = synthetic_dicom()
        boundary = "kneeoaiSynthetic" + uuid.uuid4().hex
        multipart = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="synthetic-only.dcm"\r\n'
                     'Content-Type: application/dicom\r\n\r\n').encode("ascii") + fixture + f"\r\n--{boundary}--\r\n".encode("ascii")
        failed = self.check("unconfigured_preparation_returns_location", lambda: self.client.request("POST", case_path + "/preparations", raw=multipart,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "If-Match": etag(unchanged),
                     "Idempotency-Key": uuid.uuid4().hex}).expected(503, "PROCESSOR_UNAVAILABLE"))
        location = failed.headers.get("Location", "")
        require(bool(re.fullmatch(r"/api/v1/preparations/prep_[A-Za-z0-9_-]{1,120}", location)), "PREPARATION_LOCATION_INVALID")
        prepared = self.client.request("GET", location).expected(200)
        current = self.client.request("GET", case_path).expected(200)
        image = prepared.data.get("image")
        require(isinstance(image, dict), "IMAGE_REFERENCE_REQUIRED")
        state = {"format": STATE_FORMAT, "origin": self.client.origin, "createdAt": utc_now(),
            "evaluationId": case_id, "evaluationRevisionId": identifier(current.data.get("currentRevisionId"), "rev"),
            "evaluationEtag": etag(current), "preparationId": identifier(prepared.data.get("id"), "prep"),
            "preparationRevisionId": identifier(prepared.data.get("preparationRevisionId"), "prv"),
            "preparationEtag": etag(prepared), "imageId": identifier(image.get("imageId"), "img"),
            "imageVersion": identifier(image.get("imageVersion"), "iv"),
            "fixtureSha256": hashlib.sha256(fixture).hexdigest()}
        self.check("failed_source_metadata_persisted", lambda: self.preparation(state))
        self.check("case_snapshot_and_null_bmi_preserved", lambda: self.case(state))
        self.client.save_auth(self.args.cookie_file, overwrite=False)
        write_private(self.args.state_file, state, overwrite=False)

    def load_state(self):
        state = read_private(self.args.state_file)
        keys = {"format", "origin", "createdAt", "evaluationId", "evaluationRevisionId", "evaluationEtag",
                "preparationId", "preparationRevisionId", "preparationEtag", "imageId", "imageVersion", "fixtureSha256"}
        require(set(state) == keys and state["format"] == STATE_FORMAT and state["origin"] == self.client.origin, "STATE_FILE_INVALID")
        for name, prefix in (("evaluationId", "ev"), ("evaluationRevisionId", "rev"), ("preparationId", "prep"),
                             ("preparationRevisionId", "prv"), ("imageId", "img"), ("imageVersion", "iv")):
            identifier(state[name], prefix)
        require(state["fixtureSha256"] == hashlib.sha256(synthetic_dicom()).hexdigest(), "SYNTHETIC_FIXTURE_MISMATCH")
        return state

    def resume(self, *, restored: bool):
        state = self.load_state()
        previous_csrf = self.client.load_auth(self.args.cookie_file)
        if restored:
            self.check("restored_old_session_revoked", lambda: self.client.request("GET", f'/api/v1/evaluations/{state["evaluationId"]}').expected(401, "AUTHENTICATION_REQUIRED"))
            self.check("restored_fixture_can_login", self.client.login)
        else:
            self.check("session_survives_service_restart", lambda: self.client.bootstrap(authenticated=True))
            self.check("server_secret_survives_service_restart", lambda: require(
                hmac.compare_digest(previous_csrf, self.client.csrf), "SERVER_SECRET_CHANGED"))
        self.check("saved_case_snapshot_preserved", lambda: self.case(state))
        self.check("failed_source_metadata_preserved", lambda: self.preparation(state))
        self.check("prediction_still_blocked", lambda: self.client.request("POST", f'/api/v1/evaluations/{state["evaluationId"]}/executions', body={}).expected(503, "ENGINE_NOT_AVAILABLE"))
        self.client.save_auth(self.args.cookie_file, overwrite=True)

    def run(self):
        self.check("anonymous_https_health_no_cookie", self.health)
        if self.args.mode == "seed":
            self.seed()
        else:
            self.resume(restored=self.args.mode == "restored")
        self.report["status"] = "passed"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("seed", "resume", "restored"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--ca-file", required=True, type=Path)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--cookie-file", required=True, type=Path)
    parser.add_argument("--report-file", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    runner = None
    report_path_ready = False
    report = {"format": "kneeoai-https-smoke-report-v1", "mode": args.mode,
              "startedAt": utc_now(), "status": "failed", "checks": []}
    try:
        origin = loopback_origin(args.base_url)
        require(1 <= args.timeout <= 300, "TIMEOUT_INVALID")
        args.ca_file = args.ca_file.resolve(strict=True)
        for name in ("state_file", "cookie_file", "report_file"):
            # Preserve the final path component so link checks cannot be bypassed.
            path = getattr(args, name).absolute()
            setattr(args, name, path.parent.resolve() / path.name)
        require(len({args.ca_file, args.state_file, args.cookie_file, args.report_file}) == 4, "FILE_PATHS_MUST_DIFFER")
        report_path_ready = True
        client = Client(origin, args.ca_file, args.timeout)
        runner = Runner(client, args)
        runner.run()
        report = runner.report
    except Exception as error:
        if runner is not None:
            report = runner.report
        report["status"] = "failed"
        item = {"name": runner.current if runner else "configuration", "ok": False,
                "code": error.code if isinstance(error, SmokeFailure) else "LOCAL_SMOKE_ERROR"}
        if isinstance(error, SmokeFailure) and error.status is not None:
            item["httpStatus"] = error.status
        report["checks"].append(item)
    report["finishedAt"] = utc_now()
    if report_path_ready:
        try:
            write_private(args.report_file, report, overwrite=True)
        except Exception:
            report["status"] = "failed"
            report["checks"].append({"name": "private_report_written", "ok": False, "code": "REPORT_WRITE_FAILED"})
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
