from __future__ import annotations

import base64
from contextlib import closing
import hashlib
import hmac
import json
import logging
import math
import re
import sqlite3
import time
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException

from .config import Settings, normalized_origin
from .db import Database, opaque, utc
from .errors import ApiError
from .models import Login, CreateEvaluation, UpdateEvaluation
from .security import COOKIE_NAME, COOKIE_PATH, csrf_token, load_secret, new_token, password_hash, token_hash, verify_csrf, verify_password

LOGGER = logging.getLogger("knee_app")
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")

def create_app(settings: Settings | None = None, *, clock: Callable[[], float] = time.time) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(settings.database_path, settings.busy_timeout_ms)
    db.initialize()
    secret = load_secret(settings.secret_key, settings.database_path)
    dummy_hash = password_hash(new_token())
    app = FastAPI(title="KneeOAI application API", version="0.2.0", docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    app.state.settings, app.state.db, app.state.clock = settings, db, clock

    def now():
        return float(app.state.clock())

    def error_response(request, error):
        return JSONResponse(
            status_code=error.status,
            content={"error": {"code": error.code, "message": error.message,
                "requestId": request.state.request_id, "fields": error.fields,
                "retryable": bool(error.retryable)}},
            headers=error.headers,
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = opaque("req")
        try:
            if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'} and request.url.path.startswith('/api/v1/'):
                require_mutation(request, authenticated=not (request.url.path == '/api/v1/session' and request.method == 'POST'))
            response = await call_next(request)
        except ApiError as error:
            response = error_response(request, error)
        except sqlite3.Error:
            response = error_response(request, ApiError(503, "STORAGE_UNAVAILABLE", "No se pudo confirmar la operación. Recupera el estado actual antes de reintentar.", retryable=True))
        except Exception as exc:
            LOGGER.error("Application failure requestId=%s type=%s", request.state.request_id, type(exc).__name__)
            response = error_response(request, ApiError(500, "INTERNAL_ERROR", "No se pudo completar la operación."))
        response.headers["X-Request-Id"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ApiError)
    async def handled_error(request, error):
        return error_response(request, error)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        fields = [{"path": ".".join(str(x) for x in item["loc"] if x != "body") or "body",
                   "code": "INVALID_VALUE", "message": "Revisa el formato o los campos permitidos."}
                  for item in error.errors()]
        return error_response(request, ApiError(422, "VALIDATION_FAILED", "Revisa los datos indicados.", fields=fields))

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        code = "RESOURCE_NOT_FOUND" if error.status_code == 404 else "METHOD_NOT_ALLOWED" if error.status_code == 405 else "REQUEST_FAILED"
        return error_response(request, ApiError(error.status_code, code, "No se encontró el recurso solicitado." if error.status_code == 404 else "La solicitud no está permitida."))

    def read_session(request):
        token = request.cookies.get(COOKIE_NAME)
        if not token or not TOKEN_PATTERN.fullmatch(token):
            return None, None
        with closing(db.connect()) as connection:
            row = connection.execute(
                "SELECT s.*,u.display_name,u.role,u.active FROM sessions s LEFT JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>?",
                (token_hash(token), now()),
            ).fetchone()
        if row is None or (row["user_id"] is not None and row["active"] != 1):
            return None, None
        return row, token

    def require_session(request, authenticated=True):
        row, token = read_session(request)
        if row is None or (authenticated and row["user_id"] is None):
            raise ApiError(401, "AUTHENTICATION_REQUIRED", "Inicia sesión para continuar.")
        return row, token

    def require_mutation(request, authenticated=True):
        row, token = require_session(request, authenticated)
        try:
            origin = normalized_origin(request.headers.get("origin", ""))
        except (ValueError, TypeError):
            origin = None
        if origin not in settings.allowed_origins or not verify_csrf(secret, token, request.headers.get("x-csrf-token")):
            raise ApiError(403, "ACCESS_DENIED", "No se pudo autorizar esta operación.")
        return row, token

    def session_payload(row, token):
        authenticated = row["user_id"] is not None
        return {"authenticated": authenticated,
                "user": {"id": row["user_id"], "displayName": row["display_name"], "role": row["role"]} if authenticated else None,
                "expiresAt": utc(row["expires_at"]) if authenticated else None,
                "csrfToken": csrf_token(secret, token)}

    def set_cookie(response, token, expires_at):
        response.set_cookie(COOKIE_NAME, token, max_age=max(1, math.ceil(expires_at - now())),
            httponly=True, secure=settings.secure_cookies, samesite="lax", path=COOKIE_PATH)

    def ownership(connection, evaluation_id, owner_id):
        row = connection.execute("SELECT * FROM evaluations WHERE id=? AND owner_id=?", (evaluation_id, owner_id)).fetchone()
        if row is None:
            raise ApiError(404, "RESOURCE_NOT_FOUND", "No se encontró el recurso solicitado.")
        return row

    def etag(row):
        return f'"{row["id"]}-v{row["version"]}"'

    def require_version(request, row):
        supplied = request.headers.get("if-match")
        if supplied is None:
            raise ApiError(428, "VERSION_REQUIRED", "Recupera la versión actual antes de guardar.")
        if supplied != etag(row):
            raise ApiError(412, "VERSION_CONFLICT", "El caso cambió. Recupera su versión actual antes de guardar.", headers={"ETag": etag(row)})

    def representation(row):
        return {"id": row["id"], "code": row["code"], "currentRevisionId": row["revision_id"],
                "clinical": {key: row[key] for key in ("age", "sex", "bmi", "side")},
                "preparationId": None, "reviewId": None, "currentExecutionId": None, "updatedAt": row["updated_at"]}

    @app.get("/api/v1/session")
    def get_session(request: Request):
        row, token = read_session(request)
        if row is None:
            token = new_token()
            moment = now()
            expires_at = moment + settings.anonymous_session_seconds
            with db.transaction() as connection:
                connection.execute("DELETE FROM sessions WHERE expires_at<=?", (moment,))
                connection.execute("INSERT INTO sessions(token_hash,user_id,created_at,expires_at) VALUES(?,NULL,?,?)", (token_hash(token), moment, expires_at))
            row = {"user_id": None, "expires_at": expires_at}
        response = JSONResponse(session_payload(row, token))
        set_cookie(response, token, row["expires_at"])
        return response

    def throttle_keys(request, email):
        peer = request.client.host if request.client else "unknown"
        # Never trust forwarded client-IP headers here. Deployment must specify trusted proxy behavior.
        return [hmac.new(secret, f"login-account:{email}".encode(), hashlib.sha256).hexdigest(),
                hmac.new(secret, f"login-peer:{peer}".encode(), hashlib.sha256).hexdigest()]

    @app.post("/api/v1/session")
    def login(request: Request, payload: Login):
        row, old_token = require_mutation(request, authenticated=False)
        if row["user_id"] is not None:
            raise ApiError(409, "SESSION_ALREADY_ACTIVE", "Cierra la sesión actual antes de cambiar de cuenta.")
        moment = now()
        keys = throttle_keys(request, payload.email)
        rejected = False
        with db.transaction() as connection:
            live = connection.execute("SELECT * FROM sessions WHERE token_hash=? AND user_id IS NULL AND expires_at>?", (token_hash(old_token), moment)).fetchone()
            if live is None:
                raise ApiError(401, "AUTHENTICATION_REQUIRED", "Actualiza el acceso antes de continuar.")
            for key in keys:
                attempt = connection.execute("SELECT * FROM login_attempts WHERE key_hash=?", (key,)).fetchone()
                if attempt and moment < attempt["window_start"] + settings.login_window_seconds and attempt["failures"] >= settings.login_limit:
                    wait = max(1, math.ceil(attempt["window_start"] + settings.login_window_seconds - moment))
                    raise ApiError(429, "LOGIN_RATE_LIMITED", "Espera antes de volver a intentar el acceso.", retryable=True, headers={"Retry-After": str(wait)})
            user = connection.execute("SELECT * FROM users WHERE email=?", (payload.email,)).fetchone()
            valid = verify_password(payload.password, user["password_hash"] if user else dummy_hash)
            if not user or user["active"] != 1 or not valid:
                rejected = True
                for key in keys:
                    connection.execute(
                        "INSERT INTO login_attempts(key_hash,window_start,failures) VALUES(?,?,1) ON CONFLICT(key_hash) DO UPDATE SET failures=CASE WHEN window_start+?<=? THEN 1 ELSE failures+1 END,window_start=CASE WHEN window_start+?<=? THEN excluded.window_start ELSE window_start END",
                        (key, moment, settings.login_window_seconds, moment, settings.login_window_seconds, moment),
                    )
            else:
                token = new_token()
                expires_at = moment + settings.session_seconds
                connection.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(old_token),))
                connection.execute("INSERT INTO sessions(token_hash,user_id,created_at,expires_at) VALUES(?,?,?,?)", (token_hash(token), user["id"], moment, expires_at))
                connection.execute("DELETE FROM login_attempts WHERE key_hash=?", (keys[0],))
        if rejected:
            raise ApiError(401, "AUTHENTICATION_REQUIRED", "No se pudo iniciar sesión con esos datos.")
        active = {"user_id": user["id"], "display_name": user["display_name"], "role": user["role"], "expires_at": expires_at}
        response = JSONResponse(session_payload(active, token))
        set_cookie(response, token, expires_at)
        return response

    @app.delete("/api/v1/session", status_code=204)
    def logout(request: Request):
        _, token = require_mutation(request)
        with db.transaction() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(token),))
        response = Response(status_code=204)
        response.delete_cookie(COOKIE_NAME, path=COOKIE_PATH, httponly=True, secure=settings.secure_cookies, samesite="lax")
        return response

    def encode_cursor(row, owner):
        raw = json.dumps([row["created_at"], row["id"]], separators=(",", ":")).encode()
        payload = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        signature = hmac.new(secret, f"cursor:{owner}:{payload}".encode(), hashlib.sha256).hexdigest()
        return payload + "." + signature

    def decode_cursor(cursor, owner):
        try:
            if len(cursor) > 1024:
                raise ValueError()
            payload, signature = cursor.split(".")
            expected = hmac.new(secret, f"cursor:{owner}:{payload}".encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError()
            values = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            if not isinstance(values, list) or len(values) != 2 or not all(isinstance(x, str) for x in values):
                raise ValueError()
            return values
        except Exception:
            raise ApiError(422, "VALIDATION_FAILED", "El cursor de la lista no es válido.")

    @app.get("/api/v1/evaluations")
    def list_evaluations(request: Request, cursor: str | None = None, limit: int = 25):
        session, _ = require_session(request)
        if not 1 <= limit <= 100:
            raise ApiError(422, "VALIDATION_FAILED", "El límite de la lista debe estar entre 1 y 100.")
        owner = session["user_id"]
        args = [owner]
        where = "owner_id=?"
        if cursor:
            created, identifier = decode_cursor(cursor, owner)
            where += " AND (created_at<? OR (created_at=? AND id<?))"
            args.extend([created, created, identifier])
        args.append(limit + 1)
        with closing(db.connect()) as connection:
            rows = connection.execute(f"SELECT * FROM evaluations WHERE {where} ORDER BY created_at DESC,id DESC LIMIT ?", args).fetchall()
        visible = rows[:limit]
        return {"items": [representation(row) for row in visible],
                "nextCursor": encode_cursor(visible[-1], owner) if len(rows) > limit else None}

    @app.post("/api/v1/evaluations", status_code=201)
    def create_evaluation(request: Request, payload: CreateEvaluation):
        session, _ = require_mutation(request)
        moment = now()
        identifier, revision = opaque("ev"), opaque("rev")
        clinical = payload.clinical.model_dump()
        request_key = request.headers.get('idempotency-key')
        if request_key is not None and not re.fullmatch(r'[A-Za-z0-9_-]{16,128}', request_key):
            raise ApiError(422, 'VALIDATION_FAILED', 'La clave de la solicitud no es válida.')
        body_hash = hashlib.sha256(json.dumps(clinical, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with db.transaction() as connection:
            if request_key is not None:
                prior = connection.execute('SELECT body_hash,evaluation_id FROM create_requests WHERE owner_id=? AND request_key=?', (session['user_id'], request_key)).fetchone()
                if prior is not None:
                    if not hmac.compare_digest(prior['body_hash'], body_hash):
                        raise ApiError(409, 'IDEMPOTENCY_CONFLICT', 'Esta clave ya se utilizó con otros datos.')
                    existing = ownership(connection, prior['evaluation_id'], session['user_id'])
                    return JSONResponse(representation(existing), status_code=200, headers={'ETag': etag(existing)})
            connection.execute(
                "INSERT INTO evaluations(id,code,owner_id,age,sex,bmi,side,version,revision_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,1,?,?,?)",
                (identifier, "EV-" + identifier[3:13].upper(), session["user_id"], clinical["age"], clinical["sex"], clinical["bmi"], clinical["side"], revision, utc(moment), utc(moment)),
            )
            db.record_revision(connection, evaluation_id=identifier, revision_id=revision, clinical=clinical, now=moment)
            if request_key is not None:
                connection.execute('INSERT INTO create_requests(owner_id,request_key,body_hash,evaluation_id,created_at) VALUES(?,?,?,?,?)', (session['user_id'], request_key, body_hash, identifier, utc(moment)))
            row = ownership(connection, identifier, session["user_id"])
        return JSONResponse(representation(row), status_code=201, headers={"ETag": etag(row), "Location": f"/api/v1/evaluations/{identifier}"})

    @app.get("/api/v1/evaluations/{evaluationId}")
    def get_evaluation(evaluationId: str, request: Request):
        session, _ = require_session(request)
        with closing(db.connect()) as connection:
            row = ownership(connection, evaluationId, session["user_id"])
        return JSONResponse(representation(row), headers={"ETag": etag(row)})

    @app.patch("/api/v1/evaluations/{evaluationId}")
    def update_evaluation(evaluationId: str, request: Request, payload: UpdateEvaluation):
        session, _ = require_mutation(request)
        with db.transaction() as connection:
            previous = ownership(connection, evaluationId, session["user_id"])
            require_version(request, previous)
            clinical = {key: previous[key] for key in ("age", "sex", "bmi", "side")}
            clinical.update(payload.clinical.model_dump(exclude_unset=True))
            moment, revision = now(), opaque("rev")
            changed = connection.execute(
                "UPDATE evaluations SET age=?,sex=?,bmi=?,side=?,version=version+1,revision_id=?,updated_at=? WHERE id=? AND owner_id=? AND version=?",
                (clinical["age"], clinical["sex"], clinical["bmi"], clinical["side"], revision, utc(moment), evaluationId, session["user_id"], previous["version"]),
            )
            if changed.rowcount != 1:
                raise ApiError(412, "VERSION_CONFLICT", "El caso cambió. Recupera su versión actual antes de guardar.")
            db.record_revision(connection, evaluation_id=evaluationId, revision_id=revision, clinical=clinical, now=moment)
            row = ownership(connection, evaluationId, session["user_id"])
        return JSONResponse(representation(row), headers={"ETag": etag(row)})

    @app.get("/api/v1/engine-status")
    def engine_status(request: Request):
        require_session(request)
        return {"status": "not_configured", "package": None, "message": "Motor predictivo pendiente de formalización."}

    @app.post("/api/v1/evaluations/{evaluationId}/executions")
    def disabled_prediction(evaluationId: str, request: Request):
        session, _ = require_mutation(request)
        with closing(db.connect()) as connection:
            ownership(connection, evaluationId, session["user_id"])
        raise ApiError(503, "ENGINE_NOT_AVAILABLE", "El motor predictivo todavía no está disponible.", retryable=True)

    return app
