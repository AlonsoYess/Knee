from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets

from pwdlib import PasswordHash

PASSWORD_HASH = PasswordHash.recommended()
COOKIE_NAME = "kneeoai_session"
COOKIE_PATH = "/api/v1"

def password_hash(password: str) -> str:
    if not isinstance(password, str) or len(password) < 12 or len(password) > 1024:
        raise ValueError("Use a password of 12 to 1024 characters")
    return PASSWORD_HASH.hash(password)

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        return PASSWORD_HASH.verify(password, stored_hash)
    except Exception:
        return False

def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()

def new_token() -> str:
    return secrets.token_urlsafe(32)

def csrf_token(secret: bytes, session_token: str) -> str:
    return hmac.new(secret, b"csrf:" + session_token.encode("ascii"), hashlib.sha256).hexdigest()

def verify_csrf(secret: bytes, session_token: str, candidate: str | None) -> bool:
    if not candidate or not re.fullmatch(r'[a-f0-9]{64}', candidate):
        return False
    return hmac.compare_digest(csrf_token(secret, session_token), candidate)

def load_secret(configured: bytes | None, database_path: Path) -> bytes:
    if configured is not None:
        return configured
    path = database_path.parent / "session.key"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(secrets.token_bytes(32))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except FileExistsError:
        pass
    value = path.read_bytes()
    if len(value) != 32:
        raise ValueError("Invalid persisted server secret; do not overwrite it automatically")
    return value
