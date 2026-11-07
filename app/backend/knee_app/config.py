from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
DEFAULT_ORIGINS = (
    "http://127.0.0.1:5173", "http://localhost:5173",
    "http://127.0.0.1:8000", "http://localhost:8000",
)

def normalized_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("An exact HTTP(S) origin is required")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Origins must not contain paths, queries or fragments")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    suffix = "" if port is None or (parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443) else f":{port}"
    return f"{parsed.scheme}://{host}{suffix}"

@dataclass(frozen=True)
class Settings:
    database_path: Path = DEFAULT_RUNTIME / "application.sqlite3"
    secret_key: bytes | None = None
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS
    secure_cookies: bool = False
    environment: str = "development"
    session_seconds: int = 8 * 60 * 60
    anonymous_session_seconds: int = 20 * 60
    login_limit: int = 5
    login_window_seconds: int = 15 * 60
    busy_timeout_ms: int = 5000

    def __post_init__(self):
        object.__setattr__(self, "database_path", Path(self.database_path).resolve())
        object.__setattr__(self, "allowed_origins", tuple(normalized_origin(v) for v in self.allowed_origins))
        if self.environment not in {"development", "production", "test"}:
            raise ValueError("Unsupported environment")
        if not self.allowed_origins:
            raise ValueError("At least one exact origin is required")
        if self.secret_key is not None and len(self.secret_key) < 32:
            raise ValueError("The server secret must contain at least 32 bytes")
        if any(v <= 0 for v in (self.session_seconds, self.anonymous_session_seconds, self.login_limit, self.login_window_seconds, self.busy_timeout_ms)):
            raise ValueError("Timeouts and limits must be positive")
        if self.environment == "production":
            if not self.secure_cookies or any(not v.startswith("https://") for v in self.allowed_origins):
                raise ValueError("Production requires Secure cookies and HTTPS origins")
        elif self.environment == "development":
            if any(urlsplit(v).hostname not in {"localhost", "127.0.0.1", "::1"} for v in self.allowed_origins):
                raise ValueError("Development is restricted to loopback origins")

    @classmethod
    def from_env(cls):
        environment = os.environ.get("KNEEOAI_ENV", "development")
        origins = os.environ.get("KNEEOAI_ALLOWED_ORIGINS")
        secret = os.environ.get("KNEEOAI_SECRET_KEY")
        return cls(
            database_path=Path(os.environ.get("KNEEOAI_DB_PATH", DEFAULT_RUNTIME / "application.sqlite3")),
            secret_key=secret.encode("utf-8") if secret else None,
            allowed_origins=tuple(v.strip() for v in origins.split(",") if v.strip()) if origins else DEFAULT_ORIGINS,
            secure_cookies=environment == "production",
            environment=environment,
        )
