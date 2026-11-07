from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

DEFAULT_RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
DEFAULT_ORIGINS = (
    "http://127.0.0.1:5173", "http://localhost:5173",
    "http://127.0.0.1:8000", "http://localhost:8000",
)

def production_path(value):
    path=Path(value)
    if not path.is_absolute():
        raise ValueError('Production paths must be explicit absolute paths')
    if any(candidate.is_symlink() or getattr(candidate,'is_junction',lambda:False)() for candidate in (path,*path.parents)):
        raise ValueError('Production storage must not follow symbolic links or junctions')
    return path.resolve()

def env_secret(environment):
    plain=environment.get('KNEEOAI_SECRET_KEY')
    filename=environment.get('KNEEOAI_SECRET_KEY_FILE')
    if plain is not None and filename is not None:
        raise ValueError('Configure exactly one server secret source')
    if filename is not None:
        if not filename:
            raise ValueError('The server secret file must be explicit')
        try:
            path=production_path(filename)
            if not path.is_file():
                raise ValueError('The server secret file must be a regular file')
            with path.open('rb') as stream:
                value=stream.read(4097)
        except OSError:
            raise ValueError('The server secret file could not be read') from None
        if value.endswith(b'\r\n'):
            value=value[:-2]
        elif value.endswith(b'\n'):
            value=value[:-1]
        if not re.fullmatch(rb'[0-9a-fA-F]{64}',value):
            raise ValueError('The secret file must contain one 64-character ASCII hexadecimal token')
        frontend=environment.get('KNEEOAI_FRONTEND_DIST')
        if frontend and path.is_relative_to(Path(frontend).resolve()):
            raise ValueError('The server secret must not be stored in the public distribution')
        return value
    return plain.encode('utf-8') if plain is not None else None

def normalized_origin(value: str) -> str:
    if any(character in value for character in '*\r\n\t'):
        raise ValueError('An exact HTTP(S) origin is required')
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
    secret_key: bytes | None = field(default=None,repr=False)
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS
    secure_cookies: bool = False
    environment: str = "development"
    session_seconds: int = 8 * 60 * 60
    anonymous_session_seconds: int = 20 * 60
    login_limit: int = 5
    login_window_seconds: int = 15 * 60
    busy_timeout_ms: int = 5000
    private_files_path: Path | None = None
    max_request_bytes: int = 64 * 1024 * 1024
    max_dicom_bytes: int = 60 * 1024 * 1024
    frozen_python: Path | None = None
    frozen_checkout: Path | None = None
    frozen_timeout_seconds: int = 120
    frontend_dist_path: Path | None = None
    data_dir: Path | None = None

    def __post_init__(self):
        if self.environment=='production':
            if self.data_dir is None:
                raise ValueError('Production requires an explicit persistent data directory')
            for value in (self.data_dir,self.database_path,self.private_files_path or Path(self.database_path).parent/'private_objects'):
                production_path(value)
            if self.frontend_dist_path is not None:
                production_path(self.frontend_dist_path)
        if self.data_dir is not None:
            object.__setattr__(self,'data_dir',Path(self.data_dir).resolve())
        object.__setattr__(self, "database_path", Path(self.database_path).resolve())
        object.__setattr__(self, "private_files_path", Path(self.private_files_path or self.database_path.parent / 'private_objects').resolve())
        for key in ('frozen_python', 'frozen_checkout'):
            if getattr(self, key) is not None:
                object.__setattr__(self, key, Path(getattr(self, key)).resolve())
        if self.frontend_dist_path is not None:
            # Keep the original path until the distribution validator can reject links/junctions.
            object.__setattr__(self,'frontend_dist_path',Path(self.frontend_dist_path).absolute())
        object.__setattr__(self, "allowed_origins", tuple(normalized_origin(v) for v in self.allowed_origins))
        if self.environment not in {"development", "production", "test"}:
            raise ValueError("Unsupported environment")
        if not self.allowed_origins:
            raise ValueError("At least one exact origin is required")
        if self.secret_key is not None and len(self.secret_key) < 32:
            raise ValueError("The server secret must contain at least 32 bytes")
        if any(v <= 0 for v in (self.session_seconds, self.anonymous_session_seconds, self.login_limit, self.login_window_seconds, self.busy_timeout_ms)):
            raise ValueError("Timeouts and limits must be positive")
        if not 0 < self.max_dicom_bytes <= self.max_request_bytes or self.frozen_timeout_seconds <= 0:
            raise ValueError('Upload limits and worker timeout must be positive and consistent')
        if self.environment == "production":
            if not self.secure_cookies or any(not v.startswith("https://") for v in self.allowed_origins):
                raise ValueError("Production requires Secure cookies and HTTPS origins")
            if not isinstance(self.secret_key,bytes) or len(self.secret_key)<32 or any(byte<=32 or byte>=127 for byte in self.secret_key):
                raise ValueError('Production requires an explicit printable server secret of at least 32 bytes')
            if not self.database_path.is_relative_to(self.data_dir) or self.database_path==self.data_dir or not self.private_files_path.is_relative_to(self.data_dir) or self.private_files_path==self.data_dir:
                raise ValueError('Application storage must stay inside the persistent data directory')
            if self.database_path==self.private_files_path or self.database_path.is_relative_to(self.private_files_path):
                raise ValueError('Database and private objects must use distinct paths')
            if self.frontend_dist_path is not None:
                public=self.frontend_dist_path.resolve()
                if public.is_relative_to(self.data_dir) or self.data_dir.is_relative_to(public):
                    raise ValueError('Public distribution and persistent data must be separate')
        elif self.environment == "development":
            if any(urlsplit(v).hostname not in {"localhost", "127.0.0.1", "::1"} for v in self.allowed_origins):
                raise ValueError("Development is restricted to loopback origins")

    @classmethod
    def from_env(cls,environ=None):
        source=os.environ if environ is None else environ
        environment = source.get("KNEEOAI_ENV", "development")
        origins = source.get("KNEEOAI_ALLOWED_ORIGINS")
        data_dir=Path(source['KNEEOAI_DATA_DIR']) if source.get('KNEEOAI_DATA_DIR') else None
        database=Path(source['KNEEOAI_DB_PATH']) if source.get('KNEEOAI_DB_PATH') else (data_dir/'application.sqlite3' if data_dir is not None else DEFAULT_RUNTIME/'application.sqlite3')
        private=Path(source['KNEEOAI_PRIVATE_FILES']) if source.get('KNEEOAI_PRIVATE_FILES') else (data_dir/'private_objects' if data_dir is not None else None)
        return cls(
            database_path=database,
            secret_key=env_secret(source),
            allowed_origins=tuple(v.strip() for v in origins.split(",") if v.strip()) if origins else DEFAULT_ORIGINS,
            secure_cookies=environment == "production",
            environment=environment,
            private_files_path=private,
            frozen_python=Path(source['KNEEOAI_FROZEN_PYTHON']) if source.get('KNEEOAI_FROZEN_PYTHON') else None,
            frozen_checkout=Path(source['KNEEOAI_FROZEN_CHECKOUT']) if source.get('KNEEOAI_FROZEN_CHECKOUT') else None,
            frontend_dist_path=Path(source['KNEEOAI_FRONTEND_DIST']) if source.get('KNEEOAI_FRONTEND_DIST') else None,
            data_dir=data_dir,
        )
