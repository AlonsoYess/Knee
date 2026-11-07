from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid

from .security import password_hash

SCHEMA_VERSION = 1
EXPECTED_COLUMNS = {
    'users': {'id','email','display_name','password_hash','role','active','created_at'},
    'sessions': {'token_hash','user_id','created_at','expires_at'},
    'login_attempts': {'key_hash','window_start','failures'},
    'evaluations': {'id','code','owner_id','age','sex','bmi','side','version','revision_id','created_at','updated_at'},
    'evaluation_revisions': {'revision_id','evaluation_id','clinical_json','created_at'},
    'create_requests': {'owner_id','request_key','body_hash','evaluation_id','created_at'},
}

def verify_schema(connection):
    if connection.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
        raise ValueError('Unsupported database schema version')
    for table, columns in EXPECTED_COLUMNS.items():
        actual = {row[1] for row in connection.execute(f'PRAGMA table_info({table})')}
        if not columns.issubset(actual):
            raise ValueError('Incomplete or incompatible database schema')
    if connection.execute('PRAGMA foreign_key_check').fetchone() is not None:
        raise ValueError('Database reference integrity check failed')

def utc(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def opaque(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def normalize_email(email: str) -> str:
    value = email.strip().casefold()
    if len(value) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
        raise ValueError("A valid email address is required")
    return value

class Database:
    def __init__(self, path: Path, busy_timeout_ms: int = 5000):
        self.path = Path(path).resolve()
        self.busy_timeout_ms = busy_timeout_ms

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
        return connection

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(self.connect()) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, SCHEMA_VERSION}:
                raise ValueError("Unsupported database schema version")
            if version == SCHEMA_VERSION:
                verify_schema(connection)
                connection.execute("PRAGMA journal_mode = WAL")
                if os.name != 'nt':
                    os.chmod(self.path, 0o600)
                return
            if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone():
                raise ValueError("Refusing to initialize an unrecognized nonempty database")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE users (
                    id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('doctor','administrator')),
                    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)), created_at TEXT NOT NULL
                );
                CREATE TABLE sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                    created_at REAL NOT NULL, expires_at REAL NOT NULL
                );
                CREATE INDEX sessions_user ON sessions(user_id);
                CREATE INDEX sessions_expiry ON sessions(expires_at);
                CREATE TABLE login_attempts (
                    key_hash TEXT PRIMARY KEY, window_start REAL NOT NULL, failures INTEGER NOT NULL
                );
                CREATE TABLE evaluations (
                    id TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL REFERENCES users(id),
                    age INTEGER, sex TEXT CHECK(sex IS NULL OR sex IN ('F','M')), bmi REAL,
                    side TEXT CHECK(side IS NULL OR side IN ('right','left')),
                    version INTEGER NOT NULL CHECK(version > 0), revision_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    CHECK(age IS NULL OR age > 0), CHECK(bmi IS NULL OR bmi > 0)
                );
                CREATE INDEX evaluations_owner_created ON evaluations(owner_id, created_at DESC, id DESC);
                CREATE TABLE evaluation_revisions (
                    revision_id TEXT PRIMARY KEY,
                    evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
                    clinical_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE create_requests (
                    owner_id TEXT NOT NULL REFERENCES users(id), request_key TEXT NOT NULL,
                    body_hash TEXT NOT NULL, evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
                    created_at TEXT NOT NULL, PRIMARY KEY(owner_id, request_key)
                );
                PRAGMA user_version = 1;
                COMMIT;
            """)
            verify_schema(connection)
        if os.name != 'nt':
            os.chmod(self.path, 0o600)

    def record_revision(self, connection, *, evaluation_id: str, revision_id: str, clinical: dict, now: float):
        connection.execute(
            "INSERT INTO evaluation_revisions(revision_id,evaluation_id,clinical_json,created_at) VALUES(?,?,?,?)",
            (revision_id, evaluation_id, json.dumps(clinical, allow_nan=False, sort_keys=True, separators=(",", ":")), utc(now)),
        )

def create_user(db: Database, *, email: str, password: str, display_name: str, role: str = "doctor", now: float | None = None) -> str:
    email = normalize_email(email)
    name = display_name.strip()
    if not name or len(name) > 120 or role not in {"doctor", "administrator"}:
        raise ValueError("A display name and supported role are required")
    hashed = password_hash(password)
    user_id = opaque("usr")
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO users(id,email,display_name,password_hash,role,active,created_at) VALUES(?,?,?,?,?,1,?)",
            (user_id, email, name, hashed, role, utc(time.time() if now is None else now)),
        )
    return user_id

def set_password(db: Database, *, email: str, password: str):
    email = normalize_email(email)
    hashed = password_hash(password)
    with db.transaction() as connection:
        user = connection.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if user is None:
            raise ValueError("Account not found")
        connection.execute("UPDATE users SET password_hash=? WHERE id=?", (hashed, user["id"]))
        connection.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))

def disable_user(db: Database, *, email: str):
    with db.transaction() as connection:
        user = connection.execute("SELECT id FROM users WHERE email=?", (normalize_email(email),)).fetchone()
        if user is None:
            raise ValueError("Account not found")
        connection.execute("UPDATE users SET active=0 WHERE id=?", (user["id"],))
        connection.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))

def _readonly(path: Path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)

def _verify_source(connection):
    verify_schema(connection)
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("Database integrity check failed")

def backup_database(source: Path, destination: Path):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Backup destination must be a new file")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with closing(_readonly(source)) as original:
        _verify_source(original)
        # Exclusive creation prevents overwriting a concurrent backup.
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        try:
            with closing(sqlite3.connect(destination)) as snapshot:
                original.backup(snapshot)
                _verify_source(snapshot)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

def restore_database(source: Path, destination: Path):
    # Restore into a new path only. Switch the stopped application's configuration afterwards.
    backup_database(source, destination)
    try:
        with closing(sqlite3.connect(destination)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM login_attempts")
            connection.commit()
            _verify_source(connection)
    except BaseException:
        Path(destination).unlink(missing_ok=True)
        raise
