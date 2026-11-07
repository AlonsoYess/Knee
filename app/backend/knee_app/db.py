from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import shutil
import tempfile
import time
import uuid

from .security import password_hash

SCHEMA_VERSION = 2
BASE_COLUMNS = {
    'users': {'id','email','display_name','password_hash','role','active','created_at'},
    'sessions': {'token_hash','user_id','created_at','expires_at'},
    'login_attempts': {'key_hash','window_start','failures'},
    'evaluations': {'id','code','owner_id','age','sex','bmi','side','version','revision_id','created_at','updated_at'},
    'evaluation_revisions': {'revision_id','evaluation_id','clinical_json','created_at'},
    'create_requests': {'owner_id','request_key','body_hash','evaluation_id','created_at'},
}
EXPECTED_COLUMNS = {
    **BASE_COLUMNS,
    'evaluations': BASE_COLUMNS['evaluations'] | {'preparation_id','review_id'},
    'preparations': {'id','evaluation_id','source_revision_id','image_id','image_version','source_sha','side','source_key','source_dir','candidate_dir','status','revision_id','version','source_json','candidate_json','assessment_json','reference_id','active_review_id','error_json','pipeline_version','created_at','updated_at'},
    'source_references': {'id','preparation_id','source_revision_id','payload_json','worker_json','reviewer_id','created_at'},
    'quality_reviews': {'id','preparation_id','preparation_revision_id','payload_json','assessment_json','reviewer_id','created_at'},
    'artifacts': {'id','preparation_id','revision_id','role','object_key','sha256','byte_length'},
    'preparation_requests': {'owner_id','evaluation_id','operation','request_key','body_hash','preparation_id','resource_id','created_at'},
    'preparation_activations': {'id','evaluation_id','preparation_id','reviewer_id','reused','created_at'},
    'preparation_revisions': {'id','preparation_id','source_dir','candidate_dir','source_json','candidate_json','reference_id','pipeline_version','status','created_at'},
}

def verify_schema(connection):
    version = connection.execute('PRAGMA user_version').fetchone()[0]
    if version not in {1, SCHEMA_VERSION}:
        raise ValueError('Unsupported database schema version')
    for table, columns in (BASE_COLUMNS if version == 1 else EXPECTED_COLUMNS).items():
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
            if version not in {0, 1, SCHEMA_VERSION}:
                raise ValueError("Unsupported database schema version")
            if version in {1, SCHEMA_VERSION}:
                verify_schema(connection)
                if version == 1:
                    self._migrate_v2(connection)
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
            self._migrate_v2(connection)
            verify_schema(connection)
        if os.name != 'nt':
            os.chmod(self.path, 0o600)

    def record_revision(self, connection, *, evaluation_id: str, revision_id: str, clinical: dict, now: float):
        connection.execute(
            "INSERT INTO evaluation_revisions(revision_id,evaluation_id,clinical_json,created_at) VALUES(?,?,?,?)",
            (revision_id, evaluation_id, json.dumps(clinical, allow_nan=False, sort_keys=True, separators=(",", ":")), utc(now)),
        )

    def _migrate_v2(self, connection):
        # Additive migration: existing clinical rows and immutable revisions remain untouched.
        connection.executescript('''
            BEGIN IMMEDIATE;
            CREATE TABLE preparations (
                id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
                source_revision_id TEXT NOT NULL REFERENCES evaluation_revisions(revision_id),
                image_id TEXT NOT NULL, image_version TEXT NOT NULL, source_sha TEXT NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('right','left')), source_key TEXT NOT NULL,
                source_dir TEXT, candidate_dir TEXT,
                status TEXT NOT NULL CHECK(status IN ('queued','processing','awaiting_reference','ready','failed')),
                revision_id TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
                source_json TEXT, candidate_json TEXT, assessment_json TEXT,
                reference_id TEXT, active_review_id TEXT, error_json TEXT, pipeline_version TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(evaluation_id,source_sha,side)
            );
            CREATE TABLE source_references (
                id TEXT PRIMARY KEY, preparation_id TEXT NOT NULL UNIQUE REFERENCES preparations(id),
                source_revision_id TEXT NOT NULL, payload_json TEXT NOT NULL, worker_json TEXT NOT NULL,
                reviewer_id TEXT NOT NULL REFERENCES users(id), created_at TEXT NOT NULL
            );
            CREATE TABLE preparation_revisions (
                id TEXT PRIMARY KEY, preparation_id TEXT NOT NULL REFERENCES preparations(id),
                source_dir TEXT, candidate_dir TEXT, source_json TEXT, candidate_json TEXT,
                reference_id TEXT, pipeline_version TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE quality_reviews (
                id TEXT PRIMARY KEY, preparation_id TEXT NOT NULL REFERENCES preparations(id),
                preparation_revision_id TEXT NOT NULL, payload_json TEXT NOT NULL, assessment_json TEXT NOT NULL,
                reviewer_id TEXT NOT NULL REFERENCES users(id), created_at TEXT NOT NULL
            );
            CREATE TABLE artifacts (
                id TEXT PRIMARY KEY, preparation_id TEXT NOT NULL REFERENCES preparations(id),
                revision_id TEXT NOT NULL, role TEXT NOT NULL, object_key TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL, byte_length INTEGER NOT NULL CHECK(byte_length>=0)
            );
            CREATE INDEX artifacts_preparation ON artifacts(preparation_id,role);
            CREATE TABLE preparation_requests (
                owner_id TEXT NOT NULL REFERENCES users(id), evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
                operation TEXT NOT NULL, request_key TEXT NOT NULL, body_hash TEXT NOT NULL,
                preparation_id TEXT NOT NULL REFERENCES preparations(id), resource_id TEXT, created_at TEXT NOT NULL,
                PRIMARY KEY(owner_id,evaluation_id,operation,request_key)
            );
            CREATE TABLE preparation_activations (
                id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
                preparation_id TEXT NOT NULL REFERENCES preparations(id), reviewer_id TEXT NOT NULL REFERENCES users(id),
                reused INTEGER NOT NULL CHECK(reused IN (0,1)), created_at TEXT NOT NULL
            );
            ALTER TABLE evaluations ADD COLUMN preparation_id TEXT REFERENCES preparations(id) DEFAULT NULL;
            ALTER TABLE evaluations ADD COLUMN review_id TEXT REFERENCES quality_reviews(id) DEFAULT NULL;
            PRAGMA user_version = 2;
            COMMIT;
        ''')

    def interrupt_preparations(self, now: float):
        with self.transaction() as connection:
            connection.execute("UPDATE preparations SET status='failed',error_json=?,version=version+1,updated_at=? WHERE status IN ('queued','processing')", (
                json.dumps({'code':'PROCESSING_INTERRUPTED','message':'La preparación se interrumpió. Puedes reintentar con la misma fuente y referencia.'}), utc(now),
            ))

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

def _copy_database(source: Path, destination: Path, *, include_objects=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Backup destination must be a new file")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with closing(_readonly(source)) as original:
        _verify_source(original)
        if not include_objects and original.execute('PRAGMA user_version').fetchone()[0] >= 2 and original.execute('SELECT 1 FROM artifacts LIMIT 1').fetchone():
            raise ValueError('This database references private files; use a complete package backup')
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

def backup_database(source: Path, destination: Path):
    _copy_database(source,destination)

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

def _object_path(root, key):
    from pathlib import PurePosixPath
    if not isinstance(key,str) or '\\' in key:
        raise ValueError('Invalid private object key')
    parts=PurePosixPath(key).parts
    if len(parts)!=3 or parts[0]!='objects' or not re.fullmatch(r'[a-f0-9]{32}',parts[1]) or not re.fullmatch(r'[a-z_]+\.(dcm|png|npy|json)',parts[2]):
        raise ValueError('Invalid private object key')
    root=Path(root).absolute()
    if root.is_symlink() or getattr(root,'is_junction',lambda:False)():
        raise ValueError('Symbolic links are not valid package roots')
    root=root.resolve()
    path=root
    for part in parts:
        path=path/part
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)():
            raise ValueError('Symbolic links are not valid package objects')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Private object is outside the package')
    return path

def _package_stage(destination):
    destination=Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Package destination must be a new directory')
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    return destination,Path(tempfile.mkdtemp(prefix='.knee-package-',dir=destination.parent))

def _remove_package_stage(stage,destination):
    stage=Path(stage).resolve()
    if stage.parent!=Path(destination).resolve().parent or not stage.name.startswith('.knee-package-'):
        raise ValueError('Refusing to remove a non-staging directory')
    if stage.exists():
        shutil.rmtree(stage)

def _copy_private_file(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    descriptor=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with source.open('rb') as incoming,os.fdopen(descriptor,'wb') as outgoing:
        shutil.copyfileobj(incoming,outgoing,1024*1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())

def backup_package(source:Path,destination:Path,private_files_path:Path|None=None):
    from .private_files import digest_file
    source=Path(source).resolve()
    root=Path(private_files_path or source.parent/'private_objects').resolve()
    destination,stage=_package_stage(destination)
    try:
        database=stage/'application.sqlite3'
        _copy_database(source,database,include_objects=True)
        with closing(_readonly(database)) as connection:
            version=connection.execute('PRAGMA user_version').fetchone()[0]
            objects=[]
            if version>=2:
                objects=[{'object_key':row[0],'sha256':row[1],'byte_length':row[2]} for row in connection.execute('SELECT object_key,sha256,byte_length FROM artifacts ORDER BY object_key')]
        manifest_objects=[]
        for item in objects:
            original=_object_path(root,item['object_key'])
            target=_object_path(stage/'private_objects',item['object_key'])
            _copy_private_file(original,target)
            digest,size=digest_file(target)
            if digest!=item['sha256'] or size!=item['byte_length']:
                raise ValueError('Private file does not match the database snapshot')
            manifest_objects.append({'key':item['object_key'],'sha256':digest,'bytes':size})
        digest,size=digest_file(database)
        manifest={'format':'kneeoai-private-package-v1','database':{'file':'application.sqlite3','sha256':digest,'bytes':size},'objects':manifest_objects}
        descriptor=os.open(stage/'manifest.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            json.dump(manifest,stream,allow_nan=False,sort_keys=True,separators=(',',':'))
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            raise FileExistsError('Package destination must remain new')
        os.rename(stage,destination)
    except BaseException:
        _remove_package_stage(stage,destination)
        raise

def restore_package(source:Path,destination:Path):
    from .private_files import digest_file
    source=Path(source).absolute()
    if source.is_symlink() or getattr(source,'is_junction',lambda:False)():
        raise ValueError('A package must not use symbolic links')
    source=source.resolve()
    destination,stage=_package_stage(destination)
    try:
        if source.is_symlink() or (source/'manifest.json').is_symlink():
            raise ValueError('A package must not use symbolic links')
        if (source/'manifest.json').stat().st_size > 16*1024*1024:
            raise ValueError('Package manifest exceeds its technical limit')
        manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest,dict) or set(manifest)!={'format','database','objects'} or manifest['format']!='kneeoai-private-package-v1' or not isinstance(manifest['objects'],list):
            raise ValueError('Unsupported package format')
        dbmeta=manifest['database']
        if not isinstance(dbmeta,dict) or set(dbmeta)!={'file','sha256','bytes'} or dbmeta['file']!='application.sqlite3':
            raise ValueError('Invalid package database')
        if not isinstance(dbmeta['sha256'],str) or not re.fullmatch(r'[a-f0-9]{64}',dbmeta['sha256']) or type(dbmeta['bytes'])is not int or dbmeta['bytes']<0:
            raise ValueError('Invalid package database fingerprint')
        database=source/'application.sqlite3'
        if database.is_symlink() or digest_file(database)!=(dbmeta['sha256'],dbmeta['bytes']):
            raise ValueError('Package database checksum mismatch')
        with closing(_readonly(database)) as connection:
            _verify_source(connection)
            version=connection.execute('PRAGMA user_version').fetchone()[0]
            expected={row[0]:(row[1],row[2]) for row in connection.execute('SELECT object_key,sha256,byte_length FROM artifacts')} if version>=2 else {}
        declared={}
        for item in manifest['objects']:
            if not isinstance(item,dict) or set(item)!={'key','sha256','bytes'} or not isinstance(item['key'],str) or not isinstance(item['sha256'],str) or not re.fullmatch(r'[a-f0-9]{64}',item['sha256']) or type(item['bytes'])is not int or item['bytes']<0 or item['key'] in declared:
                raise ValueError('Invalid package object manifest')
            declared[item['key']]=(item['sha256'],item['bytes'])
        if declared!=expected:
            raise ValueError('Package objects do not match the database snapshot')
        for key,(digest,size) in expected.items():
            original=_object_path(source/'private_objects',key)
            if digest_file(original)!=(digest,size):
                raise ValueError('Package object checksum mismatch')
            _copy_private_file(original,_object_path(stage/'private_objects',key))
        _copy_database(database,stage/'application.sqlite3',include_objects=True)
        with closing(sqlite3.connect(stage/'application.sqlite3')) as connection:
            connection.execute('DELETE FROM sessions')
            connection.execute('DELETE FROM login_attempts')
            connection.commit()
            _verify_source(connection)
        if destination.exists():
            raise FileExistsError('Restore destination must remain new')
        os.rename(stage,destination)
    except BaseException:
        _remove_package_stage(stage,destination)
        raise
