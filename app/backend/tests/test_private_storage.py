"""Filesystem/bundle boundary checks, using only synthetic private files."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import base64

import pytest

from knee_app.db import Database, backup_database, backup_package, restore_package
from knee_app.private_files import ApiBodyLimit, PrivateFiles
from knee_app.errors import ApiError


def database_with_object(tmp_path):
    database=Database(tmp_path/'source.sqlite3')
    database.initialize()
    root=tmp_path/'private_objects'
    key='objects/'+'a'*32+'/source.dcm'
    path=root/key
    path.parent.mkdir(parents=True)
    content=b'fictional private source; not a clinical DICOM'
    path.write_bytes(content)
    with database.transaction() as connection:
        connection.execute("INSERT INTO users VALUES('usr_fake','fake@example.invalid','Fictional','not-a-real-password-hash','doctor',1,'2026-01-01T00:00:00Z')")
        connection.execute("INSERT INTO evaluations(id,code,owner_id,age,sex,bmi,side,version,revision_id,created_at,updated_at) VALUES('ev_fake','FAKE-001','usr_fake',60,'F',NULL,'right',1,'rev_fake','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')")
        connection.execute("INSERT INTO evaluation_revisions VALUES('rev_fake','ev_fake','{}','2026-01-01T00:00:00Z')")
        connection.execute("INSERT INTO preparations(id,evaluation_id,source_revision_id,image_id,image_version,source_sha,side,source_key,status,revision_id,created_at,updated_at) VALUES('prep_fake','ev_fake','rev_fake','img_fake','iv_fake',?,'right',?,'failed','prv_fake','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')",(hashlib.sha256(content).hexdigest(),key))
        connection.execute("INSERT INTO artifacts VALUES('art_fake','prep_fake','prv_fake','dicom',?,?,?)",(key,hashlib.sha256(content).hexdigest(),len(content)))
        connection.execute("INSERT INTO sessions VALUES(?, 'usr_fake',1,9999999999)",('b'*64,))
    return database,root,key,content


@pytest.mark.parametrize('key',[
    '../outside','/absolute','C:/outside','objects/../../outside',
    'objects\\'+'a'*32+'\\source.dcm','objects/'+'a'*32+'/../source.dcm',
    'objects/'+'a'*32+'/%2e%2e','objects/'+'a'*32+'/session.key',
])
def test_private_object_keys_cannot_escape_or_address_configuration(tmp_path,key):
    files=PrivateFiles(tmp_path/'private')
    with pytest.raises(ValueError):
        files.path(key)


def test_staging_cleanup_cannot_remove_objects_or_neighbor_directories(tmp_path):
    files=PrivateFiles(tmp_path/'private')
    existing=tmp_path/'keep'
    existing.mkdir()
    (existing/'sentinel').write_text('keep',encoding='utf-8')
    with pytest.raises(ValueError):
        files.discard(existing)
    with pytest.raises(ValueError):
        files.discard(files.staging)
    assert (existing/'sentinel').read_text(encoding='utf-8')=='keep'


def test_package_roundtrip_copies_exact_objects_and_revokes_sessions(tmp_path):
    database,root,key,content=database_with_object(tmp_path)
    bundle=tmp_path/'bundle'
    backup_package(database.path,bundle,root)
    assert (bundle/'private_objects'/key).read_bytes()==content
    restored=tmp_path/'restored'
    restore_package(bundle,restored)
    assert (restored/'private_objects'/key).read_bytes()==content
    with closing(sqlite3.connect(restored/'application.sqlite3')) as connection:
        assert connection.execute('SELECT COUNT(*) FROM users').fetchone()[0]==1
        assert connection.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0]==1
        assert connection.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]==0
    with pytest.raises(ValueError,match='package'):
        backup_database(database.path,tmp_path/'incomplete.sqlite3')
    assert not (tmp_path/'incomplete.sqlite3').exists()


@pytest.mark.parametrize('tamper',['missing','changed','traversal','extra_manifest'])
def test_bad_package_never_publishes_a_partial_restore(tmp_path,tamper):
    database,root,key,_=database_with_object(tmp_path)
    bundle=tmp_path/'bundle'
    backup_package(database.path,bundle,root)
    manifest_path=bundle/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if tamper=='missing':
        (bundle/'private_objects'/key).unlink()
    elif tamper=='changed':
        (bundle/'private_objects'/key).write_bytes(b'changed fictional content')
    elif tamper=='traversal':
        manifest['objects'][0]['key']='../../outside'
        manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    else:
        manifest['objects'].append({'key':'../../outside','sha256':'a'*64,'bytes':1})
        manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    destination=tmp_path/'rejected'
    with pytest.raises((ValueError,OSError)):
        restore_package(bundle,destination)
    assert not destination.exists()
    assert not list(tmp_path.glob('.knee-package-*'))


def test_backup_and_restore_package_preserve_existing_destination(tmp_path):
    database,root,_,_=database_with_object(tmp_path)
    bundle=tmp_path/'bundle'
    backup_package(database.path,bundle,root)
    target=tmp_path/'existing'
    target.mkdir()
    sentinel=target/'keep'
    sentinel.write_bytes(b'keep')
    for operation in [lambda:backup_package(database.path,target,root),lambda:restore_package(bundle,target)]:
        with pytest.raises(FileExistsError):
            operation()
        assert sentinel.read_bytes()==b'keep'


def test_corrupt_private_object_prevents_backup_publication(tmp_path):
    database,root,key,_=database_with_object(tmp_path)
    (root/key).write_bytes(b'mismatched fictional data')
    destination=tmp_path/'bad-backup'
    with pytest.raises(ValueError,match='match'):
        backup_package(database.path,destination,root)
    assert not destination.exists()
    assert not list(tmp_path.glob('.knee-package-*'))


def test_windows_junction_is_not_followed_by_private_storage(tmp_path):
    if os.name!='nt':
        pytest.skip('Windows junction boundary')
    import _winapi
    if not hasattr(_winapi,'CreateJunction'):
        pytest.skip('Junction API unavailable')
    files=PrivateFiles(tmp_path/'private')
    outside=tmp_path/'outside'
    outside.mkdir()
    (outside/'source.dcm').write_bytes(b'keep')
    link=files.objects/('b'*32)
    _winapi.CreateJunction(str(outside),str(link))
    with pytest.raises(ValueError):
        files.path('objects/'+'b'*32+'/source.dcm')
    assert (outside/'source.dcm').read_bytes()==b'keep'


def test_body_limit_counts_chunks_without_content_length():
    import asyncio
    async def scenario():
        messages=iter([{'type':'http.request','body':b'a'*6,'more_body':True},{'type':'http.request','body':b'b'*6,'more_body':False}])
        async def receive():
            return next(messages)
        async def send(message):
            raise AssertionError('Oversize body must not produce a success')
        async def endpoint(scope,receive,send):
            await receive()
            await receive()
        with pytest.raises(ApiError) as captured:
            await ApiBodyLimit(endpoint,10)({'type':'http','headers':[]},receive,send)
        assert captured.value.status==413
    asyncio.run(scenario())


def test_authenticated_view_rejects_a_changed_png_without_exposing_its_path(tmp_path):
    from fastapi.testclient import TestClient
    from knee_app.api import create_app
    from knee_app.config import Settings
    from knee_app.security import COOKIE_NAME,new_token,token_hash
    database,root,key,_=database_with_object(tmp_path)
    png=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQ0AAAAASUVORK5CYII=')
    image_key=key.replace('source.dcm','full_a.png')
    image_path=root/image_key
    image_path.write_bytes(png)
    token=new_token()
    with database.transaction() as connection:
        connection.execute("INSERT INTO artifacts VALUES('art_png','prep_fake','prv_fake','full_a',?,?,?)",(image_key,hashlib.sha256(png).hexdigest(),len(png)))
        connection.execute("INSERT INTO sessions VALUES(?,'usr_fake',?,?)",(token_hash(token),time.time(),time.time()+3600))
    application=create_app(Settings(database_path=database.path,private_files_path=root,secret_key=b'x'*32,environment='test',allowed_origins=('http://testserver',)))
    with TestClient(application,base_url='http://testserver') as client:
        client.cookies.set(COOKIE_NAME,token,path='/api/v1')
        route='/api/v1/preparations/prep_fake/views/full_a?revision=prv_fake'
        response=client.get(route)
        assert response.status_code==200 and response.content==png
        # A same-length mutation specifically exercises SHA, not only the size check.
        image_path.write_bytes(png[:-1]+bytes([png[-1]^1]))
        response=client.get(route)
        assert response.status_code==503
        assert response.json()['error']['code']=='PRIVATE_FILE_UNAVAILABLE'
        assert str(image_path) not in response.text


def test_pipeline_and_qc_version_changes_hide_effective_review_without_deleting_history(tmp_path):
    from fastapi.testclient import TestClient
    from knee_app.api import create_app
    from knee_app.config import Settings
    from knee_app.security import COOKIE_NAME,new_token,token_hash
    database,root,_,_=database_with_object(tmp_path)
    class Processor:
        pipeline_version='old-pipeline'
        qc_policy_version='old-policy'
        def is_available(self):
            return True
    processor=Processor()
    token=new_token()
    saved={'id':'qc_fake','qcPolicyVersion':'old-policy','preparationRevisionId':'prv_fake','decision':'accepted'}
    with database.transaction() as connection:
        connection.execute("UPDATE preparations SET status='ready',pipeline_version='old-pipeline',active_review_id='qc_fake' WHERE id='prep_fake'")
        connection.execute("INSERT INTO quality_reviews VALUES('qc_fake','prep_fake','prv_fake',?,'{}','usr_fake','2026-01-01T00:00:00Z')",(json.dumps(saved),))
        connection.execute("UPDATE evaluations SET preparation_id='prep_fake',review_id='qc_fake' WHERE id='ev_fake'")
        connection.execute("INSERT INTO sessions VALUES(?,'usr_fake',?,?)",(token_hash(token),time.time(),time.time()+3600))
    application=create_app(Settings(database_path=database.path,private_files_path=root,secret_key=b'x'*32,environment='test',allowed_origins=('http://testserver',)),processor=processor)
    with TestClient(application,base_url='http://testserver') as client:
        client.cookies.set(COOKIE_NAME,token,path='/api/v1')
        assert client.get('/api/v1/evaluations/ev_fake').json()['reviewId']=='qc_fake'
        processor.pipeline_version='new-pipeline'
        assert client.get('/api/v1/evaluations/ev_fake').json()['reviewId'] is None
        preparation=client.get('/api/v1/preparations/prep_fake').json()
        assert preparation['activeReview'] is None
        assert preparation['reviewContext']['canReview'] is False
        assert 'PIPELINE_VERSION_MISMATCH' in preparation['reviewContext']['blockerCodes']
        processor.pipeline_version='old-pipeline'
        processor.qc_policy_version='new-policy'
        assert client.get('/api/v1/evaluations/ev_fake').json()['reviewId'] is None
    with closing(database.connect()) as connection:
        assert connection.execute('SELECT COUNT(*) FROM quality_reviews').fetchone()[0]==1
        assert connection.execute("SELECT review_id FROM evaluations WHERE id='ev_fake'").fetchone()[0]=='qc_fake'


def test_expected_localizer_abstention_is_persisted_as_nonretryable_422(tmp_path):
    from knee_app.preparations import PreparationService
    database,root,_,_=database_with_object(tmp_path)
    service=PreparationService(database,PrivateFiles(root),None,time.time)
    with pytest.raises(ApiError) as failure:
        service.worker_failure('prep_fake',ApiError(409,'CROP_ABSTAIN','internal explanatory text'))
    assert failure.value.status==422 and failure.value.retryable is False
    with closing(database.connect()) as connection:
        prepared=service.get(connection,'prep_fake','usr_fake')
        stored=json.loads(prepared['error_json'])
    assert stored['code']=='CROP_ABSTAIN' and stored['retryable'] is False
    assert service.dto(prepared,'usr_fake')['error']['retryable'] is False
