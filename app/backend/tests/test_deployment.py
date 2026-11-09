"""Portable app-only deployment checks; no containers, external services or scientific data."""
from dataclasses import replace
from contextlib import closing
import os
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient

from knee_app import __version__
from knee_app import deploy
from knee_app.config import DEFAULT_RUNTIME,Settings
from knee_app.db import create_user


ORIGIN='https://deployment.example.invalid'
SECRET='a3'*32


@pytest.fixture
def production_env(tmp_path):
    data=tmp_path/'persistent'
    data.mkdir()
    frontend=tmp_path/'frontend'
    frontend.mkdir()
    (frontend/'index.html').write_bytes(b'<!doctype html><title>Fictional deployment shell</title>')
    secret=tmp_path/'mounted-secret'
    secret.write_bytes(SECRET.encode('ascii'))
    return {'KNEEOAI_ENV':'production','KNEEOAI_DATA_DIR':str(data),'KNEEOAI_FRONTEND_DIST':str(frontend),
            'KNEEOAI_ALLOWED_ORIGINS':ORIGIN,'KNEEOAI_SECRET_KEY_FILE':str(secret)}


def test_data_dir_derivation_and_file_secret_are_shared_with_existing_cli_settings(production_env):
    settings=Settings.from_env(production_env)
    data=Path(production_env['KNEEOAI_DATA_DIR'])
    assert settings.database_path==data/'application.sqlite3'
    assert settings.private_files_path==data/'private_objects'
    assert settings.secret_key==SECRET.encode('ascii')
    assert settings.secure_cookies is True
    assert SECRET not in repr(settings)
    assert not settings.database_path.exists()
    legacy=Settings.from_env({})
    assert legacy.environment=='development' and legacy.data_dir is None
    assert legacy.database_path==DEFAULT_RUNTIME/'application.sqlite3'


def test_app_only_deployment_rejects_local_predictor_configuration(production_env):
    production_env['KNEEOAI_PREDICTOR_BUNDLE'] = '/private/fictional-development-probe'
    with pytest.raises(ValueError, match='app-only'):
        deploy.deployment_settings(production_env)


@pytest.mark.parametrize('ending',[b'',b'\n',b'\r\n'])
def test_one_exact_terminal_line_ending_does_not_change_the_stable_secret(production_env,ending):
    Path(production_env['KNEEOAI_SECRET_KEY_FILE']).write_bytes(SECRET.encode('ascii')+ending)
    assert Settings.from_env(production_env).secret_key==SECRET.encode('ascii')
    assert Settings.from_env(production_env).secret_key==Settings.from_env(production_env).secret_key


@pytest.mark.parametrize('value',[b'',b'f'*32,b'f'*63,b'f'*65,b'g'*64,b'f'*32+b'\n'+b'f'*32,b'f'*64+b'\n\n',b' '+b'f'*64,b'f'*64+b' ',b'\xff'*64])
def test_bad_secret_file_is_rejected_without_printing_contents_or_using_another_key(production_env,value,capsys):
    Path(production_env['KNEEOAI_SECRET_KEY_FILE']).write_bytes(value)
    assert deploy.main(['--check'],environ=production_env)==2
    output=capsys.readouterr()
    assert SECRET not in output.out+output.err
    assert production_env['KNEEOAI_SECRET_KEY_FILE'] not in output.out+output.err
    assert not (Path(production_env['KNEEOAI_DATA_DIR'])/'session.key').exists()


def test_two_secret_sources_are_rejected_in_cli_and_deployment_configuration(production_env):
    environment={**production_env,'KNEEOAI_SECRET_KEY':'different-fictitious-env-secret-value'}
    with pytest.raises(ValueError):
        Settings.from_env(environment)
    with pytest.raises(ValueError):
        deploy.check_deployment(environment)


def test_explicit_environment_secret_and_file_containing_same_token_match(production_env):
    environment={key:value for key,value in production_env.items() if key!='KNEEOAI_SECRET_KEY_FILE'}
    environment['KNEEOAI_SECRET_KEY']=SECRET
    assert Settings.from_env(environment).secret_key==Settings.from_env(production_env).secret_key


@pytest.mark.parametrize('change',[
    {'KNEEOAI_ENV':'development'},{'KNEEOAI_ENV':''},{'KNEEOAI_DATA_DIR':''},{'KNEEOAI_FRONTEND_DIST':''},
    {'KNEEOAI_ALLOWED_ORIGINS':''},{'KNEEOAI_ALLOWED_ORIGINS':'http://deployment.example.invalid'},
    {'KNEEOAI_ALLOWED_ORIGINS':'https://*.example.invalid'},{'KNEEOAI_ALLOWED_ORIGINS':'https://deployment.example.invalid/private'},
    {'KNEEOAI_ALLOWED_ORIGINS':'https://name:password@deployment.example.invalid'},
    {'KNEEOAI_DATA_DIR':'relative-state'},{'KNEEOAI_FROZEN_PYTHON':''},{'KNEEOAI_FROZEN_CHECKOUT':'/unapproved'},
    {'PORT':'0'},{'PORT':'65536'},{'PORT':'not-a-port'},{'WEB_CONCURRENCY':'2'},
    {'UVICORN_WORKERS':'2'},{'FORWARDED_ALLOW_IPS':'*'},
])
def test_deployment_fails_closed_on_missing_insecure_or_shadow_settings(production_env,change):
    with pytest.raises((ValueError,OSError)):
        deploy.check_deployment({**production_env,**change})
    assert not (Path(production_env['KNEEOAI_DATA_DIR'])/'application.sqlite3').exists()


def test_absent_secret_and_weak_explicit_secret_do_not_generate_a_shadow_file(production_env):
    environment={key:value for key,value in production_env.items() if key!='KNEEOAI_SECRET_KEY_FILE'}
    for extra in [{},{'KNEEOAI_SECRET_KEY':'short'}]:
        with pytest.raises(ValueError):
            deploy.check_deployment({**environment,**extra})
    assert list(Path(production_env['KNEEOAI_DATA_DIR']).iterdir())==[]


@pytest.mark.parametrize('change',[
    {'KNEEOAI_DB_PATH':'outside'}, {'KNEEOAI_PRIVATE_FILES':'outside'},
    {'KNEEOAI_DB_PATH':'inside_other_db'}, {'KNEEOAI_PRIVATE_FILES':'inside_other_objects'},
    {'KNEEOAI_FRONTEND_DIST':'inside_public'},
])
def test_storage_and_public_paths_cannot_overlap_or_select_a_shadow_database(production_env,tmp_path,change):
    data=Path(production_env['KNEEOAI_DATA_DIR'])
    chosen={
        'outside':tmp_path/'outside','inside_other_db':data/'other.sqlite3',
        'inside_other_objects':data/'other_objects','inside_public':data/'public',
    }
    environment={**production_env,**{key:str(chosen[value]) for key,value in change.items()}}
    with pytest.raises(ValueError):
        deploy.check_deployment(environment)
    assert not (data/'application.sqlite3').exists()


def test_check_roundtrip_probe_is_temporary_and_does_not_initialize_database(production_env,capsys):
    assert deploy.main(['--check'],environ=production_env)==0
    data=Path(production_env['KNEEOAI_DATA_DIR'])
    assert list(data.iterdir())==[]
    output=capsys.readouterr()
    assert SECRET not in output.out+output.err
    assert str(data) not in output.out+output.err


def test_missing_or_unwritable_volume_stops_startup_without_alternate_storage(production_env,monkeypatch,tmp_path):
    missing=tmp_path/'missing-volume'
    with pytest.raises(ValueError):
        deploy.check_deployment({**production_env,'KNEEOAI_DATA_DIR':str(missing)})
    assert not missing.exists()
    original=os.open
    def denied(path,flags,*args,**kwargs):
        if Path(path).name.startswith('.kneeoai-ready-'):
            raise PermissionError('Fictional unreadable mount')
        return original(path,flags,*args,**kwargs)
    monkeypatch.setattr(deploy.os,'open',denied)
    assert deploy.main(['--check'],environ=production_env)==2
    assert not (Path(production_env['KNEEOAI_DATA_DIR'])/'application.sqlite3').exists()


def test_probe_collision_never_replaces_or_removes_an_existing_file(production_env,monkeypatch):
    data=Path(production_env['KNEEOAI_DATA_DIR'])
    existing=data/('.kneeoai-ready-'+'f'*32)
    existing.write_bytes(b'fictional existing file')
    monkeypatch.setattr(deploy.secrets,'token_hex',lambda size:'f'*32)
    with pytest.raises(FileExistsError):
        deploy.check_deployment(production_env)
    assert existing.read_bytes()==b'fictional existing file'


def test_existing_incompatible_database_is_rejected_without_reinitialization(production_env):
    database=Path(production_env['KNEEOAI_DATA_DIR'])/'application.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute('PRAGMA user_version=999')
    with pytest.raises(ValueError):
        deploy.check_deployment(production_env)
    with sqlite3.connect(database) as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0]==999
        assert connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]==0


def test_factory_restarts_with_same_secret_database_and_authenticated_session(production_env):
    first=deploy.create_app(production_env)
    create_user(first.state.db,email='deploy@example.invalid',password='Fictional-Deployment-Password!',display_name='Fictional Deployed Doctor')
    with TestClient(first,base_url=ORIGIN) as client:
        anonymous=client.get('/api/v1/session')
        logged_in=client.post('/api/v1/session',headers={'Origin':ORIGIN,'X-CSRF-Token':anonymous.json()['csrfToken']},json={'email':'deploy@example.invalid','password':'Fictional-Deployment-Password!'})
        assert logged_in.status_code==200 and 'Secure' in logged_in.headers['set-cookie']
        expected_csrf=logged_in.json()['csrfToken']
        second=deploy.create_app(production_env)
        with TestClient(second,base_url=ORIGIN) as restarted:
            restarted.cookies.update(client.cookies)
            session=restarted.get('/api/v1/session')
            assert session.status_code==200 and session.json()['authenticated'] is True
            assert session.json()['csrfToken']==expected_csrf
            assert restarted.get('/acceso').status_code==200
    assert not (Path(production_env['KNEEOAI_DATA_DIR'])/'session.key').exists()


def test_production_never_downgrades_secure_cookie_for_direct_http(production_env):
    application=deploy.create_app(production_env)
    create_user(application.state.db,email='deploy@example.invalid',password='Fictional-Deployment-Password!',display_name='Fictional')
    with TestClient(application,base_url=ORIGIN.replace('https:','http:')) as client:
        initial=client.get('/api/v1/session')
        assert 'Secure' in initial.headers['set-cookie']
        login=client.post('/api/v1/session',headers={'Origin':ORIGIN,'X-CSRF-Token':initial.json()['csrfToken']},json={'email':'deploy@example.invalid','password':'Fictional-Deployment-Password!'})
        assert login.status_code==401
        assert client.get('/api/v1/health').status_code==200


def test_public_health_reads_only_select_one_and_never_creates_session(production_env,monkeypatch):
    from knee_app import api as api_module
    application=deploy.create_app(production_env)
    statements=[]
    original=sqlite3.connect
    def recorded(*args,**kwargs):
        connection=original(*args,**kwargs)
        connection.set_trace_callback(statements.append)
        return connection
    monkeypatch.setattr(api_module.sqlite3,'connect',recorded)
    with TestClient(application,base_url=ORIGIN) as client:
        response=client.get('/api/v1/health')
        assert response.status_code==200 and response.json()=={'status':'ok','version':__version__}
        assert response.headers['cache-control']=='no-store' and 'set-cookie' not in response.headers
    assert [statement.strip().upper() for statement in statements]==['SELECT 1']
    with closing(original(application.state.db.path)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]==0


def test_public_health_missing_database_is_503_and_does_not_create_a_replacement(production_env):
    application=deploy.create_app(production_env)
    database=application.state.db.path
    relocated=database.with_name('temporarily-unavailable.sqlite3')
    database.rename(relocated)
    with TestClient(application,base_url=ORIGIN) as client:
        response=client.get('/api/v1/health')
    assert response.status_code==503 and response.json()['error']['code']=='STORAGE_UNAVAILABLE'
    assert response.headers['cache-control']=='no-store' and 'set-cookie' not in response.headers
    assert not database.exists() and relocated.exists()
    assert str(database) not in response.text and SECRET not in response.text


@pytest.mark.parametrize('value',['*','0.0.0.0/0','::/0','127.0.0.1,*','proxy.example.invalid','10.0.0.1/24'])
def test_proxy_trust_never_uses_wildcards_or_hostname_discovery(production_env,value):
    with pytest.raises(ValueError):
        deploy.deployment_settings({**production_env,'KNEEOAI_FORWARDED_ALLOW_IPS':value})


@pytest.mark.parametrize('proxy',[None,'127.0.0.1,10.20.0.0/24'])
def test_entrypoint_uses_one_worker_explicit_port_and_only_declared_proxy_trust(production_env,monkeypatch,proxy):
    captured={}
    def run(application,**kwargs):
        captured.update(kwargs)
        assert application.version==__version__
    monkeypatch.setattr(deploy.uvicorn,'run',run)
    environment={**production_env,'PORT':'8099'}
    if proxy is not None:
        environment['KNEEOAI_FORWARDED_ALLOW_IPS']=proxy
    assert deploy.main([],environ=environment)==0
    assert captured['host']=='0.0.0.0' and captured['port']==8099 and captured['workers']==1
    assert captured['reload'] is False and captured['access_log'] is False
    assert captured['proxy_headers'] is (proxy is not None)
    assert captured['forwarded_allow_ips']==(['127.0.0.1','10.20.0.0/24'] if proxy is not None else [])


def test_production_settings_cannot_use_generated_secret_or_implicit_runtime_paths(production_env):
    valid=Settings.from_env(production_env)
    with pytest.raises(ValueError):
        replace(valid,secret_key=None)
    with pytest.raises(ValueError):
        replace(valid,data_dir=None)
    with pytest.raises(ValueError):
        replace(valid,database_path=DEFAULT_RUNTIME/'application.sqlite3')
