"""Public release delivery only. Every file, identity and credential is synthetic."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import pytest
from fastapi.testclient import TestClient

from knee_app.api import create_app
from knee_app.config import Settings
from knee_app.db import create_user


HTML=b'<!doctype html><html><head><title>Fictional shell</title></head><body><div id="app"></div><script src="/assets/app-AaBb0011.js"></script></body></html>'
JS=b'globalThis.publicFixture=true;'
PRIVATE=b'PRIVATE_PLACEHOLDER_NEVER_SERVE'


@pytest.fixture
def public_build(tmp_path):
    root=tmp_path/'dist'
    files={
        'index.html':HTML,'sw.js':b'/* fictional service worker */',
        'manifest.webmanifest':b'{"name":"Fictional prototype","start_url":"/acceso"}',
        'pwa-version.json':b'{"buildId":"fixture-a","contentSha256":"fixture","appVersion":"test","staticassets":[]}',
        'assets/app-AaBb0011.js':JS,'assets/app-BbCc0022.css':b'body{color:#123}',
        'icons/kneeoai.svg':b'<svg xmlns="http://www.w3.org/2000/svg"/>',
        'icons/kneeoai-maskable.svg':b'<svg xmlns="http://www.w3.org/2000/svg"/>',
        'icons/kneeoai-192.png':b'fictional-png-bytes','icons/kneeoai-512.png':b'fictional-png-bytes',
        '.env':PRIVATE,'session.key':PRIVATE,'application.sqlite3':PRIVATE,
        'private/source.dcm':PRIVATE,'api/v1/private.json':PRIVATE,
        'assets/plain.js':PRIVATE,'assets/app-AaBb0011.js.map':PRIVATE,
        'assets/records-AaBb0011.json':PRIVATE,'assets/image-AaBb0011.dcm':PRIVATE,
        'demo/radiografia-demo-a.svg':PRIVATE,'icons/patient.png':PRIVATE,
        'knee_app/api.py':PRIVATE,
    }
    for relative,content in files.items():
        path=root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.fixture
def configured(tmp_path,public_build):
    return Settings(database_path=tmp_path/'runtime'/'application.sqlite3',private_files_path=tmp_path/'private',
        secret_key=b'f'*32,environment='test',allowed_origins=('http://testserver',),frontend_dist_path=public_build)


@pytest.fixture
def distributed(configured):
    app=create_app(configured)
    with TestClient(app,base_url='http://testserver',raise_server_exceptions=False) as client:
        yield app,client


@pytest.mark.parametrize('path',[
    '/','/acceso','/evaluaciones','/evaluaciones/','/evaluaciones/nueva',
    '/evaluaciones/ev_0123/editar','/evaluaciones/ev_0123/radiografia',
    '/evaluaciones/ev_0123/revision','/evaluaciones/ev_0123/resumen',
    '/evaluaciones/ev_0123/resumen/?search=a%2Fb',
])
def test_only_known_vue_navigation_gets_public_shell(distributed,path):
    _,client=distributed
    response=client.get(path)
    assert response.status_code==200
    assert response.content==HTML
    assert response.headers['content-type'].startswith('text/html')
    assert 'no-cache' in response.headers['cache-control']
    assert 'set-cookie' not in response.headers
    assert PRIVATE not in response.content


@pytest.mark.parametrize('path',[
    '/unknown','/evaluaciones/missing','/evaluaciones/nueva/extra',
    '/evaluaciones/ev_0123/unknown','/evaluaciones/ev_0123/resumen/extra',
    '/evaluaciones//resumen','/evaluaciones///',
    '/evaluaciones/'+'x'*129+'/resumen',
])
def test_unknown_vue_routes_do_not_get_spa_200(distributed,path):
    _,client=distributed
    response=client.get(path)
    assert response.status_code==404
    assert response.headers['content-type'].startswith('application/json')
    assert response.content!=HTML


@pytest.mark.parametrize('path',[
    '/.env','/session.key','/application.sqlite3','/private/source.dcm',
    '/assets/plain.js','/assets/app-AaBb0011.js.map','/assets/records-AaBb0011.json',
    '/assets/image-AaBb0011.dcm','/assets/','/assets/missing-AaBb0011.js',
    '/demo/radiografia-demo-a.svg','/icons/patient.png','/knee_app/api.py',
    '/assets/%2e%2e/index.html','/%2e%2e/.env','/%2561cceso',
    '/%61cceso','/%69ndex.html','/assets/%61pp-AaBb0011.js',
    '/assets/..%5c.env',
])
def test_traversal_and_unlisted_files_never_expose_contents_or_fallback(distributed,path):
    _,client=distributed
    response=client.get(path)
    assert response.status_code==404,response.text
    assert PRIVATE not in response.content and response.content!=HTML
    assert 'no-store' in response.headers['cache-control']


@pytest.mark.parametrize('path,mime',[
    ('/index.html','text/html'),('/sw.js','application/javascript'),
    ('/manifest.webmanifest','application/manifest+json'),('/pwa-version.json','application/json'),
    ('/icons/kneeoai.svg','image/svg+xml'),('/icons/kneeoai-maskable.svg','image/svg+xml'),
    ('/icons/kneeoai-192.png','image/png'),('/icons/kneeoai-512.png','image/png'),
])
def test_shell_worker_manifest_version_and_icons_have_explicit_mime_without_immutable_cache(distributed,path,mime):
    _,client=distributed
    response=client.get(path)
    assert response.status_code==200
    assert response.headers['content-type'].split(';')[0]==mime
    assert 'no-cache' in response.headers['cache-control']
    assert 'immutable' not in response.headers['cache-control']
    assert response.headers['x-content-type-options']=='nosniff'
    if path=='/sw.js':
        assert response.headers['service-worker-allowed']=='/'


@pytest.mark.parametrize('path,mime',[('/assets/app-AaBb0011.js','application/javascript'),('/assets/app-BbCc0022.css','text/css')])
def test_only_hashed_public_build_assets_have_long_immutable_cache(distributed,path,mime):
    _,client=distributed
    response=client.get(path)
    assert response.status_code==200
    assert response.headers['content-type'].split(';')[0]==mime
    assert response.headers['cache-control']=='public, max-age=31536000, immutable'
    assert response.headers['x-content-type-options']=='nosniff'
    assert 'set-cookie' not in response.headers
    conditional=client.get(path,headers={'If-None-Match':response.headers['etag']})
    assert conditional.status_code==304 and conditional.content==b''


def test_head_has_no_body_but_keeps_representation_length(distributed):
    _,client=distributed
    for path,expected in [('/acceso',HTML),('/assets/app-AaBb0011.js',JS)]:
        response=client.head(path)
        assert response.status_code==200 and response.content==b''
        assert response.headers['content-length']==str(len(expected))


@pytest.mark.parametrize('path',['/api','/api/v1/no-such-endpoint','/api/v2/no-such-endpoint','/api/v1/assets/app-AaBb0011.js','/api/v1/private.json'])
def test_api_404_is_json_no_store_and_never_spa(distributed,path):
    _,client=distributed
    response=client.get(path,headers={'Accept':'text/html'})
    assert response.status_code==404
    assert response.headers['content-type'].startswith('application/json')
    assert response.json()['error']['code']=='RESOURCE_NOT_FOUND'
    assert response.headers['cache-control']=='no-store'
    assert response.content!=HTML and PRIVATE not in response.content


def test_public_shell_and_authenticated_api_work_on_exactly_same_origin(distributed):
    app,client=distributed
    create_user(app.state.db,email='distribution@example.invalid',password='Fictional-Distribution-Password!',display_name='Fictional Doctor')
    public_before=client.get('/acceso')
    assert client.get('/api/v1/evaluations').status_code==401
    anonymous=client.get('/api/v1/session')
    assert anonymous.status_code==200 and anonymous.headers['cache-control']=='no-store'
    logged_in=client.post('/api/v1/session',headers={'Origin':'http://testserver','X-CSRF-Token':anonymous.json()['csrfToken']},json={'email':'distribution@example.invalid','password':'Fictional-Distribution-Password!'})
    assert logged_in.status_code==200 and logged_in.headers['cache-control']=='no-store'
    data=client.get('/api/v1/evaluations')
    assert data.status_code==200 and data.json()['items']==[]
    assert data.headers['cache-control']=='no-store'
    public_after=client.get('/acceso')
    assert public_before.content==public_after.content==HTML
    assert 'cookie' not in public_after.request.headers
    assert 'distribution@example.invalid' not in public_after.text
    assert client.post('/acceso',json={}).status_code==404


def test_development_does_not_enable_distribution_implicitly(configured):
    settings=replace(configured,frontend_dist_path=None)
    with TestClient(create_app(settings),base_url='http://testserver') as client:
        for path in ['/acceso','/index.html','/sw.js','/assets/app-AaBb0011.js']:
            assert client.get(path).status_code==404
        assert client.get('/api/v1/session').status_code==200


def test_frontend_distribution_environment_is_explicit_opt_in(monkeypatch,public_build):
    monkeypatch.delenv('KNEEOAI_FRONTEND_DIST',raising=False)
    assert Settings.from_env().frontend_dist_path is None
    monkeypatch.setenv('KNEEOAI_FRONTEND_DIST',str(public_build))
    assert Settings.from_env().frontend_dist_path==public_build.absolute()


@pytest.mark.parametrize('mode',['database_inside','private_inside','distribution_inside_private','missing_index'])
def test_invalid_or_private_build_configuration_fails_before_database_creation(configured,public_build,tmp_path,mode):
    if mode=='database_inside':
        settings=replace(configured,database_path=public_build/'application.sqlite3')
    elif mode=='private_inside':
        settings=replace(configured,private_files_path=public_build/'private')
    elif mode=='distribution_inside_private':
        settings=replace(configured,private_files_path=public_build.parent)
    else:
        (public_build/'index.html').unlink()
        settings=configured
    with pytest.raises(ValueError):
        create_app(settings)
    assert not configured.database_path.exists()


def make_directory_link(link,target):
    if os.name=='nt':
        import _winapi
        if not hasattr(_winapi,'CreateJunction'):
            pytest.skip('Windows junction API unavailable')
        _winapi.CreateJunction(str(target),str(link))
    else:
        link.symlink_to(target,target_is_directory=True)


def test_asset_directory_junction_or_symlink_is_never_traversed(configured,public_build,tmp_path):
    assets=public_build/'assets'
    assert assets.resolve().is_relative_to(tmp_path.resolve()) and assets.name=='assets'
    shutil.rmtree(assets)
    outside=tmp_path/'outside'
    outside.mkdir()
    (outside/'app-AaBb0011.js').write_bytes(PRIVATE)
    make_directory_link(assets,outside)
    with TestClient(create_app(configured),base_url='http://testserver') as client:
        response=client.get('/assets/app-AaBb0011.js')
        assert response.status_code==404 and PRIVATE not in response.content


def test_linked_distribution_root_is_rejected(configured,public_build,tmp_path):
    link=tmp_path/'linked-dist'
    make_directory_link(link,public_build)
    with pytest.raises(ValueError):
        create_app(replace(configured,frontend_dist_path=link))


def test_release_inventory_and_content_are_immutable_for_app_instance(distributed,public_build):
    _,client=distributed
    (public_build/'assets'/'new-DdEe0044.js').write_bytes(PRIVATE)
    (public_build/'assets'/'app-AaBb0011.js').write_bytes(b'different release content')
    assert client.get('/assets/new-DdEe0044.js').status_code==404
    assert client.get('/assets/app-AaBb0011.js').content==JS
