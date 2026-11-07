"""Loopback-only TWO-BUILD test harness. Never deploy this test entry point.

Version switching belongs exclusively to this throwaway browser-test server.
Neither production API nor the user's database has a switching endpoint.
"""
from pathlib import Path
import os
import secrets
import sys
import time
import uuid

BACKEND = Path(__file__).resolve().parents[3] / 'backend'
FRONTEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND/'tests'))

import uvicorn
from starlette.responses import JSONResponse
from knee_app.api import create_app
from knee_app.config import Settings
from knee_app.db import create_user
from fake_imaging import SyntheticProcessor


def test_server():
    port = int(os.environ.get('E2E_PWA_PORT','8021'))
    if not 1024 <= port <= 65535: raise ValueError('Invalid test port')
    directory = BACKEND/'runtime'/('pwa-test-'+uuid.uuid4().hex)
    directory.mkdir(parents=True)
    secret, processor = secrets.token_bytes(32), SyntheticProcessor()
    apps = {}
    for version in ('a','b'):
        apps[version] = create_app(Settings(database_path=directory/'cases.sqlite3', secret_key=secret,
            allowed_origins=(f'http://127.0.0.1:{port}',), environment='test', login_limit=100,
            frontend_dist_path=FRONTEND/'test-results'/'pwa-builds'/version), processor=processor)
    for email, name in [('browser-a@kneeoai.invalid','Médico A de pruebas'),('browser-b@kneeoai.invalid','Médico B de pruebas')]:
        create_user(apps['a'].state.db,email=email,password='BrowserTestsOnly-2026!',display_name=name,now=time.time())
    selected = 'a'

    async def switcher(scope,receive,send):
        nonlocal selected
        if scope['type']=='http' and scope['path'].startswith('/__pwa_test__/version/'):
            headers=dict(scope['headers'])
            version=scope['path'].removeprefix('/__pwa_test__/version/')
            if scope['method']!='POST' or headers.get(b'x-pwa-test-control')!=b'synthetic-only-not-production' or version not in apps:
                await JSONResponse({'test':'unavailable'},status_code=404,headers={'Cache-Control':'no-store'})(scope,receive,send)
                return
            selected=version
            await JSONResponse({'version':selected},headers={'Cache-Control':'no-store'})(scope,receive,send)
            return
        await apps[selected](scope,receive,send)
    return switcher,port


if __name__=='__main__':
    app,port=test_server()
    uvicorn.run(app,host='127.0.0.1',port=port,access_log=False)
