"""HTTPS production-entrypoint rehearsal on the host, NOT a Docker/Linux validation.

Creates a NEW fictional database and a two-day private test CA. Does not install
certificates in the operating system or touch the user's live application.
"""
from pathlib import Path
from contextlib import closing
import argparse
import hashlib
import json
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
import urllib.error

APP = Path(__file__).resolve().parents[1]
BACKEND = APP/'backend'
SMOKE = Path(__file__).with_name('smoke_https.py')
PRIVATE_LOG = None


def run(arguments, *, env=None, expected=0):
    result=subprocess.run([str(value) for value in arguments],env=env,cwd=BACKEND,capture_output=True,text=True,encoding='utf-8',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode!=expected:
        if PRIVATE_LOG is not None:
            with PRIVATE_LOG.open('a',encoding='utf-8') as stream:
                stream.write(f'Exit status: {result.returncode}\n{result.stdout}\n{result.stderr}\n')
        raise RuntimeError('A rehearsal command did not finish as expected; consult its private report.')
    return result


def certificates(openssl,directory):
    ca_key,ca_cert=directory/'ca.key',directory/'ca.pem'
    key,csr,cert=directory/'localhost.key',directory/'localhost.csr',directory/'localhost.pem'
    ca_config=directory/'ca.cnf'
    ca_config.write_text('[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ca\n[dn]\nCN=KneeOAI isolated rehearsal CA\n[ca]\nbasicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n',encoding='ascii')
    extensions=directory/'localhost.ext'
    extensions.write_text('basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost,IP:127.0.0.1\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n',encoding='ascii')
    run([openssl,'req','-x509','-newkey','rsa:2048','-nodes','-keyout',ca_key,'-out',ca_cert,'-days','2',
         '-config',ca_config])
    run([openssl,'req','-newkey','rsa:2048','-nodes','-keyout',key,'-out',csr,'-subj','/CN=localhost','-config',ca_config])
    run([openssl,'x509','-req','-in',csr,'-CA',ca_cert,'-CAkey',ca_key,'-CAcreateserial','-out',cert,'-days','2','-sha256','-extfile',extensions])
    run([openssl,'verify','-CAfile',ca_cert,'-purpose','sslserver','-verify_ip','127.0.0.1',cert])
    for path in (ca_key,key):
        path.chmod(0o600)
    return ca_cert,key,cert


def stop(process):
    if process.poll() is not None: return
    if os.name=='nt':
        # Only the process tree created by this test is targeted. No name/port-based kill.
        subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    else:
        process.terminate()
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait(timeout=5)


def serve(env,port,key,cert,ca,logs):
    stream=logs.open('ab')
    process=subprocess.Popen([sys.executable,'-m','uvicorn','knee_app.deploy:create_app','--factory','--host','127.0.0.1',
        '--port',str(port),'--workers','1','--no-proxy-headers','--no-access-log','--ssl-keyfile',str(key),'--ssl-certfile',str(cert)],
        env=env,cwd=BACKEND,stdout=stream,stderr=stream,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    stream.close()
    try:
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cafile=str(ca))
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=context))
        last_error=None
        for _ in range(80):
            if process.poll() is not None: raise RuntimeError('The isolated application failed to start.')
            try:
                with opener.open(f'https://127.0.0.1:{port}/api/v1/health',timeout=1) as response:
                    if response.status==200: return process
            except (OSError,urllib.error.URLError) as error:
                last_error=error
                time.sleep(.15)
        if PRIVATE_LOG is not None:
            with PRIVATE_LOG.open('a',encoding='utf-8') as stream: stream.write(f'Readiness: {last_error}\n')
        raise RuntimeError('The isolated application did not become ready.')
    except BaseException:
        stop(process)
        raise


def main():
    global PRIVATE_LOG
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--frontend-dist',required=True,type=Path)
    parser.add_argument('--openssl',default=shutil.which('openssl'))
    parser.add_argument('--port',type=int,default=18443)
    args=parser.parse_args()
    if not args.openssl or not 18000<=args.port<=18999: parser.error('Use OpenSSL and a dedicated port between 18000 and 18999.')
    if args.output.exists(): parser.error('The rehearsal output directory must be new.')
    with closing(socket.socket()) as probe: probe.bind(('127.0.0.1',args.port))
    output=args.output.absolute()
    output.mkdir(parents=True,mode=0o700)
    PRIVATE_LOG=output/'commands.log'
    original=output/'data'
    original.mkdir(mode=0o700)
    ca,key,cert=certificates(args.openssl,output)
    secret=output/'session.key'
    secret.write_text(secrets.token_hex(32),encoding='ascii'); secret.chmod(0o600)
    env={key:value for key,value in os.environ.items() if not key.startswith(('KNEEOAI_','UVICORN_')) and key not in {'WEB_CONCURRENCY','FORWARDED_ALLOW_IPS'}}
    env.update(KNEEOAI_ENV='production',KNEEOAI_DATA_DIR=str(original),KNEEOAI_FRONTEND_DIST=str(args.frontend_dist.absolute()),
        KNEEOAI_ALLOWED_ORIGINS=f'https://127.0.0.1:{args.port}',KNEEOAI_SECRET_KEY_FILE=str(secret),PORT=str(args.port),
        PYTHONPATH=str(BACKEND),PYTHONUTF8='1')
    summary={'scope':'host_https_production_entrypoint_only','dockerImageBuilt':False,'containerExecutionVerified':False,
             'osCertificateStoreChanged':False,'syntheticOnly':True,'passed':False,'checks':{}}
    process=None
    try:
        run([sys.executable,'-m','knee_app.deploy','--check'],env=env)
        summary['checks']['strictProductionPreflight']=True
        run([sys.executable,'-c',"import time;from knee_app.config import Settings;from knee_app.db import Database,create_user;s=Settings.from_env();d=Database(s.database_path);d.initialize();create_user(d,email='browser-a@kneeoai.invalid',password='BrowserTestsOnly-2026!',display_name='Fictional deployment rehearsal',now=time.time())"],env=env)
        state,cookies=output/'state.json',output/'test-cookies.json'
        for mode in ('seed','resume'):
            process=serve(env,args.port,key,cert,ca,output/'server.log')
            run([sys.executable,SMOKE,'--mode',mode,'--base-url',f'https://127.0.0.1:{args.port}','--ca-file',ca,
                 '--state-file',state,'--cookie-file',cookies,'--report-file',output/f'{mode}.json'])
            summary['checks'][mode]=True
            stop(process); process=None
        backup=output/'backup'
        run([sys.executable,'-m','knee_app.cli','backup','--package','--destination',backup],env=env)
        manifest=json.loads((backup/'manifest.json').read_text(encoding='utf-8'))
        for item in manifest['objects']:
            data=(backup/'private_objects'/item['key']).read_bytes()
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise RuntimeError('Backup object failed integrity verification.')
        if not manifest['objects']: raise RuntimeError('The rehearsal must include a referenced synthetic file.')
        fixture_digest=json.loads(state.read_text(encoding='utf-8'))['fixtureSha256']
        if not any(item['sha256']==fixture_digest for item in manifest['objects']):
            raise RuntimeError('The original generated DICOM is missing from the complete backup.')
        summary['checks']['completeBackupWithVerifiedObjects']=True
        restored=output/'restored'
        run([sys.executable,'-m','knee_app.cli','restore','--package','--source',backup,'--destination',restored],env=env)
        for item in manifest['objects']:
            data=(restored/'private_objects'/item['key']).read_bytes()
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise RuntimeError('Restored object bytes differ from the verified backup.')
        env['KNEEOAI_DATA_DIR']=str(restored)
        process=serve(env,args.port,key,cert,ca,output/'server.log')
        run([sys.executable,SMOKE,'--mode','restored','--base-url',f'https://127.0.0.1:{args.port}','--ca-file',ca,
             '--state-file',state,'--cookie-file',cookies,'--report-file',output/'restored.json'])
        summary['checks']['restoredAndOldSessionsRevoked']=True
        summary['passed']=True
    except Exception as error:
        summary['error']=str(error)
    finally:
        if process is not None: stop(process)
        (output/'host-rehearsal.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary))
    return 0 if summary['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
