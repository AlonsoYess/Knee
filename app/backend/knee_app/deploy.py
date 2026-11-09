"""Strict app-only deployment entrypoint. No environment discovery, training or model loading."""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import secrets
import sqlite3
import sys

import uvicorn

from .api import create_app as application_factory
from .config import Settings,production_path
from .db import verify_schema
from .frontend_distribution import FrontendDistribution


@dataclass(frozen=True)
class Deployment:
    settings:Settings
    port:int
    forwarded_allow_ips:tuple[str,...]


def deployment_settings(environ=None):
    source=dict(os.environ if environ is None else environ)
    if source.get('KNEEOAI_ENV')!='production':
        raise ValueError('Deployment requires explicit production configuration')
    if not source.get('KNEEOAI_DATA_DIR') or not source.get('KNEEOAI_FRONTEND_DIST') or not source.get('KNEEOAI_ALLOWED_ORIGINS'):
        raise ValueError('Deployment requires explicit data, frontend and HTTPS origin settings')
    if any(key.startswith(('KNEEOAI_FROZEN_', 'KNEEOAI_PREDICTOR_')) for key in source):
        raise ValueError('This deployment is app-only and cannot enable a scientific adapter')
    if any(key.startswith('UVICORN_') for key in source) or 'FORWARDED_ALLOW_IPS' in source:
        raise ValueError('Use the explicit deployment entrypoint settings')
    if source.get('WEB_CONCURRENCY','1')!='1':
        raise ValueError('This deployment uses exactly one worker')
    settings=Settings.from_env(source)
    data=production_path(settings.data_dir)
    if settings.database_path!=data/'application.sqlite3' or settings.private_files_path!=data/'private_objects':
        raise ValueError('Deployment database and object paths must use the declared data directory')
    secret_file=source.get('KNEEOAI_SECRET_KEY_FILE')
    if secret_file:
        path=production_path(secret_file)
        if path==settings.database_path or path.is_relative_to(settings.private_files_path):
            raise ValueError('Secret files must be separate from application database and objects')
    raw_port=source.get('PORT','8080')
    if not isinstance(raw_port,str) or not raw_port.isascii() or not raw_port.isdecimal() or not 1<=int(raw_port)<=65535:
        raise ValueError('Deployment requires a valid listener port')
    forwarded=[]
    raw_proxy=source.get('KNEEOAI_FORWARDED_ALLOW_IPS','')
    if raw_proxy:
        for value in raw_proxy.split(','):
            value=value.strip()
            if not value or value=='*':
                raise ValueError('Proxy trust requires explicit IP addresses or networks')
            try:
                network=ipaddress.ip_network(value,strict=True)
            except ValueError:
                raise ValueError('Proxy trust requires explicit IP addresses or networks') from None
            if network.prefixlen==0:
                raise ValueError('Proxy trust cannot cover every client')
            canonical=str(network) if '/' in value else str(network.network_address)
            if canonical not in forwarded:
                forwarded.append(canonical)
    return Deployment(settings,int(raw_port),tuple(forwarded))


def writable_directory(directory):
    directory=production_path(directory)
    if not directory.is_dir():
        raise ValueError('The declared persistent directory must already exist')
    sentinel=directory/('.kneeoai-ready-'+secrets.token_hex(16))
    descriptor=None
    created=False
    try:
        descriptor=os.open(sentinel,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600)
        created=True
        os.write(descriptor,b'kneeoai-storage-probe')
        os.fsync(descriptor)
        os.lseek(descriptor,0,os.SEEK_SET)
        if os.read(descriptor,64)!=b'kneeoai-storage-probe':
            raise OSError('Storage probe did not roundtrip')
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            sentinel.unlink()


def check_deployment(environ=None):
    plan=deployment_settings(environ)
    settings=plan.settings
    FrontendDistribution(settings.frontend_dist_path,database_path=settings.database_path,private_files_path=settings.private_files_path)
    writable_directory(settings.data_dir)
    if settings.private_files_path.exists():
        writable_directory(settings.private_files_path)
    if settings.database_path.exists():
        path=production_path(settings.database_path)
        if not path.is_file():
            raise ValueError('The application database path must be a regular file')
        descriptor=os.open(path,os.O_RDWR)
        os.close(descriptor)
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=1.0)) as connection:
            connection.execute('SELECT 1').fetchone()
            verify_schema(connection)
    return plan


def create_app(environ=None):
    """ASGI factory usable by deployment smoke checks; uses only explicit production settings."""
    plan=check_deployment(environ)
    return application_factory(plan.settings)


def main(argv=None,*,environ=None):
    parser=argparse.ArgumentParser(description='Validar e iniciar KneeOAI en producción con un único proceso; no habilita modelos científicos.')
    parser.add_argument('--check',action='store_true',help='Comprobar configuración y almacenamiento sin iniciar servidor ni inicializar la base')
    args=parser.parse_args(argv)
    try:
        plan=check_deployment(environ)
        if args.check:
            print('Configuración y almacenamiento verificados. No se inició el servidor.')
            return 0
        application=application_factory(plan.settings)
    except (ValueError,OSError,sqlite3.Error):
        print('No se pudo preparar el despliegue: revisa la configuración explícita y el almacenamiento persistente.',file=sys.stderr)
        return 2
    uvicorn.run(application,host='0.0.0.0',port=plan.port,workers=1,reload=False,
        proxy_headers=bool(plan.forwarded_allow_ips),forwarded_allow_ips=list(plan.forwarded_allow_ips),
        access_log=False,server_header=False)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
