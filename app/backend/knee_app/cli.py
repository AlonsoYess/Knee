from __future__ import annotations

import argparse
import os
from dataclasses import replace
from getpass import getpass
from pathlib import Path
import sqlite3
import sys

from .config import Settings
from .db import Database, backup_database, backup_package, create_user, disable_user, restore_database, restore_package, set_password

def prompted_password():
    password = getpass('Contraseña nueva (mínimo 12 caracteres): ')
    if password != getpass('Repite la contraseña: '):
        raise ValueError('Las contraseñas no coinciden')
    return password

def main(argv=None):
    parser = argparse.ArgumentParser(description='Administración local de KneeOAI; no carga datos científicos ni modelos.')
    parser.add_argument('--database', type=Path, help='Base de aplicación; por defecto KNEEOAI_DB_PATH o backend/runtime/application.sqlite3')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init', help='Inicializar o verificar el esquema, sin crear cuentas')
    create = commands.add_parser('create-user', help='Crear una cuenta con contraseña solicitada de forma privada')
    create.add_argument('--email', required=True)
    create.add_argument('--name', required=True)
    create.add_argument('--role', choices=['doctor','administrator'], default='doctor')
    reset = commands.add_parser('set-password', aliases=['reset-password'], help='Cambiar contraseña y revocar todas las sesiones de esa cuenta')
    reset.add_argument('--email', required=True)
    disable = commands.add_parser('disable-user', help='Desactivar cuenta y revocar sus sesiones')
    disable.add_argument('--email', required=True)
    backup = commands.add_parser('backup', help='Crear una copia coherente de la base mediante SQLite Backup API')
    backup.add_argument('--destination', type=Path, required=True)
    backup.add_argument('--package', action='store_true', help='Incluir base e imágenes privadas en una carpeta verificada')
    restore = commands.add_parser('restore', help='Restaurar en un archivo nuevo; el servidor debe estar detenido antes de cambiar su configuración')
    restore.add_argument('--source', type=Path, required=True)
    restore.add_argument('--destination', type=Path, required=True)
    restore.add_argument('--package', action='store_true', help='Restaurar un paquete de base e imágenes en una carpeta nueva')
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.database:
            settings = replace(settings, database_path=args.database, private_files_path=Path(os.environ['KNEEOAI_PRIVATE_FILES']) if os.environ.get('KNEEOAI_PRIVATE_FILES') else None)
        if args.command == 'restore':
            if args.package:
                restore_package(args.source,args.destination)
            else:
                restore_database(args.source, args.destination)
            print('Copia restaurada en el destino nuevo; las sesiones anteriores fueron revocadas. Cambia la configuración solo con el servicio detenido.')
            return 0
        if args.command == 'backup':
            if args.package:
                backup_package(settings.database_path,args.destination,settings.private_files_path)
            else:
                backup_database(settings.database_path, args.destination)
            print('Copia consistente creada y verificada. Aún debe comprobarse su restauración.')
            return 0
        db = Database(settings.database_path, settings.busy_timeout_ms)
        db.initialize()
        if args.command == 'init':
            print('Esquema de aplicación verificado. No se creó ninguna cuenta.')
        elif args.command == 'create-user':
            create_user(db, email=args.email, display_name=args.name, role=args.role, password=prompted_password())
            print('Cuenta creada. No se imprimió ni guardó su contraseña en un archivo de configuración.')
        elif args.command in {'set-password','reset-password'}:
            set_password(db, email=args.email, password=prompted_password())
            print('Contraseña actualizada y sesiones de la cuenta revocadas.')
        elif args.command == 'disable-user':
            disable_user(db, email=args.email)
            print('Cuenta desactivada y sesiones revocadas.')
        return 0
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f'No se completó la operación: {error}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
