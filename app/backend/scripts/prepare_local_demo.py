"""Create an explicitly requested local demonstration account with a private random password."""
from pathlib import Path
from contextlib import closing
import os
import secrets
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from knee_app.config import Settings
from knee_app.db import Database, create_user

def main():
    settings = Settings.from_env()
    target = settings.database_path.parent / "ACCESO_LOCAL.txt"
    database = Database(settings.database_path, settings.busy_timeout_ms)
    database.initialize()
    email = "medico@kneeoai.invalid"
    with closing(database.connect()) as connection:
        existing = connection.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if existing is not None:
        print("La cuenta local ya existe; se conservó su contraseña. Usa el acceso que guardaste o knee_app.cli set-password para restablecerlo.")
        return
    if target.exists():
        raise ValueError("Existe un archivo de acceso sin su cuenta. Se conservó intacto; revisa la instalación antes de continuar.")
    password = secrets.token_urlsafe(18)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never replaces an existing credential file.
    content = "KneeOAI — acceso local para casos ficticios\n\n"
    content += "Dirección: http://127.0.0.1:5173/acceso\n"
    content += f"Correo: {email}\nContraseña: {password}\n\n"
    content += "Archivo privado. No se publica en GitHub. Puedes eliminarlo después de conservar tu contraseña de forma segura.\n"
    created = False
    try:
        with target.open("x", encoding="utf-8") as stream:
            created = True
            try:
                os.chmod(target, 0o600)
            except OSError:
                pass
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    try:
        create_user(database, email=email, password=password, display_name="Médico de demostración", now=time.time())
    except BaseException:
        # The credential file precedes account creation; no account is committed before a durable copy exists.
        target.unlink(missing_ok=True)
        raise
    print("Cuenta local creada. Consulta runtime/ACCESO_LOCAL.txt; la contraseña no se imprime en la consola.")

if __name__ == "__main__":
    main()
