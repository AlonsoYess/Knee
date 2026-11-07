# Backend de aplicación KneeOAI — fase 2

Este servidor implementa acceso real y guardado de datos basales con **FastAPI y SQLite**. No importa src/knee, no descubre paquetes científicos, no carga imágenes ni modelos y no abre la prueba reservada. La preparación DICOM y el QC real corresponden a otra fase. El motor informa not_configured; solicitar predicción sobre un caso accesible devuelve 503 sin ejecutar nada.

## Entorno independiente

Usar Python 3.12 y las dependencias fijadas de esta carpeta. No instalar estos paquetes en los entornos de entrenamiento. Los archivos requirements.lock y requirements-dev.lock describen el entorno de ejecución y las pruebas; sus fuentes directas están en requirements.in y requirements-dev.in.

Desde app/backend, en PowerShell:

    python -m venv .venv
    .venv/Scripts/python.exe -m pip install -r requirements-dev.lock
    .venv/Scripts/python.exe -m knee_app.cli init
    .venv/Scripts/python.exe -m knee_app.cli create-user --email medico@example.invalid --name "Médico de demostración"

create-user solicita la contraseña en la terminal mediante getpass y la confirma. No acepta contraseña en argumentos, no la imprime, no crea usuarios por defecto y no incluye una contraseña pública. El correo del ejemplo es un identificador ficticio de cuenta; no se envía correo. La contraseña debe tener entre 12 y 1024 caracteres y se guarda como hash Argon2 mediante pwdlib.

Arranque local:

    .venv/Scripts/python.exe -m uvicorn knee_app.api:create_app --factory --host 127.0.0.1 --port 8000 --no-proxy-headers

La SPA debe usar su proxy /api hacia este servidor durante el desarrollo. El navegador conserva un solo origen. No se publica un servicio mediante estos comandos. El servidor local no es el entorno de entrenamiento.

## Configuración

Settings permite configuración explícita en pruebas; create_app(settings, clock=...) acepta un reloj inyectable. Al utilizar la fábrica sin argumentos se leen:

| Variable | Uso |
| --- | --- |
| KNEEOAI_DB_PATH | Ruta absoluta a la base de aplicación. Por defecto backend/runtime/application.sqlite3. |
| KNEEOAI_ENV | development, test o production; el valor predeterminado es development. |
| KNEEOAI_ALLOWED_ORIGINS | Orígenes exactos separados por coma, sin comodines ni rutas. |
| KNEEOAI_SECRET_KEY | Secreto privado de al menos 32 bytes UTF-8. Si no se configura, se genera un archivo session.key junto a la base. |

En desarrollo solo se admiten orígenes loopback; los valores predeterminados son localhost y 127.0.0.1, puertos 5173 y 8000. Production exige orígenes HTTPS y cookies Secure. No se reutiliza una excepción HTTP de desarrollo en publicación. El eventual proxy debe declarar sus direcciones de confianza; este servidor no toma X-Forwarded-For arbitrario como identidad de cliente.

runtime/, .venv/, archivos de base y secretos están excluidos de Git. La carpeta nueva de base utiliza permisos privados en POSIX y el archivo SQLite se limita a 0600; Windows requiere conservar permisos de la cuenta local sobre esa carpeta. No compartir ni sincronizar la base activa por Drive. Una ruta personalizada debe estar en almacenamiento privado persistente.

## Acceso y permisos

- GET /api/v1/session crea un contexto anónimo y devuelve CSRF; no concede acceso a casos.
- POST /api/v1/session autentica una cuenta administrada y rota sesión y CSRF.
- DELETE /api/v1/session revoca el registro del servidor y expira la cookie.
- La sesión usa identificadores aleatorios opacos: SQLite guarda solo su SHA-256. CSRF se deriva con HMAC y no se almacena en claro en la base.
- La cookie kneeoai_session es HttpOnly, SameSite=Lax y Secure en producción. Su ruta es /api/v1.
- Toda mutación comprueba sesión, Origin permitido y X-CSRF-Token antes de procesar el cuerpo. No basta manipular el estado de Vue.
- Las sesiones autenticadas caducan por defecto a las 8 horas; los contextos anónimos, a los 20 minutos. No hay renovación silenciosa ilimitada.
- Los fallos de acceso se limitan por cuenta y cliente directo: 5 intentos en 15 minutos por defecto. Al excederlos se devuelve 429 y Retry-After.
- Un usuario solo accede a sus evaluaciones. El rol administrator no obtiene acceso clínico transversal implícito. Un caso ajeno y uno inexistente devuelven el mismo 404.

La API entrega errores uniformes, X-Request-Id y Cache-Control: no-store. No imprime credenciales, datos basales ni rutas privadas en el registro de errores.

## Datos y concurrencia

El borrador acepta edad, sexo, IMC y lado, incluidos ausentes null. Las comprobaciones son de formato; no se añade un criterio clínico de exclusión. Código, propietario, revisión y fechas son asignados por servidor en UTC.

GET devuelve ETag fuerte. PATCH exige If-Match: 428 si falta; 412 si el caso cambió. La modificación y su revisión inmutable se guardan en una transacción BEGIN IMMEDIATE. Si falla una parte, se revierte el conjunto.

POST /evaluations admite Idempotency-Key opcional. La interfaz debe enviar una clave y conservarla si queda incierto si el servidor aceptó. Misma cuenta, clave y cuerpo clínico canónico devuelve 200 con el mismo caso y su representación actual, sin reaplicar datos. Otra carga con esa clave produce 409. Una creación nueva devuelve 201. La asociación y el caso se guardan en la misma transacción.

El listado utiliza cursores firmados ligados al usuario y orden estable de creación. No hay rutas de borrado de casos en esta fase. preparationId, reviewId y currentExecutionId siguen siendo null: la demostración visual de imagen y revisión no se convierte en QC científico ni se guarda como aceptación real.

## Esquema SQLite

Se usa PRAGMA user_version=1 para el esquema inicial de aplicación, con users, sessions, login_attempts, evaluations, evaluation_revisions y create_requests. La inicialización es idempotente para un esquema compatible. Se rechazan versiones desconocidas, tablas o columnas obligatorias ausentes y referencias inválidas; no se intenta convertir una base arbitraria.

Las conexiones activan foreign_keys y busy_timeout; el archivo usa WAL. Las revisiones y asociaciones idempotentes se guardan transaccionalmente. Las conexiones de consultas y copias se cierran de forma explícita.

## Administración y recuperación

Comandos desde app/backend:

    .venv/Scripts/python.exe -m knee_app.cli set-password --email medico@example.invalid
    .venv/Scripts/python.exe -m knee_app.cli disable-user --email medico@example.invalid
    .venv/Scripts/python.exe -m knee_app.cli backup --destination runtime/backups/application-001.sqlite3
    .venv/Scripts/python.exe -m knee_app.cli restore --source runtime/backups/application-001.sqlite3 --destination runtime/restored/application.sqlite3

set-password (también reset-password) solicita la nueva contraseña sin argumentos secretos y revoca las sesiones de esa cuenta. disable-user desactiva y revoca. No se implementa recuperación por correo.

La copia utiliza SQLite Backup API y comprueba esquema e integridad. No se copia a mano un archivo WAL activo. Backup y restore rechazan un destino existente; no sobrescriben datos. La restauración crea una base nueva y elimina sesiones e intentos de acceso para impedir reactivar credenciales de sesión antiguas. Conserva cuentas, evaluaciones, revisiones y asociaciones de creación.

Para adoptar una restauración, detener el servidor, verificar el destino y cambiar KNEEOAI_DB_PATH; después arrancar y comprobar los casos con una sesión nueva. No reemplazar archivos de una base abierta. Los backups contienen información de aplicación y hashes de cuenta: guardarlos en una ubicación privada fuera del repositorio y verificar una restauración. El secreto CSRF puede generarse de nuevo porque las sesiones se invalidan al restaurar.

En fase 2 no hay archivos clínicos que respaldar; cuando se añadan imágenes, base y archivos requerirán una copia coherente conjunta.

## Pruebas y estado

    .venv/Scripts/python.exe -m pytest tests

Las pruebas utilizan bases temporales y cuentas ficticias. Cubren sesión, CSRF, permisos, validación, versiones, transacciones, idempotencia y recuperación; no utilizan datos científicos. Su resultado debe registrarse al ejecutar la entrega, no inferirse de este documento.

El servidor no implementa todavía DICOM, QC científico, inferencia, PWA o despliegue. Las rutas de fases futuras permanecen ausentes, salvo el estado del motor y el rechazo explícito de predicción autenticada. La autenticación real de fase 2 no acredita utilidad clínica del predictor.
