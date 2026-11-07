# Backend de aplicación KneeOAI — fase 3

FastAPI y SQLite conservan usuarios, sesiones, datos basales, DICOM, referencias visuales, recortes y decisiones de calidad. La predicción de progresión sigue bloqueada (`not_configured`; ejecución 503). No se abre la prueba reservada.

## Entorno y configuración

Usar Python 3.12.14 con `.venv` exclusivo de aplicación. Instalar `requirements.lock` para ejecución o `requirements-dev.lock` para pruebas. No instalar esos paquetes en entornos científicos. Inicializar con `python -m knee_app.cli init`; crear cuenta con `create-user --email medico@example.invalid --name "Médico"`. La contraseña se solicita privadamente.

Arranque: `.venv/Scripts/python.exe -m uvicorn knee_app.api:create_app --factory --host 127.0.0.1 --port 8000`. Linux utiliza `.venv/bin/python`.

| Variable | Uso |
| --- | --- |
| `KNEEOAI_DB_PATH` | SQLite; defecto `runtime/application.sqlite3`. |
| `KNEEOAI_PRIVATE_FILES` | Almacén privado; defecto `private_objects` junto a la base. |
| `KNEEOAI_SECRET_KEY` | Secreto de al menos 32 bytes; ausente en desarrollo se genera y conserva privadamente. |
| `KNEEOAI_ENV` | `development`, `test` o `production`. |
| `KNEEOAI_ALLOWED_ORIGINS` | Orígenes exactos separados por comas. |
| `KNEEOAI_FROZEN_PYTHON` | Intérprete del entorno científico aprobado existente. |
| `KNEEOAI_FROZEN_CHECKOUT` | Checkout científico con sus registros y activos privados congelados. |

Si falta configuración científica, acceso y borradores funcionan pero la preparación queda bloqueada. `app/IniciarServidor.ps1` carga opcionalmente ubicaciones desde el archivo ignorado `runtime/frozen_processor.json`, claves `python` y `checkout`. Las rutas reales no pertenecen a Git. Desarrollo admite solo loopback; producción exige HTTPS y cookies Secure. Una sola instancia con disco privado persistente; no sincronizar la base activa por Drive. PWA y despliegue son fases posteriores.

## Preparación y calidad

1. Guardar datos basales y lado.
2. Cargar un único DICOM original: máximo 60 MiB por archivo y 64 MiB por solicitud, contando bytes reales. Son límites de recursos.
3. Inspeccionar original bilateral y mitad en ventanas 1–99 y 5–95. Sellar centro visual, inspección, cegamiento y nota. La referencia es inmutable.
4. Solo tras sellar una referencia resuelta, ejecutar localizador y recorte aprobados. El centro nunca guía ni altera la geometría.
5. Registrar observación completa. El QC congelado decide si permite aceptar. Rechazar administrativamente requiere motivo y puede carecer de observación científica; nunca acredita aceptación.

`processor.py` comunica JSON con un proceso separado. FastAPI no importa `src/knee`, PyTorch ni pesos. `frozen_worker.py` comprueba componentes/dependencias y reutiliza lectura, espaciado, separación, localización, recorte y QC aprobados. Red bloqueada, sin descargas ni entrenamiento; escrituras limitadas al almacén y temporales privados. Este localizador prepara la imagen, no predice progresión.

Se conservan DICOM y matrices nativas. PNG autenticados, ligados a revisión, sirven solo para inspección. No hay descarga pública de DICOM ni corrección manual del recorte. Una abstención o fallo no genera aprobación.

## Sesiones y conservación

Cookie opaca HttpOnly/SameSite=Lax (Secure en producción), hash en servidor, vencimiento, rotación sesión/CSRF, límite de intentos y revocación al salir o cambiar contraseña. Toda mutación verifica usuario, origen y CSRF. Recurso ajeno/inexistente: 404. Respuestas `no-store`, errores uniformes y `X-Request-Id`; sin credenciales, cabeceras DICOM ni rutas privadas.

`If-Match` evita sobrescrituras. Carga, referencia, reintento y QC exigen `Idempotency-Key`: repetir recupera resultado sin duplicar ni reactivar historia. Una selección nueva y explícita de los mismos bytes puede reutilizar preparación/referencia, dejando historial. Editar edad, sexo o IMC conserva QC radiográfico; cambiar lado o imagen invalida la asociación activa.

SQLite `user_version=2` migra la versión 1 compatible conservando usuarios/revisiones. Esquemas desconocidos o incompletos se rechazan. Archivos inmutables se publican antes de guardar referencias transaccionales. Una interrupción queda registrada y admite reintento con la fuente/referencia ya guardadas. El inicio supone una sola instancia, sin coordinación entre varios servidores.

## Administración y respaldo

Desde `app/backend`:

```powershell
.venv/Scripts/python.exe -m knee_app.cli set-password --email medico@example.invalid
.venv/Scripts/python.exe -m knee_app.cli disable-user --email medico@example.invalid
.venv/Scripts/python.exe -m knee_app.cli backup --package --destination runtime/backups/copia-001
.venv/Scripts/python.exe -m knee_app.cli restore --package --source runtime/backups/copia-001 --destination runtime/restaurada-001
```

El paquete incluye snapshot consistente de SQLite, todos sus archivos y manifiesto de tamaños/SHA-256. Se verifica al copiar/restaurar. Restaurar solo escribe en destino nuevo y revoca sesiones copiadas. Detener servicio antes de cambiar configuración a `application.sqlite3` y `private_objects` restaurados. Conservar copia independiente y probar restauración. Se rechaza respaldo de solo SQLite cuando ya hay imágenes.

`runtime/`, archivos privados, claves y entornos quedan fuera de Git/exportaciones. Pruebas con procesador ficticio inyectado exclusivamente por harnesses, sin opción de producción. No acreditan precisión clínica: paridad del puente científico se comprueba aparte con entorno congelado y píxeles sintéticos.
