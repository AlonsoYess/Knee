# Preparación del despliegue — aplicación 0.12.0

Esta entrega prepara la aplicación para un servidor con HTTPS y almacenamiento persistente. No crea servicios externos, no compra dominio, no publica la web y no incluye datos o activos científicos. La ejecución Linux/Docker está pendiente: en el equipo de ensayo la virtualización está desactivada. El ensayo HTTPS descrito abajo se ejecutó en Windows y no sustituye esa comprobación.

## Archivos y límites

- `../Dockerfile`: compilación Vue con Node 22.17.0 y ejecución Python 3.12.14, imágenes oficiales fijadas por digest; runtime sin Node ni dependencias de pruebas. Proceso con usuario 10001 y un worker.
- `../.dockerignore` y `build_context.py`: lista cerrada de código público y recursos de interfaz. Excluyen runtime, claves, bases, DICOM, matrices, pesos, notebooks y entornos científicos.
- `compose.yml`: aplicación sin puerto público, volumen de datos explícito y secreto externo. Espera un ingreso HTTPS del proveedor; no lo contrata.
- `compose.rehearsal.yml` y `Caddyfile.rehearsal`: proxy únicamente para el ensayo local. Publica solo `127.0.0.1:18443`, usando un certificado de prueba y puerto interno 8443. Receta declarada validada por Compose; proxy no ejecutado todavía.
- `production.env.example`: nombres y rutas de ejemplo; no contiene una clave ni una dirección contratada.
- `rehearse_host.py` y `smoke_https.py`: ensayo reproducible con base nueva, cuenta ficticia y DICOM generado en memoria. No leen imágenes o evaluaciones del investigador.

La imagen de esta etapa incluye acceso, guardado, interfaz y comprobación operativa. No contiene el checkout/entorno científico ni pesos: su entrada estricta rechaza variables `KNEEOAI_FROZEN_*`. Preparación radiográfica y predicción quedan bloqueadas expresamente en este paquete. La copia local aprobada conserva su configuración propia. No se usa un modelo ficticio como sustituto; integrar el paquete científico corresponde a la fase 6, con runtime Linux compatible y recursos medidos.

## Configuración obligatoria

| Variable | Valor o responsabilidad |
| --- | --- |
| `KNEEOAI_ENV` | `production`, explícito. |
| `KNEEOAI_DATA_DIR` | Directorio absoluto y persistente; imagen: `/var/lib/kneeoai`. |
| `KNEEOAI_FRONTEND_DIST` | Distribución compilada; imagen: `/opt/kneeoai/frontend`. |
| `KNEEOAI_ALLOWED_ORIGINS` | Dirección HTTPS exacta, sin comodines ni rutas. |
| `KNEEOAI_SECRET_KEY_FILE` | Archivo privado con 64 caracteres hexadecimales ASCII; estable entre reinicios. |
| `KNEEOAI_SECRET_KEY` | Alternativa gestionada por el proveedor; no configurar a la vez que `_FILE`. |
| `KNEEOAI_FORWARDED_ALLOW_IPS` | Vacío inicialmente. Solo direcciones/redes de proxies conocidos; nunca `*` o toda Internet. |
| `PORT` | Puerto interno que facilite el proveedor; defecto 8080. |

`DATA_DIR` deriva `application.sqlite3` y `private_objects/`; las rutas adicionales, si se indican, deben coincidir. Se comprueban enlaces, lectura y escritura sin sobrescribir archivos existentes. `python -m knee_app.deploy --check` valida configuración y almacenamiento sin crear la base ni una clave. No puede demostrar por sí solo que el proveedor conserva el disco: eso requiere recreación y restauración reales.

No hay contraseña predeterminada ni creación automática de cuentas. La cuenta real se administra después con la CLI y contraseña solicitada de forma privada. La cuenta conocida de `smoke_https.py` existe únicamente en la base nueva del ensayo y no es una cuenta que deba añadirse a producción.

## Verificaciones disponibles ahora

Desde la raíz del repositorio, en Windows:

```powershell
app/backend/.venv/Scripts/python.exe -m pytest app/backend/tests -q
app/backend/.venv/Scripts/python.exe app/deploy/test_packaging.py
app/backend/.venv/Scripts/python.exe app/deploy/build_context.py --output RUTA_NUEVA/application-context.tar --report RUTA_NUEVA/context.json
```

El segundo comando ejecuta ocho comprobaciones sintéticas; su comprobación opcional con Docker aparece omitida cuando no hay motor. No contabilizarla como aprobada. El tercero crea exclusivamente un TAR nuevo con el contexto público y su huella; nunca sobrescribe el destino ni visita los directorios privados excluidos.

Ensayo HTTPS sin Docker, con OpenSSL disponible, desde la raíz:

```powershell
app/backend/.venv/Scripts/python.exe app/deploy/rehearse_host.py --output RUTA_PRIVADA_NUEVA --frontend-dist RUTA_ABSOLUTA/app/frontend/dist --openssl "C:/Program Files/Git/usr/bin/openssl.exe"
```

Compilar previamente la interfaz con `npm run build` en `app/frontend`. Linux usa `.venv/bin/python` y puede indicar `--openssl openssl`. El directorio de salida debe ser nuevo; puerto de ensayo predeterminado 18443, sin afectar el 8000 existente.

El ensayo crea una CA y un certificado localhost de dos días, comprobando cadena, propósito y nombre. Confía esa CA solo en sus clientes de prueba: no modifica el almacén de certificados del sistema ni desactiva su validación. Inicia y detiene únicamente sus propios procesos; hace acceso/guardado, reinicio con la misma clave y restauración en un directorio distinto. Guarda informes sin tokens ni contenido de casos. Su archivo de cookies separado contiene solo la sesión ficticia y permanece fuera de Git.

## Cuando esté habilitado Docker

Primero comprobar que `docker version` devuelve servidor Linux. La instalación Windows de Docker exige virtualización habilitada; no la cambia esta entrega. [Requisitos oficiales](https://docs.docker.com/desktop/setup/install/windows-install/#system-requirements).

Después, desde la raíz:

```powershell
$env:KNEEOAI_TEST_DOCKER_CONTEXT='1'
app/backend/.venv/Scripts/python.exe app/deploy/test_packaging.py
docker build --tag kneeoai-app:0.12.0-preparation app
```

Para construir desde el TAR auditado en Linux/Bash o CI puede utilizarse `docker build --tag kneeoai-app:0.12.0-preparation - < application-context.tar`. No pasar ese TAR por una tubería de texto de PowerShell. No subir la imagen a un registro sin revisar su contenido y destino.

Antes de ejecutar Compose hay que elegir una etiqueta de imagen ya comprobada y un nombre de proyecto nuevo, preparar un archivo de secreto privado y establecer los valores del ejemplo. Para el ensayo usar un origen local HTTPS y las rutas de certificado/clave generadas por el helper. Validar primero con `docker compose -f app/deploy/compose.yml -f app/deploy/compose.rehearsal.yml config --quiet`; este comando no prueba ejecución ni crea contenedores. El proxy no es necesario cuando el proveedor gestiona HTTPS.

Quedan por ejecutar: construcción Linux, inspección de capas y filesystem, usuario/permisos del volumen, arranque HTTPS por proxy, recreación con sesiones/datos conservados, respaldo/restauración en otro volumen y recursos máximos de la imagen. Los artefactos binarios Linux del lock se pudieron descargar; eso no acredita instalación, importación o ejecución dentro de la imagen.

## Persistencia, respaldo y retorno

Usar una sola instancia sobre el volumen. Persistir el directorio completo de datos, incluyendo archivos SQLite WAL/SHM; montar solo el `.sqlite3` no basta. El secreto debe persistir separadamente y no cambiar en cada inicio. No sincronizar el archivo activo por Drive.

El respaldo utiliza `python -m knee_app.cli backup --package --destination DESTINO_NUEVO`, con la configuración del servicio. Incluye snapshot SQLite, todos los objetos referenciados y manifiesto SHA/tamaños. Copiar luego el paquete fuera del almacenamiento del servicio y verificar su recuperación. Una copia en el mismo volumen no es un respaldo externo. La clave, los certificados y la configuración se gestionan separadamente; el paquete no los incluye.

Para restaurar: detener el servicio que vaya a utilizar el destino, ejecutar `python -m knee_app.cli restore --package --source PAQUETE --destination DIRECTORIO_NUEVO`, verificar archivos/configuración y adoptar esa carpeta como nuevo `DATA_DIR`. No sobrescribir una base abierta. Las sesiones copiadas se revocan expresamente: hay que volver a iniciar sesión, aunque el secreto sea el mismo.

Para volver a una versión anterior conservar conjuntamente su imagen/digest, configuración y respaldo de datos compatible. No bajar manualmente `PRAGMA user_version`; no hay un downgrade implementado. Si una migración futura no es compatible hacia atrás, restaurar el respaldo anterior en otro volumen implica volver a su punto de recuperación. El rollback con imágenes sigue pendiente del ensayo Docker.

## Publicación y dominio posteriores

Antes de contratar: presentar proveedor, dirección provisional disponible, costo mensual total, disco, copia externa y recursos medidos, dentro del límite acordado de US$15. Render sigue siendo candidato. Un disco persistente requiere servicio compatible de pago; no usar almacenamiento efímero para SQLite o imágenes. [Discos de Render](https://render.com/docs/disks).

La dirección gratuita del proveedor puede utilizarse antes de comprar el dominio. Al añadir después el dominio, conservar el servicio/disco y actualizar DNS, HTTPS y orígenes permitidos. Las cuentas y evaluaciones del servidor permanecen; el navegador requerirá acceso e instalación PWA desde el nuevo origen. [Dominios de Render](https://render.com/docs/custom-domains).

No hay sincronización entre las bases local y publicada. No copiar la base real, originales OAI, cohortes o pesos a un alojamiento como parte de este ensayo. La integración y autorización de activos científicos y su capacidad dentro del presupuesto tienen una verificación propia.
