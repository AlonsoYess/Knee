# Fase 5 — comprobación Docker local

Fecha: 9 de noviembre de 2026. Aplicación 0.12.0, API 0.5.0, esquema SQLite 2.

**Resultado:** construcción Linux, ejecución por proxy HTTPS, recreación del contenedor y recuperación desde un respaldo externo a Docker comprobadas con datos ficticios. Queda resuelta la comprobación local que impedía la virtualización desactivada. La publicación en Internet sigue pendiente; esta entrega no contrata alojamiento ni compra dominio.

## Evidencia ejecutada

| Comprobación | Resultado |
| --- | --- |
| Empaquetado | 9/9 correctas, incluida exportación real de un contexto Docker sintético; ninguna omitida. |
| Construcción | Correcta desde TAR auditado: 62 archivos públicos, 532480 bytes. |
| Imagen | Linux/amd64, Python 3.12.14; dependencias instaladas y `pip check` correcto. |
| Límite de la distribución | 10 capas inspeccionadas; sin bases, originales, pesos ni entorno científico en las capas de aplicación. Interfaz y PWA presentes. |
| Ejecución | Aplicación UID/GID 10001; proxy UID 10002. Sistemas raíz de solo lectura y capacidades retiradas; datos en volumen separado. |
| HTTPS | Certificado y nombre verificados con CA local explícita. Sin instalar certificados en Windows ni desactivar su validación. Puerto publicado solo en `127.0.0.1:18443`. |
| Acceso y persistencia | 11 comprobaciones iniciales, 6 después de recrear realmente el contenedor y 6 después de restaurar: 23 correctas. |
| Interfaz y recursos PWA | HTML y script compilado servidos; manifiesto y versión correctos; 8 recursos comparados por SHA256 y política de entrega del service worker comprobada. |
| Respaldo | Base y original DICOM generado incluidos y verificados por tamaño/SHA256. Paquete exportado al equipo, fuera de los volúmenes Docker. |
| Recuperación | Fuente exportada montada solo para lectura; restauración en otro volumen nuevo, sin depender del volumen original. Datos y bytes del original conservados; sesiones anteriores revocadas y nuevo acceso correcto. |
| Finalización | Contenedores y red del ensayo detenidos/eliminados. Volúmenes ficticios e informes conservados como evidencia; imagen local disponible. |

Imagen local: `kneeoai-app:0.12.0-verification-20261109`.

Identificador comprobado: `sha256:b379b2cb703dff566e6e29e8964822d4ff598a71cf54dd24e662a5b2be32385a`.

Contexto construido: `sha256:1c32201ee82bda7b50a8d4117383cd39bcc5f8804ec865efcc89dab332efff79`.

## Correcciones encontradas por las pruebas reales

1. Las excepciones de directorio de `.dockerignore` admitían tres archivos adicionales del árbol ficticio de prueba. Se retiraron esas aperturas, conservando las excepciones de archivos permitidos. La exportación Docker coincide ahora exactamente con el selector público `build_context.py`.
2. El proxy no seleccionaba el certificado local cuando el cliente conectaba por IP sin SNI. Se declaró `default_sni localhost` únicamente en la receta de ensayo. El certificado contiene SAN de localhost y de 127.0.0.1; el cliente sigue verificando su CA y el nombre. [Opción oficial de Caddy](https://caddyserver.com/docs/caddyfile/options#default-sni).
3. La prueba de empaquetado usa el motor Docker seleccionado por el ejecutor, sin exigir el nombre de contexto exclusivo de Docker Desktop.

## Límites y siguiente paso

El ensayo se coordinó desde Windows sobre el motor Linux de Docker Desktop. No acredita ejecución del helper en un anfitrión Linux, persistencia de un proveedor, acceso externo ni un rollback entre versiones de esquema. La entrega de archivos PWA no sustituye las pruebas anteriores de navegador ni constituye una instalación nueva de la PWA.

Los recursos se observaron con una sola evaluación ficticia pequeña: aproximadamente 38 MiB para la app y 15 MiB para el proxy en una muestra posterior al recorrido final. No son máximos de memoria, mediciones de carga ni requisitos del motor científico. El alojamiento del predictor completo y su costo todavía deben medirse.

La preparación y la predicción continúan bloqueadas en esta imagen de aplicación. No contiene ni intenta sustituir los modelos reales. No se modificaron metodología, entrenamientos, pesos ni prueba reservada.

El siguiente paso de la fase 5 es seleccionar alojamiento con disco persistente, comprobar su costo total dentro del presupuesto aprobado y preparar la publicación con dirección provisional y HTTPS. La contratación y publicación requieren su autorización propia; el dominio puede añadirse después. La integración científica conserva su fase 6.

Herramienta reproducible y responsabilidades de respaldo: [deploy/README.md](deploy/README.md).
