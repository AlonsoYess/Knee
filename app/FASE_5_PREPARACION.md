# Fase 5 — preparación, sin contratación ni publicación

Aplicación 0.12.0, API 0.5.0, esquema SQLite 2. Fecha: 7 de noviembre de 2026. Rama: `codex/prototipo-fase5-preparacion`.

**Estado:** paquete y configuración preparados; ensayo HTTPS nativo y recuperación comprobados. **Imagen y ejecución Docker pendientes** por virtualización desactivada, confirmada por el investigador. La fase 5 completa no está cerrada: no hay alojamiento contratado, dominio comprado ni aplicación publicada en Internet.

## Qué quedó preparado

- Dockerfile con etapas separadas, imágenes oficiales fijadas por digest, dependencias bloqueadas y usuario sin privilegios. Solo incluye código público de aplicación; no contiene cuentas, bases, imágenes clínicas, entornos ni modelos científicos.
- Contexto de construcción cerrado y TAR determinista auditable. Dos recursos PNG corresponden exclusivamente al icono de la aplicación.
- Entrada de producción explícita, un worker, secreto estable, origen HTTPS y rutas de datos/distribución declaradas. Se rechazan valores inseguros, fuentes de secreto contradictorias y activación científica en este paquete.
- Salud pública mínima sin cookie, casos ni rutas; diferencia disponibilidad de la aplicación y disponibilidad predictiva.
- Receta Compose con volumen de datos y secreto externo; receta de ensayo con proxy TLS solo loopback. Validación declarativa correcta, sin afirmar que esos servicios hayan ejecutado.
- Ensayo HTTPS reproducible y guía de persistencia, restauración, rollback y conexión posterior del dominio: [deploy/README.md](deploy/README.md).

## Evidencia ejecutada

| Comprobación | Resultado |
| --- | --- |
| Backend, incluidas 57 comprobaciones nuevas de despliegue | 242 pruebas correctas. |
| Empaquetado con árboles sintéticos y exclusión de datos/secretos/enlaces | 8 correctas; una comprobación Docker omitida y pendiente. |
| Contrato OpenAPI 3.1 | 17 operaciones y 142 ejemplos ficticios validados. |
| Compilación Vue/TypeScript | Correcta. |
| Receta Compose | Configuración declarativa válida, sin ejecución. |
| Dependencias Python Linux x86_64 fijadas | Artefactos binarios disponibles y descargados, sin instalar ni ejecutar en Linux. |
| HTTPS real en el equipo, con configuración de producción y base nueva | Correcto: CA explícita, cadena y nombre validados, sin modificar confianza del sistema. |
| Pruebas HTTPS de acceso/guardado/errores, reinicio y restauración | 23 comprobaciones correctas: 11 iniciales, 6 de reinicio y 6 de restauración. |
| Respaldo completo | Snapshot y objeto DICOM generado verificados por SHA/tamaño; restauración nueva y sesión anterior revocada. |

El ensayo inició tres veces el servicio nativo de prueba, en un puerto separado. Conservó sesión/CSRF al reiniciar con la misma clave; tras restaurar exigió nuevo acceso y recuperó caso/preparación ficticios. Sin activos científicos, la carga quedó con fallo explícito de preparador no disponible; el original sintético referenciado se conservó y se respaldó. No se generó un QC falso ni una predicción.

La verificación de salud y permisos no demuestra por sí sola persistencia en un proveedor. Las pruebas nativas no demuestran que la imagen Linux construya o ejecute; la evidencia diferencia expresamente ambos alcances. No se modificó BIOS, virtualización, características de Windows ni el almacén de certificados. No se reinició el equipo y el servicio local existente permanece separado del ensayo.

## Pendientes concretos

1. Habilitar la virtualización cuando el investigador lo decida y comprobar el servidor Linux de Docker.
2. Construir e inspeccionar la imagen; comprobar instalación Linux, capas, usuario y permisos reales.
3. Ensayar proxy HTTPS y recreación/restauración sobre volúmenes de contenedores. Medir disco, memoria y tiempos de esa distribución.
4. Elegir proveedor y costo total dentro del límite; obtener aprobación del gasto antes de contratar. La capacidad del paquete científico completo aún no se ha medido para ese alojamiento.
5. Publicar con dirección provisional y HTTPS; comprobar acceso externo, respaldo independiente y recuperación. El dominio propio puede añadirse después.

La metodología, los entrenamientos, sus pesos y la prueba reservada permanecen fuera de esta preparación. No se alojó ni transfirió contenido científico privado. La preparación de despliegue no acredita cierre predictivo ni validación clínica.
