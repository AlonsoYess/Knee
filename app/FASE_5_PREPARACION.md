# Fase 5 — preparación, sin contratación ni publicación

Aplicación 0.12.0, API 0.5.0, esquema SQLite 2. Preparación: 7 de noviembre de 2026; comprobación Docker: 9 de noviembre. Rama: `codex/prototipo-fase5-preparacion`.

**Estado:** paquete, construcción Linux, proxy HTTPS y recuperación en otro volumen Docker comprobados. La virtualización ya está habilitada y la comprobación local pendiente quedó resuelta. Ver [evidencia Docker del 9 de noviembre](FASE_5_VERIFICACION_DOCKER.md). La fase 5 completa no está cerrada: no hay alojamiento contratado, dominio comprado ni aplicación publicada en Internet.

## Qué quedó preparado

- Dockerfile con etapas separadas, imágenes oficiales fijadas por digest, dependencias bloqueadas y usuario sin privilegios. Solo incluye código público de aplicación; no contiene cuentas, bases, imágenes clínicas, entornos ni modelos científicos.
- Contexto de construcción cerrado y TAR determinista auditable. Dos recursos PNG corresponden exclusivamente al icono de la aplicación.
- Entrada de producción explícita, un worker, secreto estable, origen HTTPS y rutas de datos/distribución declaradas. Se rechazan valores inseguros, fuentes de secreto contradictorias y activación científica en este paquete.
- Salud pública mínima sin cookie, casos ni rutas; diferencia disponibilidad de la aplicación y disponibilidad predictiva.
- Receta Compose con volumen de datos y secreto externo; receta de ensayo con proxy TLS solo loopback. Declaración y ejecución comprobadas con almacenamiento ficticio nuevo.
- Ensayo HTTPS reproducible y guía de persistencia, restauración, rollback y conexión posterior del dominio: [deploy/README.md](deploy/README.md).

## Evidencia ejecutada

| Comprobación | Resultado |
| --- | --- |
| Backend, incluidas 57 comprobaciones nuevas de despliegue | 242 pruebas correctas en la preparación del 7 de noviembre; backend sin cambios en la comprobación Docker. |
| Empaquetado con árboles sintéticos y exclusión de datos/secretos/enlaces | 9 correctas el 9 de noviembre, incluida la comprobación Docker; ninguna omitida. |
| Contrato OpenAPI 3.1 | 17 operaciones y 142 ejemplos ficticios validados. |
| Compilación Vue/TypeScript | Correcta; repetida dentro de la construcción Docker. |
| Receta Compose | Configuración y ejecución comprobadas; aplicación y proxy sin privilegios. |
| Dependencias Python Linux x86_64 fijadas | Instaladas dentro de la imagen; importaciones Linux y `pip check` correctos. |
| HTTPS real en el equipo, con configuración de producción y base nueva | Correcto: CA explícita, cadena y nombre validados, sin modificar confianza del sistema. |
| Pruebas HTTPS de acceso/guardado/errores, reinicio y restauración | 23 comprobaciones correctas: 11 iniciales, 6 de reinicio y 6 de restauración. |
| Respaldo completo | Snapshot y objeto DICOM generado verificados por SHA/tamaño; restauración nueva y sesión anterior revocada. |
| HTTPS y recuperación Docker | 23 comprobaciones correctas, recreación real del contenedor, respaldo exportado al equipo y restauración en otro volumen. |

El ensayo inició tres veces el servicio nativo de prueba, en un puerto separado. Conservó sesión/CSRF al reiniciar con la misma clave; tras restaurar exigió nuevo acceso y recuperó caso/preparación ficticios. Sin activos científicos, la carga quedó con fallo explícito de preparador no disponible; el original sintético referenciado se conservó y se respaldó. No se generó un QC falso ni una predicción.

La verificación local no demuestra persistencia en un proveedor. La evidencia distingue el ensayo nativo del 7 de noviembre y el ensayo Docker del 9 de noviembre. El investigador habilitó la virtualización; las herramientas de comprobación no modificaron BIOS, características de Windows ni el almacén de certificados, ni reiniciaron el equipo. La base local permanece separada del ensayo.

## Pendientes concretos

1. Elegir proveedor y costo total dentro del límite; obtener aprobación del gasto antes de contratar. El consumo observado solo cubre aplicación y proxy con un caso pequeño, sin modelos.
2. Publicar con dirección provisional y HTTPS; comprobar acceso externo, respaldo independiente y recuperación en el proveedor. El dominio propio puede añadirse después.
3. Mantener la integración y medición del paquete científico en su fase 6; esta imagen no ejecuta la predicción.

La metodología, los entrenamientos, sus pesos y la prueba reservada permanecen fuera de esta preparación. No se alojó ni transfirió contenido científico privado. La preparación de despliegue no acredita cierre predictivo ni validación clínica.
