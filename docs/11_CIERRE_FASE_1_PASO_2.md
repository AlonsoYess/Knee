# Cierre verificado de la Fase 1, paso 2

## Estado

El paso 2 de la Fase 1 queda cerrado técnicamente el 29 de septiembre de 2026. La verificación independiente de las evidencias privadas confirmó que las 1,916 adquisiciones radiográficas basales esperadas están disponibles, son legibles y permanecen vinculadas al contrato de 2,778 rodillas elegibles. No quedan adquisiciones pendientes ni archivos inesperados.

Este cierre autoriza continuar con el diseño y validación de la separación bilateral y la lateralidad del paso 3. No autoriza recortes masivos, particiones definitivas, entrenamiento de modelos ni apertura de la prueba reservada.

## Revisión ejecutada

La descarga automática de los lotes 3 a 20 se ejecutó sobre la revisión pública `8c3be777d3fb3edfbbb82ff4aabb100d59ef4143`. El registro consolidado guardado en Drive declara estado completo, los dieciocho lotes previstos como completados, ningún lote fallido, cero pendientes y ninguna consulta de la prueba reservada.

Los lotes 1 y 2 ya habían sido ejecutados y verificados mediante sus revisiones registradas. La asignación adquisición–lote permaneció congelada durante todo el procedimiento.

## Evidencia final contrastada

| Control | Resultado |
| --- | ---: |
| Adquisiciones esperadas por el contrato | 1,916 |
| Adquisiciones inspeccionadas | 1,916 |
| Adquisiciones clasificadas como `DISPONIBLE` | 1,916 |
| Pendientes de descarga o reemplazo | 0 |
| Lotes pendientes | 0 |
| Archivos inesperados | 0 |
| Paquetes canónicos procedentes del piloto aprobado | 10 |
| Paquetes descargados selectivamente en los lotes 1–20 | 1,906 |
| Lotes de 100 paquetes | 19 |
| Paquetes del lote 20 | 6 |
| Paquetes esperados, encontrados, legibles y seleccionados en los 20 lotes | 1,906 en cada control |
| Paquetes promovidos a la ubicación canónica | 1,906 |
| Faltantes, ilegibles, duplicados, conflictos o inesperados en los lotes | 0 |
| Carpetas canónicas de lote verificadas | 20 |
| Total de paquetes comprobados directamente en esas carpetas | 1,906 |

Cada uno de los veinte resúmenes públicos de lote tiene estado `ok`, su número de lote correspondiente, un conjunto vacío de fallos y los indicadores `training_executed: false` y `reserved_test_opened: false`. Los registros privados conservan las huellas SHA-256 de paquete, objeto DICOM y matriz de píxeles sin publicarlas en GitHub.

## Cumplimiento de los criterios de cierre

1. Las 1,916 adquisiciones tienen una clasificación verificable y todas quedaron disponibles.
2. La cola selectiva quedó vacía y no existen lotes pendientes.
3. Los paquetes disponibles fueron abiertos, decodificados y registrados por contenido.
4. No se detectaron ausencias reales, ilegibilidades persistentes, duplicados ni conflictos que requieran exclusión.
5. El inventario final registra cero archivos ajenos a la cohorte.
6. La evidencia por lote, el registro consolidado, el inventario final y la revisión Git ejecutada permanecen guardados en Drive privado.

## Frontera de publicación y seguridad

GitHub conserva únicamente código, configuraciones de ejemplo, pruebas y recuentos agregados. Drive privado conserva los DICOM, las rutas, los identificadores, las huellas individuales, la cola histórica, el plan congelado y los registros de ejecución. Las credenciales se mantuvieron fuera del repositorio y de Drive.

No se entrenó ningún modelo, no se generaron particiones y no se abrió ni creó el conjunto de prueba reservado.

## Siguiente paso

Corresponde iniciar la Fase 1, paso 3: diseñar y validar una regla reproducible para separar las rodillas en las radiografías bilaterales y determinar su lateralidad sin consultar la etiqueta de progresión. Antes de aplicarla a toda la cohorte se deberá definir el protocolo, seleccionar casos de desarrollo, establecer controles de baja confianza y documentar la revisión visual.
