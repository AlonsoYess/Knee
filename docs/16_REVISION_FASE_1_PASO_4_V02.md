# Revisión de la Fase 1, paso 4: rechazo de `tibiofemoral_crop_v0.2_pilot`

## Resultado

La segunda versión del localizador tibiofemoral procesó sin fallos de integridad los diez estudios y produjo veinte recortes por rodilla. La revisión se realizó sobre vistas seudonimizadas y sin consultar el desenlace de progresión estructural.

El resultado agregado fue:

| Criterio | Resultado |
| --- | ---: |
| Rodillas revisadas | 20 |
| Recortes íntegramente aceptables | 8 |
| Recortes rechazados | 12 |
| Líneas articulares no centradas | 8 |
| Desplazamientos hacia la tibia | 7 |
| Desplazamientos hacia el fémur | 1 |
| Recortes con regla o puntos periféricos | 5 |
| Anatomías tibiofemorales completas | 20 |
| Exclusiones técnicas de DICOM | 0 |

Un recorte presentó simultáneamente descentrado y puntos de la regla, por lo que los defectos suman trece observaciones sobre doce recortes rechazados. La anatomía relevante permaneció visible en las veinte vistas. Ninguna adquisición ni participante fue excluido: los defectos pertenecen al procedimiento candidato y no a la integridad de la fuente.

## Comparación controlada con `v0.1`

`v0.2` redujo las líneas no centradas de nueve a ocho y los campos con regla o puntos de siete a cinco. Sin embargo, los dos defectos afectaron rodillas distintas con menor superposición y la aceptación integral descendió de 9/20 a 8/20. Por tanto, una mejora aislada en cada recuento de defecto no equivale a superar el criterio conjunto del recorte.

La confianza automática tampoco separó adecuadamente los resultados:

| Confianza automática | Aceptables | Rechazados |
| --- | ---: | ---: |
| `HIGH` | 1 | 2 |
| `LOW` | 7 | 10 |

No se corregirá esta limitación cambiando únicamente el umbral de confianza. La decisión debe conservar puertas explícitas para localización vertical y exclusión de artefactos.

## Decisión

`tibiofemoral_crop_v0.2_pilot` queda **rechazado después de revisión visual ciega**. Sus parámetros no se congelan y la versión no puede utilizarse para el procesamiento masivo, las particiones ni el entrenamiento.

La evidencia privada conserva las veinte decisiones individuales, las coordenadas, las huellas y las imágenes. Este documento publica únicamente resultados agregados.

## Siguiente acto permitido

Se autoriza documentar y evaluar una propuesta `v0.3` limitada al mismo localizador determinista y multiseñal previsto en el protocolo. La propuesta se conserva en [`17_PROPUESTA_FASE_1_PASO_4_V03.md`](17_PROPUESTA_FASE_1_PASO_4_V03.md).

La propuesta no equivale a autorizar su implementación. Antes de modificar el algoritmo deberá existir aprobación expresa del investigador. Una alternativa entrenada o un cambio de población, entrada, desenlace, horizonte, unidad de análisis o modelos requerirá el control de cambios metodológicos correspondiente.

## Bloqueos vigentes

Hasta que una versión supere la revisión completa:

- no se procesarán masivamente las 1,916 adquisiciones;
- no se congelarán recortes ni una cohorte radiográfica;
- no se crearán particiones;
- no se entrenará ningún modelo;
- no se abrirá la prueba reservada.

