# Revisión de la Fase 1, paso 4: rechazo de `tibiofemoral_crop_v0.1_pilot`

## Resultado

La primera versión del localizador tibiofemoral procesó sin fallos de integridad los diez estudios y produjo veinte recortes por rodilla. La revisión se realizó sobre vistas seudonimizadas, sin consultar el desenlace de progresión estructural.

El resultado agregado fue:

| Criterio | Resultado |
| --- | ---: |
| Rodillas revisadas | 20 |
| Recortes íntegramente aceptables | 9 |
| Recortes rechazados | 11 |
| Líneas articulares no centradas | 9 |
| Recortes con regla o puntos periféricos | 7 |
| Exclusiones técnicas de DICOM | 0 |

Los defectos se superponen: algunos recortes presentaron simultáneamente una línea desplazada y elementos de la regla. La anatomía tibiofemoral permaneció visible en las veinte vistas, por lo que no se excluyeron adquisiciones ni participantes. El defecto pertenece al procedimiento candidato y no a la integridad de la fuente.

## Decisión

`tibiofemoral_crop_v0.1_pilot` queda **rechazado después de revisión visual ciega**. Sus parámetros no se congelan y la versión no puede utilizarse para el procesamiento masivo, las particiones ni el entrenamiento.

La confianza automática inicial tampoco se considera suficiente: varios errores visuales habían recibido nivel `HIGH`. Por ello no se corrige el problema mediante un único umbral de confianza.

## Corrección autorizada dentro del piloto

La versión `tibiofemoral_crop_v0.2_pilot` mantiene el mismo procedimiento determinista y físico, pero:

1. calcula perfiles separados para los compartimentos medial y lateral;
2. excluye la región intercondílea central de la decisión primaria;
3. mide la concordancia vertical entre los máximos de ambos compartimentos;
4. deriva las discordancias a revisión en vez de declararlas automáticamente correctas;
5. conserva el campo de visión candidato de 140 × 140 mm;
6. exige una separación física del borde interno para evitar la regla central.

Esta corrección no modifica población, predictores, desenlace, horizonte, unidad de análisis, particiones, métricas ni arquitecturas. Se ejecutará nuevamente sobre las mismas veinte rodillas del piloto mediante [`08_repeticion_localizacion_tibiofemoral_v02.ipynb`](../notebooks/08_repeticion_localizacion_tibiofemoral_v02.ipynb).

## Bloqueos vigentes

Hasta que la nueva versión supere la revisión completa:

- no se procesarán masivamente las 1,916 adquisiciones;
- no se congelarán recortes ni una cohorte radiográfica;
- no se crearán particiones;
- no se entrenará ningún modelo;
- no se abrirá la prueba reservada.

Las decisiones individuales, las coordenadas, las huellas y las imágenes permanecen únicamente en el almacenamiento privado autorizado.
