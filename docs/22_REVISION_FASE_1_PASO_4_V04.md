# Revisión visual ciega de `tibiofemoral_crop_v0.4_pilot`

## Estado

**Versión rechazada; parámetros no congelados y paso 4 abierto.** La libreta `11_repeticion_localizacion_tibiofemoral_v04.ipynb` se ejecutó sobre los mismos diez estudios y veinte rodillas del piloto desde la revisión Git `c8460c1e4feea45e02d955254a05afcfa0198f09`. No hubo fallos técnicos ni se consultó el desenlace.

## Resultado agregado

La revisión visual ciega registró:

- 20 rodillas revisadas;
- 8 recortes íntegramente aceptables;
- 12 recortes rechazados;
- 9 líneas articulares no centradas;
- 0 recortes con anatomía tibiofemoral incompleta;
- 6 recortes con puntos de la regla dentro del campo;
- 0 exclusiones técnicas.

Los defectos se superponen: tres recortes presentaron simultáneamente una línea no centrada y puntos de la regla. Un recorte solo podía aceptarse si la línea estaba centrada, la anatomía estaba completa, los artefactos periféricos estaban excluidos y no existía exclusión técnica.

## Hallazgos

`v0.4` recuperó la conservación anatómica de las veinte vistas y evitó la regresión horizontal observada en `v0.3`. Sin embargo, el consenso entre las dos familias deterministas de señal no discriminó de forma estable la interlínea tibiofemoral: ocho candidatos quedaron por encima de la interlínea y uno por debajo.

La validación periférica tampoco excluyó todos los puntos de la regla. Seis campos conservaron esos artefactos; en tres de ellos el defecto periférico coincidió con un error vertical. La mejora de 0/20 a 8/20 recortes aceptables no satisface el criterio integral del piloto.

## Consecuencia metodológica

`tibiofemoral_crop_v0.4_pilot` queda rechazado. Sus parámetros no se congelan y no se autoriza:

- procesamiento masivo;
- creación de particiones;
- entrenamiento;
- apertura de la prueba reservada.

La libreta [`12_cierre_revision_localizacion_tibiofemoral_v04.ipynb`](../notebooks/12_cierre_revision_localizacion_tibiofemoral_v04.ipynb) queda preparada para verificar el CSV privado, exigir exactamente los recuentos agregados anteriores y generar el registro reproducible del cierre. Hasta que esa libreta se ejecute en Colab y se contraste su salida, la revisión está documentada pero el cierre operativo permanece pendiente.

No se crea automáticamente una `v0.5`. Sustituir la familia determinista por un localizador anatómico aprendido requiere una solicitud de cambio metodológico separada, evaluación de procedencia y licencia de los pesos, análisis de solapamiento con OAI y aprobación explícita antes de implementarla.

## Cierre operativo posterior del 1 de octubre de 2026

El investigador ejecutó en Colab la libreta 12 sobre el CSV privado completado. Los archivos `resumen_revision_visual_publico.json` y `registro_cierre_revision.json` se contrastaron en Drive: 20 revisiones, 8 aceptables, 12 rechazadas, 0 exclusiones, 9 fallos de centrado, 6 artefactos y parámetros no congelados. El registro identifica `Google Colab, libreta 12` y conserva la procedencia de una **nueva revisión técnica asistida por Codex** con el agregado histórico 8/20 conocido. No es transcripción de las decisiones individuales históricas ni una revisión independiente. El rechazo formal de v0.4 queda acreditado; el paso 4 sigue abierto. La sustitución aprobada vigente es MCR-2026-004, no una v0.5 ni la solicitud MCR-2026-003 retirada.
