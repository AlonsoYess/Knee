# Revisión visual ciega de `tibiofemoral_crop_v0.3_pilot`

## Estado

**Versión rechazada; parámetros no congelados y paso 4 abierto.** La libreta `09_repeticion_localizacion_tibiofemoral_v03.ipynb` se ejecutó sobre los mismos diez estudios y veinte rodillas del piloto desde la revisión Git `173fc7f6b7e8585f9d0c02346cad1bd03d0e90ec`. No hubo fallos técnicos ni se consultó el desenlace.

## Resultado agregado

La revisión visual ciega registró:

- 20 rodillas revisadas;
- 0 recortes íntegramente aceptables;
- 20 recortes rechazados;
- 8 líneas articulares no centradas;
- 11 recortes con anatomía tibiofemoral incompleta;
- 14 recortes con borde, fondo negro o puntos de la regla dentro del campo;
- 0 exclusiones técnicas.

Un recorte solo podía aceptarse si la línea estaba centrada, la anatomía estaba completa, los artefactos periféricos estaban excluidos y no existía exclusión técnica. Ninguna de las veinte vistas cumplió simultáneamente las cuatro condiciones.

## Hallazgos

La banda periférica adaptativa de `v0.3` no resolvió el problema de manera estable. En varias rodillas desplazó horizontalmente la caja hacia el límite del campo unilateral, lo que introdujo fondo negro o cortó un compartimento. En otras, el par de bordes elegido situó la línea por encima o por debajo de la interlínea tibiofemoral.

La única candidata etiquetada automáticamente como `HIGH` conservó puntos de la regla. Por tanto, las puertas automáticas de confianza tampoco demostraron una separación fiable entre casos aceptables y no aceptables.

## Consecuencia metodológica

`tibiofemoral_crop_v0.3_pilot` queda rechazado. Sus parámetros no se congelan y no se autoriza:

- procesamiento masivo;
- creación de particiones;
- entrenamiento;
- apertura de la prueba reservada.

La libreta [`10_cierre_revision_localizacion_tibiofemoral_v03.ipynb`](../notebooks/10_cierre_revision_localizacion_tibiofemoral_v03.ipynb) quedó preparada para validar el CSV privado y generar el resumen público reproducible. Su ejecución en Colab es el único cierre operativo pendiente de esta revisión. Cualquier nueva versión o cambio de familia técnica requiere una propuesta explícita y autorización separada.
