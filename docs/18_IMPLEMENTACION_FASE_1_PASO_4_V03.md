# Implementación preparada de `tibiofemoral_crop_v0.3_pilot`

## Estado

**Implementación ejecutada y rechazada en el piloto privado.** La autorización recibida cubrió el localizador determinista `v0.3`, su configuración pública, las pruebas sintéticas y una libreta nueva para repetir exclusivamente las mismas veinte rodillas del piloto. No autorizó procesamiento masivo, particiones, entrenamiento ni apertura de la prueba reservada.

La ejecución se realizó desde la revisión Git `173fc7f6b7e8585f9d0c02346cad1bd03d0e90ec`. La revisión visual ciega posterior aceptó 0/20 recortes, por lo que los parámetros no se congelan y el paso 4 continúa abierto. El resultado se documenta en [`19_REVISION_FASE_1_PASO_4_V03.md`](19_REVISION_FASE_1_PASO_4_V03.md).

## Cambios implementados

`tibiofemoral_crop_v0.3_pilot` incorpora:

1. gradientes verticales con signo y pares ordenados de borde femoral descendente y borde tibial ascendente;
2. selección conjunta de los compartimentos mediante concordancia del centro y del ancho físico del espacio candidato;
3. detección determinista de componentes brillantes compactos junto al borde interno y un margen efectivo igual al mayor entre el mínimo físico y la extensión detectada más su resguardo;
4. puertas obligatorias independientes para bordes, consenso, límites, fondo, saturación y separación del artefacto;
5. confianza `HIGH` solo cuando todas las puertas son válidas;
6. trazabilidad privada de bordes, anchos, componentes periféricos, puertas y motivos de revisión;
7. despacho explícito por estrategia para conservar sin cambios el comportamiento histórico de `v0.2`.

La configuración candidata está en [`configs/joint_localization.v0.3.example.json`](../configs/joint_localization.v0.3.example.json). La libreta [`09_repeticion_localizacion_tibiofemoral_v03.ipynb`](../notebooks/09_repeticion_localizacion_tibiofemoral_v03.ipynb) valida primero el rechazo de `v0.2` —8 aceptables y 12 rechazados— y después prepara una salida independiente en `v0_3_piloto`.

## Verificación automática

Las pruebas sintéticas cubren:

- delimitación del espacio verdadero por bordes opuestos;
- resistencia ante una banda oscura falsa inferior;
- revisión obligatoria cuando los compartimentos son incompatibles;
- ampliación del margen interno ante puntos brillantes de regla;
- imposibilidad de confianza alta si falla una puerta obligatoria;
- validez estructural, limpieza y bloqueos de la libreta nueva.

La ejecución privada se realizó en Colab y sus artefactos permanecen en Drive; no forma parte de las pruebas sintéticas del repositorio.

## Secuencia autorizada completada

1. se identificó la revisión de Git que contiene esta implementación;
2. se ejecutó la libreta 09 en Colab con el secreto `KNEE_DATA_ROOT`;
3. se procesaron exactamente diez estudios y veinte rodillas, sin fallos de integridad;
4. se revisaron visualmente las veinte vistas sin consultar el desenlace;
5. se registró el rechazo de `v0.3`;
6. no se congelaron parámetros ni se cerró el paso 4.

La libreta 10 quedó preparada para generar el resumen público reproducible desde el CSV privado ya completado.

Al finalizar esa secuencia permanecen en `False`:

- `mass_processing_executed`;
- `partitions_created`;
- `training_executed`;
- `reserved_test_opened`.

## Criterio de no avance

Si `v0.3` no centra la articulación y excluye la regla en las veinte rodillas, sus parámetros no se congelarán. Tampoco se corregirán manualmente casos individuales. Cualquier cambio de familia técnica —por ejemplo, un localizador supervisado— requerirá evidencia, evaluación de impacto y autorización metodológica separada.
