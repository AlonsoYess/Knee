# Implementación preparada de `tibiofemoral_crop_v0.4_pilot`

## Estado y autorización

**Implementación preparada y no ejecutada sobre los datos privados.** El investigador autorizó implementar `v0.4` y publicar los cambios en la rama actual. La autorización cubre el localizador determinista, su configuración pública, pruebas automáticas, la libreta de repetición sobre las mismas veinte rodillas y la copia aprobada de esa libreta en el área privada de notebooks.

La autorización no incluye ejecutar Colab en nombre del investigador, procesar masivamente las 1,916 adquisiciones, crear particiones, entrenar modelos ni abrir la prueba reservada. El paso 4 continúa abierto hasta completar y cerrar una nueva revisión visual ciega.

## Cambios implementados

`tibiofemoral_crop_v0.4_pilot` introduce cuatro controles acotados:

1. **consenso vertical entre familias de señal:** combina el máximo multiseñal por compartimento de `v0.2` con los pares de bordes dirigidos de `v0.3` y mide su distancia física;
2. **soporte óseo bilateral:** exige evidencia por encima y por debajo del espacio candidato en varias subbandas de ambos compartimentos;
3. **selección horizontal gobernada por anatomía:** enumera cajas físicas de 140 × 140 mm que conservan una envolvente anatómica con margen de seguridad antes de considerar artefactos;
4. **artefactos como puertas de rechazo:** componentes compactos y líneas brillantes periféricas, fondo, borde, desplazamiento y saturación pueden impedir `HIGH`, pero no desplazan la caja fuera de la anatomía.

Cuando no existe consenso vertical se registra `REVIEW_REQUIRED_VERTICAL_DISAGREEMENT`. Cuando ninguna caja conserva simultáneamente anatomía y límites se registra `REVIEW_REQUIRED_NO_FEASIBLE_CROP`. La mejor caja anatómica permanece disponible para revisión, pero ninguna puerta fallida puede ser compensada por la puntuación continua.

La estrategia anterior permanece accesible por despacho explícito; la implementación no sustituye ni reescribe `v0.2` o `v0.3`.

## Artefactos preparados

- [`configs/joint_localization.v0.4.example.json`](../configs/joint_localization.v0.4.example.json) define la versión, las puertas físicas y una salida independiente en `v0_4_piloto`.
- [`notebooks/11_repeticion_localizacion_tibiofemoral_v04.ipynb`](../notebooks/11_repeticion_localizacion_tibiofemoral_v04.ipynb) verifica primero el cierre rechazado de `v0.3`, ejecuta únicamente las mismas diez adquisiciones y veinte rodillas, muestra las veinte vistas y registra que la revisión continúa pendiente.
- `src/knee/joint_localization.py` conserva en la salida privada el consenso entre métodos, el soporte óseo, la envolvente anatómica, las puertas y los motivos de revisión.

La libreta obtiene la raíz privada desde `KNEE_DATA_ROOT`; no publica rutas, identificadores, huellas ni decisiones individuales.

## Verificación automática

Las pruebas añadidas comprueban:

- lectura y validación completa de la configuración `v0.4`;
- acuerdo entre los dos estimadores verticales en un escenario sintético válido;
- imposibilidad de confianza `HIGH` cuando los estimadores discrepan;
- conservación anatómica ante artefactos periféricos sin desplazar obligatoriamente la caja;
- detección de estructuras lineales brillantes periféricas;
- campo físico exacto y conservación del despacho histórico;
- estructura, limpieza, cierre previo y bloqueos de la libreta 11.

Estas pruebas demuestran el contrato del código y de la orquestación; no sustituyen la revisión visual de las radiografías privadas.

## Próxima secuencia controlada

1. el investigador abre la copia aprobada de la libreta 11 en Colab;
2. ejecuta todas las celdas con `KNEE_DATA_ROOT` configurado;
3. confirma que se generaron exactamente veinte vistas y comparte la salida final;
4. completa la revisión visual ciega de las veinte rodillas;
5. solo después se prepara un cierre reproducible de `v0.4`.

Hasta entonces, los parámetros no se congelan y permanecen en `False`:

- `mass_processing_executed`;
- `partitions_created`;
- `training_executed`;
- `reserved_test_opened`.

Si `v0.4` vuelve a ser rechazada, no se creará automáticamente una `v0.5`. Se documentará el límite del enfoque determinista y cualquier cambio de familia técnica requerirá una propuesta metodológica separada y autorización explícita.
