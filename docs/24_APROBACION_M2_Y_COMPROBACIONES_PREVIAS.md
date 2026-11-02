# MCR-2026-003: aprobación y comprobaciones previas

| Campo | Valor |
| --- | --- |
| Fecha | 30 de septiembre de 2026 |
| Versión | 1.0 |
| Decisión | Aprobada por el investigador; `aprobada_no_aplicada` |
| Solicitud vinculada | [MCR-2026-003](23_MCR_2026_003_LOCALIZACION_ANATOMICA.md), contenido técnico 1.0 |
| Evidencia | Respuesta directa «si apruebo, hazlo bien» después de confirmar alcance y exclusiones |
| Naturaleza | Registro documental y comprobaciones de lectura; no experimento |

## 1. Alcance de la aprobación

Se aprueba el contenido técnico presentado: localizador externo congelado, KNEEL como candidato condicionado, referencias humanas, geometría de 140 mm, piloto histórico y 60 participantes nuevos reservados para desarrollo. Se mantienen sin cambios la muestra, umbrales, condiciones y exclusiones de la solicitud. La revisión documental 1.1 agrega únicamente la decisión y su seguimiento.

Se permite iniciar las comprobaciones previas. La integración/evaluación se realizará solamente después de acreditar G0–G2. No están autorizados entrenamiento ni ajuste fino, procesamiento masivo, particiones definitivas, apertura de prueba, cambio de candidato automático o adaptación de umbrales tras conocer resultados.

## 2. Estado comprobado de las puertas

| Puerta | Evidencia disponible | Estado / evidencia que falta |
| --- | --- | --- |
| G0 — decisión | Aprobación expresa y registro JSON válido | Aprobación registrada |
| G0 — cierre operativo v0.4 | Revisión agregada local: 8/20 aceptables; libreta 12 preparada | No verificado: la carpeta privada consultada no devuelve los dos archivos de cierre esperados |
| G0 — coherencia documental | Contrato rector 1.2, reglas 1.3 y contrato ejecutable 1.3 registran la aprobación condicionada | Enmienda de 3.2.1 aprobada; incorporación al Word académico y respaldo de nuevas versiones pendientes |
| G1 — referencia y campo | Plan aprobado; imágenes del piloto ya disponibles | Lectores no confirmados, sin anotaciones adjudicadas ni prueba geométrica frente a referencia |
| G2 — pesos | README oficial y ficha de distribución consultados | Acceso sujeto a condiciones; licencia exacta, checkpoint, procedencia completa y disyunción no acreditados |
| G3–G6 | Ninguna ejecución del nuevo candidato | No iniciadas; no se declara integración disponible |

No acreditado no significa fallo definitivo del candidato. Tampoco puede convertirse en un requisito satisfecho por cambiar una bandera en el registro.

### Comprobación privada de v0.4

Se localizó la carpeta del piloto por nombre y relación de carpeta padre, se listaron sus siete elementos y se consultaron los dos JSON de preparación. Una búsqueda exacta dentro de esa misma carpeta no devolvió `registro_cierre_revision.json` ni `resumen_revision_visual_publico.json`. Los artefactos disponibles corresponden a la preparación para revisión, no al cierre.

Este resultado acredita ausencia de evidencia accesible en la ubicación consultada, no que el investigador jamás haya ejecutado el cierre en otro entorno. No se modificó Drive ni se volvió a ejecutar el piloto. La decisión de rechazo 8/20 se conserva, pero no se sustituye la validación del CSV con ese número agregado.

Para completar esta parte de G0 se debe ejecutar y contrastar la [libreta 12](../notebooks/12_cierre_revision_localizacion_tibiofemoral_v04.ipynb), con revisión completa y código identificable. Sus salidas deben acreditar versión correcta, 20 revisiones, 8 aceptables, 12 rechazadas, cero exclusiones, cegamiento y parámetros no congelados. Si el CSV no coincide, se detiene el cierre y se resuelve la discrepancia; no se fuerza el resumen esperado. El notebook está preparado localmente: no se afirma publicado ni ejecutado desde esta comprobación.

### Consulta pública de KNEEL

El [README oficial](https://github.com/imedslab/KNEEL) describe inferencia bilateral y acceso mediante Hugging Face. La [ficha de pesos](https://huggingface.co/imedslab/kneel) y su [listado público](https://huggingface.co/imedslab/kneel/tree/main) muestran acceso condicionado. La información pública revisada no acredita la lista completa de sujetos utilizados en la cadena del checkpoint que se distribuiría.

Por tanto se conserva `provenance_unresolved`: ningún peso se declaró elegible. No se inició sesión, aceptaron términos, descargaron pesos, contactaron autores ni enviaron imágenes. Esta inspección no prueba incompatibilidad del modelo; identifica documentación y acceso que faltan.

## 3. Enmienda académica y trazabilidad

El texto de la sección 6 de MCR-2026-003 queda aprobado como enmienda de 3.2.1, sin alteraciones técnicas posteriores a la decisión. Los contratos del repositorio enlazan esa enmienda y distinguen el procedimiento histórico del candidato autorizado.

Los Word entregados se conservan intactos. La incorporación material deberá hacerse en una versión académica controlada, preservando el original y verificando que no se alteren capítulos, variables o compromisos ajenos a M2. Mientras no exista esa evidencia, G0 no está completa y no comienza G3. No se considera que actualizar un JSON equivalga a actualizar la tesis.

`knee.change_control` puede devolver `AUTHORIZED` para la solicitud: significa aprobación para el alcance exacto. No comprueba por sí solo los datos, lectores, licencia ni geometría. `integration_ready=false` registra que esas condiciones todavía faltan; no se presenta como una barrera de ejecución ya implementada en un localizador futuro.

## 4. Acciones siguientes, en orden

1. Contrastar los archivos de cierre de v0.4 y completar la incorporación académica controlada. No publicar evidencia individual ni sobrescribir originales.
2. Confirmar disponibilidad de los dos lectores y adjudicación experta. Preparar el manual, verificar semántica de puntos y obtener referencias sobre las veinte rodillas antes de validar milímetros o factibilidad.
3. Acreditar licencia y procedencia del checkpoint real, con cruce privado por participante contra los universos definidos. Si se necesita contacto con autores o aceptar condiciones de una cuenta, gestionarlo expresamente; no compartir identificadores de la cohorte por correo público.
4. Solo con G0–G2 cumplidas: integrar el adaptador, ejecutar regresión y congelar antes de seleccionar/evaluar la calificación según el orden de M2.

En este turno no se seleccionaron los 60 participantes, no se generaron anotaciones de referencia y no se realizaron experimentos. El paso 4 sigue abierto. La siguiente decisión operativa depende de aportar el cierre, confirmar lectores y resolver los pesos; no se solicita una segunda aprobación de M2 ni se cambia la solución aprobada.
