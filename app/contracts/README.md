# Contrato Vue–FastAPI, v1

Estado: **acceso, datos basales, preparación DICOM y QC persistidos; predicción no disponible**. Fecha: 7 de noviembre de 2026.

[openapi.json](openapi.json) contiene OpenAPI 3.1.0 y JSON Schema 2020-12. La versión **1.0.0-phase3** identifica la comunicación de la aplicación, no un modelo ni una aprobación de despliegue. Todos sus ejemplos son ficticios. No incluyen porcentajes predictivos, datos de investigación ni credenciales operativas. Cada operación declara `x-implementation-status`.

Vue consume la API de `app/backend`. El backend conserva sesiones, casos, referencias y decisiones; los archivos permanecen privados. Un proceso separado utiliza el entorno científico configurado y las funciones congeladas de `src/knee`. El localizador anatómico puede cargar sus pesos locales; los modelos de progresión y la prueba reservada permanecen fuera de este flujo. No se instala el entorno científico dentro del backend ni se modifican sus transformaciones.

## Operaciones y alcance actual

Las rutas parten de **/api/v1**, bajo el mismo origen de la SPA.

| Método y ruta | Comportamiento |
| --- | --- |
| GET /session | Consultar sesión o crear contexto anónimo y entregar CSRF. |
| POST /session | Autenticar cuenta administrada y rotar sesión y CSRF. |
| DELETE /session | Revocar sesión y expirar cookie. |
| GET /evaluations | Listar casos propios con cursor. |
| POST /evaluations | Crear borrador basal; clave idempotente opcional, siempre enviada por Vue. |
| GET /evaluations/{evaluationId} | Leer caso y su ETag. |
| PATCH /evaluations/{evaluationId} | Guardar basales con control de versión. |
| POST /evaluations/{evaluationId}/preparations | Cargar un DICOM, guardar fuente y preparar vistas previas. |
| GET /preparations/{preparationId} | Leer fuente, referencia, candidato, decisión vigente y ETag. |
| GET /preparations/{preparationId}/views/{view}?revision=... | Obtener un PNG privado autenticado. |
| POST /preparations/{preparationId}/reference | Sellar referencia visual previa y, si está resuelta, preparar candidato. |
| POST /preparations/{preparationId}/retry | Reintentar explícitamente una preparación fallida con la misma fuente y referencia. |
| POST /preparations/{preparationId}/reviews | Guardar decisión de QC vinculada al candidato exacto. |
| GET /engine-status | Informar `not_configured` para el motor predictivo. |
| POST /evaluations/{evaluationId}/executions | Devolver 503 después de comprobar acceso; no ejecuta predicción. |
| GET /executions/{executionId} | Contrato futuro, ruta todavía no implementada. |

Las cuentas se administran por CLI. Recuperación por correo, corrección geométrica manual, selección de modelos, entrenamiento, publicación, PWA y sincronización quedan fuera de este contrato implementado. El flujo científico vigente no permite trasladar, redimensionar, reparar o ajustar manualmente la caja del recorte.

## Caso y datos basales

`Evaluation` identifica edad, sexo, IMC, lado y una revisión basal persistida. Las claves HTTP son `age`, `sex`, `bmi` y `side`; la interfaz adapta su representación interna a esas claves. Código, propietario, revisión y fechas los asigna el servidor. No recibe KL, desenlace ni particiones.

Los borradores admiten `null`. Una ausencia no se convierte aquí en exclusión clínica automática. La aptitud de la futura inferencia dependerá del paquete congelado: si admite IMC ausente e imputación, conservará `null` y aplicará exclusivamente el estado ya ajustado. No se calcularán nuevas estadísticas en el prototipo. Edad entera positiva e IMC positivo son comprobaciones de formato, no límites clínicos de elegibilidad.

## Carga y preparación de la fuente

La carga es `multipart/form-data` con **exactamente un archivo en el campo `file`**, sin campos adicionales. Admite las declaraciones MIME `application/dicom` y `application/octet-stream`; el contenido se inspecciona igualmente. Los límites predeterminados son **60 MiB por archivo** y **64 MiB por solicitud**, configurables como límites técnicos. ZIP, TAR, SVG y PNG no son alternativas de entrada de esta operación. El nombre suministrado por el navegador no se usa como ruta privada.

Antes de cargar se guarda el lado del caso. La solicitud exige sesión, Origin autorizado, CSRF, ETag del caso en `If-Match` e `Idempotency-Key`. El servidor registra identidad, bytes y preparación de forma trazable. Una carga nueva devuelve **202** con `Preparation`; la implementación actual realiza la etapa de fuente antes de responder y normalmente entrega `awaiting_reference`. El código 202 no implica que exista una cola distribuida ni que el QC haya sido aprobado.

`Preparation` contiene:

- `sourceContext.views`: `full_a`, `full_b`, `half_a` y `half_b`.
- `sourceContext.fullNativeShape`: `[alto, ancho]` del original bilateral.
- `nativeHalfShape`: `[alto, ancho]` de la mitad seleccionada, en el nivel principal del objeto.
- `sourceReference`: referencia sellada o `null`.
- `cropGeometry`: geometría del candidato o `null`.
- `reviewContext`: contexto disponible para QC o `null`.
- `activeReview`: decisión vigente y compatible o `null`.
- `isCurrent`: si corresponde a la imagen y lado activos del caso.
- `reused`: si esa respuesta procede de recuperación/reutilización; no expresa calidad.

`full` representa el bilateral original, `half` la mitad seleccionada y `crop` el recorte. La ventana **a** utiliza percentiles 1–99 y **b** utiliza 5–95. Los PNG conservan las dimensiones nativas para relacionar posiciones visuales y coordenadas. Son vistas de ocho bits; los arrays enteros originales permanecen privados y no se sustituyen por esas vistas ni por un tensor 224×224.

## Referencia antes del localizador

El usuario inspecciona original y mitad, sin mostrar aún el recorte automático. Envía `SourceReferenceSubmission` con revisión de fuente, lado confirmado, `centerResolved`, `centerXY`, inspección, cegamiento y una nota específica. `centerXY` son enteros **[x,y] de la mitad nativa**, nunca coordenadas CSS ni del recorte.

El servidor valida la referencia mediante el contrato congelado y la persiste **antes** del localizador. La referencia no se pasa al algoritmo geométrico. Una referencia resuelta permite preparar el candidato; una referencia no resuelta debe llevar `centerXY=null` y queda sellada sin candidato ni aprobación. Una referencia sellada no tiene ruta de edición. Reintentar una operación no permite cambiarla.

La respuesta de `/reference` es `SourceReference` (**201** nueva; **200** repetición), no `Preparation`. Después deben consultarse preparación y caso para recuperar su estado y versiones. `SourceReference.preparationRevisionId` identifica la revisión de **fuente** que se inspeccionó; la revisión principal de `Preparation` puede cambiar al generar el candidato. Esa diferencia es esperada y no debe corregirse desde Vue.

Si la generación del candidato falla después de sellar, la referencia permanece guardada. Una respuesta de error no demuestra que nada se haya persistido: primero se recupera el estado y, cuando corresponda, se solicita `/retry`.

## Candidato y control de calidad

Estados de preparación: **processing → awaiting_reference → processing → ready**, con **failed** ante fallos de preparación. Una referencia no resuelta permanece en `awaiting_reference` con `sourceReference` presente; no debe ofrecerse una nueva anotación sobre la misma fuente.

`ready` significa candidato disponible, no aprobación. `cropGeometry.nativeHalfBox` usa el orden público **[x0,y0,x1,y1]**, con límites superiores exclusivos, en la mitad nativa. `previewShapes.crop_a` y `crop_b` son `[alto, ancho]`. El servidor convierte expresamente el orden interno del científico; el cliente no debe inferirlo ni recalcular la caja.

La solicitud de revisión contiene `preparationRevisionId`, `confirmedSide`, `decision`, `confirmedCheckIds`, `reason` y `observation`. En esta fase **confirmedCheckIds debe ser []**: los controles reales son la observación congelada completa. `reviewContext.checks` también está vacío. Un identificador de checkbox ficticio no acredita inspección.

`QualityObservation` conserva exactamente estos campos:

- Lateralidad y cobertura: `side_supported`, `coverage_complete`, `lateral_margins_visible`.
- Visibilidad medial, lateral y subcondral en `visibility`: `visible`, `obscured` o `uncertain`.
- `hardware` y `text_or_mask`: `absent`, `present` o `uncertain`.
- `marks`, `mark_effect`, `unclassified_mark_present` y `physical_origin_known`.
- `original_half_inspected`, `both_windows_inspected`, `outcome_blinded` y `note`.
- `ruler_outside_bone` y `ruler_locations_xy`, en coordenadas enteras de la mitad nativa.

El cliente no envía `knee_alias`; el servidor establece la identidad. El evaluador compartido comprueba además coherencia de marcas, referencia central, posiciones de regla e incertidumbre. Origen físico desconocido por sí solo no implica exclusión; defectos o dudas bloqueantes se conservan. La interfaz no inventa umbrales nuevos ni diagnostica artrosis.

**Aceptar** requiere observación completa y `assessment.accepted=true` del evaluador congelado. Si no se cumple, devuelve **409 QC_NOT_ACCEPTABLE** y no guarda una aceptación. **Rechazar** exige motivo y puede llevar `observation=null`, con `assessment=null`; es un rechazo administrativo y no fabrica una evaluación científica. Cuando se proporciona observación, queda registrada su evaluación aunque la decisión final sea rechazo.

Cada decisión es inmutable. Otra revisión crea otro registro y actualiza la referencia activa. Identidad de imagen, lado, revisor, política y fechas proceden del servidor. `QualityAssessment` describe aceptación técnica y motivos; no certifica precisión anatómica, seguridad frente a atajos ni utilidad clínica.

## Sesión, propiedad y archivos privados

GET `/session` entrega contexto anónimo y CSRF sin conceder acceso a casos. Login rota identificador y token. Toda mutación exige cookie, Origin permitido y **X-CSRF-Token**. En producción, `kneeoai_session` es HttpOnly, Secure, SameSite=Lax y Path=/api/v1; el desarrollo loopback usa su excepción documentada. El token CSRF permanece en memoria.

Los recursos se limitan al propietario. Ser administrador no concede acceso clínico transversal. Recurso ajeno e inexistente reciben el mismo **404**. Proteger una ruta Vue o conocer una URL de imagen no sustituye la autorización del servidor.

Todas las respuestas privadas llevan **Cache-Control: no-store**. La futura PWA no deberá almacenar casos, imágenes, referencias, decisiones ni resultados. Las vistas son URL relativas del mismo origen; no se publican rutas de disco ni DICOM. `original` y `crop` son alias de `half_a` y `crop_a`.

La consulta de vista exige `revision` igual a la revisión actual de esa preparación. Una revisión anterior del mismo recurso recibe **409**. Una preparación histórica del caso sigue siendo visible para su propietario mediante su propia revisión actual; `isCurrent=false` impide modificarla. El cambio de revisión no convierte una URL previa en autorización para mostrar otra imagen silenciosamente.

## Versiones, reutilización e idempotencia

ETag es la versión HTTP de la representación; las revisiones basales y de preparación identifican entradas persistidas. Se conserva el ETag completo, incluidas comillas. No se aceptan listas, comodines ni valores débiles.

- PATCH y carga utilizan el **ETag del caso**.
- Referencia, reintento y QC utilizan el **ETag de preparación**.
- Falta versión: **428**. Versión discrepante: **412**, sin sobrescribir cambios ajenos.
- Referencia, reintento, QC y carga exigen **Idempotency-Key**; falta clave: **428 IDEMPOTENCY_REQUIRED**.
- Misma clave y solicitud aceptada recuperan el recurso con **200**, antes de exigir un ETag actualizado. Otra carga con esa clave produce **409**. La asociación persiste tras reiniciar.

Crear caso con la misma cuenta, clave y basales devuelve el mismo caso actual sin reaplicar datos. Cargar mismos bytes/caso/lado con una nueva intención reutiliza y activa la preparación existente; una repetición de la clave anterior únicamente recupera su recurso, que puede ser histórico. Deben comprobarse `isCurrent` y `reused`. El servidor conserva las activaciones y decisiones previas.

Cambiar edad, sexo o IMC conserva el QC ligado a la misma imagen y lado; cambia la revisión basal de una futura predicción. Cambiar lado retira las referencias activas del caso y conserva el historial. Una nueva imagen requiere su preparación y controles. Una reutilización solo recupera aceptación compatible con la política y procedimiento vigentes.

`/retry` recibe **{}** y solo actúa sobre una preparación actual `failed`. Reutiliza fuente y referencia; una referencia no resuelta no permite calcular. Una nueva intención devuelve 202 y su repetición 200 sin relanzar. Otro fallo requiere recuperar estado y decidir explícitamente una intención nueva. El transporte no reintenta mutaciones automáticamente.

La carga y el sellado pueden haber quedado guardados antes de un fallo del procesador. Por eso una respuesta422/503 de ese tramo puede incluir `Location` y dejar un recurso `failed` recuperable. No se supone que la operación no ocurrió ni se cambia la clave automáticamente tras perder una respuesta.

## Frontera con la predicción futura

La preparación utiliza el localizador anatómico congelado. El motor **predictivo** continúa `not_configured`; solicitar cálculo devuelve **503 ENGINE_NOT_AVAILABLE** tras autenticar y comprobar propiedad. En esta fase no procesa parámetros del paquete, registra ejecución, calibra ni emite probabilidad.

`PredictionRequest`, `InputSnapshot`, `EngineVersion`, `Execution` y `PredictionResult` se conservan como esquemas futuros. El POST de ejecución guarda su contrato proyectado en `x-future-parameters`, `x-future-requestBody` y `x-future-responses`; no declara éxito 200/202 implementado. GET de ejecución conserva `x-implementation-status=planned`.

La futura instantánea deberá fijar revisión basal, preparación, QC y paquete formalizado. El cliente no elegirá pesos, calibradores ni umbrales. La descripción futura mantiene horizonte de 48 meses, probabilidad calibrada y clasificación autorizadas, sin fijar aquí valores ni afirmar que la calibración está resuelta. No se combina la validación por pliegues por iniciativa de la aplicación.

## Errores y verificación

JSON directo, sin wrapper `data`. Los errores tienen `error.code`, `message`, `requestId`, `fields` y `retryable`; cada campo de validación incluye `path`, `code` y `message`. Las respuestas HTTP incluyen **X-Request-Id**. Una preparación puede conservar además un error persistido de su operación anterior. No se entregan trazas, credenciales, rutas privadas ni datos completos del caso.

Estados relevantes: 401 acceso requerido; 403 Origin/CSRF o permiso; 404 no accesible; 409 estado/revisión lógica/idempotencia; 412 versión HTTP discrepante; 422 entrada u observación inválida; 428 precondición ausente; 413 tamaño; 415 tipo de carga; 503 componente no disponible. `retryable` informa, no ordena reintentar automáticamente.

Desde `app/frontend`, `npm run test:contracts` comprueba OpenAPI 3.1, referencias internas y ejemplos con JSON Schema. La validación del contrato no sustituye las pruebas de propiedad, transacciones, almacenamiento ni del procedimiento científico. Las pruebas de software utilizan DICOM y observaciones sintéticos; no acreditan calidad anatómica de imágenes reales ni utilidad clínica del predictor.

Las reglas científicas se reutilizan de `src/knee/roi_contact_confirmation.py`, `roi_routine_contract.py`, `roi_confirmation_v2.py` y componentes congelados asociados. Este contrato no cambia esos archivos ni las reglas de la tesis. Referencias técnicas: [OpenAPI 3.1](https://spec.openapis.org/oas/v3.1.0) y [sesiones mediante cookies](https://swagger.io/docs/specification/v3_0/authentication/cookie-authentication/).
