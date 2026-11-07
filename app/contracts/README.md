# Contrato propuesto Vue–FastAPI, v1

Estado: **sesión y datos basales implementados en fase 2; preparación y resultados futuros**. Fecha: 7 de noviembre de 2026.

[openapi.json](openapi.json) describe OpenAPI 3.1.0 con esquemas JSON Schema 2020-12. Se usa JSON para leerlo sin instalar un procesador YAML. La versión es **1.0.0-phase2**: no es una versión de modelo ni una aprobación de despliegue. IDs, datos basales y trabajos de ejemplo son completamente ficticios. No hay ejemplos de porcentajes predictivos ni credenciales operativas. Cada operación declara su estado mediante `x-implementation-status`.

La implementación de sesiones y evaluaciones está en `app/backend`; Vue utiliza estas operaciones. Las cuentas se administran mediante CLI, fuera de este contrato web. La preparación de imágenes, QC y resultados siguen previstos para fases posteriores; no se cargan pesos ni se ejecuta inferencia. La solicitud de cálculo solo implementa el bloqueo 503 después de comprobar el acceso. La simulación SVG de la interfaz no genera una preparación o decisión QC en este contrato.

## Cuatro contratos funcionales

1. **Caso:** Evaluation identifica datos basales y una revisión persistida e inmutable. Los borradores conservan ausencias como null. El servidor asigna código, propietario, revisión y fecha. No recibe KL, desenlace ni particiones.
2. **Preparación y QC:** Preparation vincula imagen, versión, lado, procesamiento y política. QualityReview conserva la decisión y su identidad exacta. Revisor y fechas los determina el servidor.
3. **Solicitud:** PredictionRequest referencia caso, preparación y revisión aceptada vigentes. El servidor construye InputSnapshot y fija la versión del paquete. El cliente no elige pesos, modelos, calibradores o umbrales.
4. **Ejecución y resultado:** Execution conserva estado, instantánea, paquete y eventual PredictionResult. Solo succeeded contiene probabilidad. La ausencia del motor produce 503 ENGINE_NOT_AVAILABLE, sin resultado sustituto.

Los campos son age, sex, bmi y side. La propiedad knee de la demostración se adaptará a side en la futura integración; no es un cambio científico.

**Ausencias:** null no implica exclusión clínica automática. La aptitud para inferencia depende del contrato del paquete congelado. Si este admite IMC ausente e imputación, el servidor conserva null en la instantánea y aplica exclusivamente el estado de transformación ya ajustado; no calcula una nueva media, mediana o regla. La API no introduce casos completos, rangos clínicos nuevos ni nuevos predictores. Edad entera positiva e IMC positivo cuando se proporcionan son validaciones de formato, no criterios de elegibilidad.

## Operaciones

Todas las rutas parten de **/api/v1** y del mismo origen que la SPA.

| Método y ruta | Función |
| --- | --- |
| GET /session | Consultar sesión o crear contexto anónimo y entregar CSRF. |
| POST /session | Autenticar cuenta administrada y rotar sesión/token. |
| DELETE /session | Revocar sesión y expirar cookie. |
| GET /evaluations | Listar casos accesibles con cursor. |
| POST /evaluations | Crear borrador. |
| GET /evaluations/{evaluationId} | Leer caso y ETag. |
| PATCH /evaluations/{evaluationId} | Guardar cambios basales. |
| POST /evaluations/{evaluationId}/preparations | Recibir DICOM e iniciar preparación persistida. |
| GET /preparations/{preparationId} | Consultar preparación y contexto de revisión. |
| GET /preparations/{preparationId}/views/{view}?revision=... | Obtener PNG autenticado del original o recorte. |
| POST /preparations/{preparationId}/reviews | Registrar decisión sobre una preparación exacta. |
| GET /engine-status | Consultar disponibilidad técnica del paquete formalizado. |
| POST /evaluations/{evaluationId}/executions | Registrar cálculo idempotente de una revisión exacta. |
| GET /executions/{executionId} | Consultar estado y resultado sin ejecutar otra vez. |

Se mantienen fuera de este contrato administración completa de cuentas, recuperación de contraseña, corrección geométrica manual, selección de modelos, entrenamiento, publicación y sincronización. Una corrección geométrica que permita el flujo vigente deberá crear otra revisión de preparación; este documento no autoriza reglas nuevas para corregir imágenes.

## Sesión, permisos y CSRF

En fase 2, POST /evaluations admite Idempotency-Key opcional y la interfaz siempre lo envía. Una nueva intención crea el caso con 201; la repetición exacta de una creación aceptada devuelve 200 con el mismo caso y su estado actual, sin aplicar nuevamente el cuerpo. Otra carga con esa clave devuelve 409. La asociación se registra en la misma transacción que el caso y persiste tras un reinicio.

- GET /session permite el primer acceso sin credenciales y emite cookie de contexto anónimo más csrfToken. Una sesión válida devuelve usuario y vencimiento. El contexto anónimo no permite acceder a evaluaciones.
- Login exige esa cookie y **X-CSRF-Token**. Tras autenticar se rotan identificador y token. Toda mutación, incluido logout, exige sesión y CSRF. La marca actual de sessionStorage no participa en la autorización del servidor.
- En HTTPS, **kneeoai_session** utiliza HttpOnly, Secure, SameSite=Lax y Path=/api/v1. Cualquier excepción técnica del transporte local se documentará por separado y no se trasladará a publicación. JavaScript no lee la cookie; el token CSRF queda en memoria y se recupera mediante la sesión al recargar.
- El servidor verifica sesión, origen/CSRF y permisos en cada operación. Proteger una ruta Vue no sustituye esos controles.
- Los recursos se limitan al propietario o a un permiso explícito. Ser administrador no concede acceso clínico transversal implícito. Un recurso inexistente o ajeno recibe el mismo 404; 403 indica operación prohibida o CSRF inválido.
- Respuestas de sesión, datos, errores y vistas privadas llevan **Cache-Control: no-store**. La futura PWA no debe almacenar esas respuestas, imágenes ni resultados.

## Versiones e invariantes

**ETag** es la versión HTTP opaca de una representación. **currentRevisionId** y **preparationRevisionId** son identidades persistidas e inmutables de las entradas. No son intercambiables.

- Se conserva el ETag completo, incluidas comillas, en **If-Match**. Solo se admite un valor fuerte, sin comodín, lista ni prefijo débil W/.
- PATCH evaluación, carga de preparación y solicitud de cálculo utilizan el ETag de evaluación. Enviar QC utiliza el ETag de preparación.
- Falta If-Match: **428 VERSION_REQUIRED**. No coincide: **412 VERSION_CONFLICT**, sin escritura. **409** corresponde a estado incompatible, revisión no vigente o idempotencia. El cliente recupera la versión actual y conserva la edición pendiente; no sobrescribe automáticamente.
- Cambiar entradas relevantes crea otra revisión y deja sin vigencia resultados anteriores. El progreso de un trabajo no modifica por sí solo su revisión de entrada.
- Cambiar imagen, lado o geometría invalida confirmación y QC vinculados a esa identidad. Cambiar solo edad, sexo o IMC invalida la predicción, sin inventar un defecto radiográfico ni borrar la evidencia histórica de revisión.
- Preparación ready no equivale a QC aprobado. Para aceptar se verifican permisos, identidad exacta, política, controles requeridos y ausencia de impedimentos del flujo científico. Los checks que muestra Vue provienen de esa política.
- Un accepted del cliente no fuerza aprobación. Rechazar exige motivo. Una nueva decisión conserva el registro previo, actualiza la referencia activa del caso e invalida sus resultados.
- Tras crear preparación o guardar QC se vuelven a consultar caso y preparación para recuperar sus versiones. El ETag de un recurso no se usa en otro.
- Preparation.activeReview devuelve la decisión persistida vigente o null. Permite recuperar la revisión al recargar sin fabricar una aceptación desde el navegador; una decisión desactualizada se conserva en el historial del servidor, pero no se devuelve como activa.
- La aceptación de inferencia valida referencias vigentes, crea la instantánea y fija el paquete formalizado en una transacción. El cálculo no vuelve a leer el borrador mutable ni combina pliegues por iniciativa de la aplicación.
- **isCurrentInput=false** identifica ejecución histórica. No se muestra como resultado de la edición actual y una respuesta tardía no reemplaza una revisión nueva.
- La revisión declarada en la solicitud, la instantánea y el resultado deben coincidir; lo mismo aplica a executionId, identidad de imagen/lado/preparación y versiones del paquete. Estas relaciones se verifican en servidor aunque un esquema individual no pueda compararlas entre recursos.

## Idempotencia y estados de cálculo

POST de ejecuciones exige **Idempotency-Key**. Se genera una clave por intención, sin información clínica.

1. Antes de devolver una repetición se verifican de nuevo sesión, CSRF y permisos.
2. La clave se limita a usuario, operación y evaluación. Clave aceptada y mismo cuerpo canónico: 200 con la ejecución original, sin otro cálculo. El ETag antiguo no impide recuperar ese trabajo. Otro cuerpo con esa clave: 409 IDEMPOTENCY_CONFLICT.
3. Clave nueva: validar If-Match, revisión vigente, entradas conforme al paquete, QC y motor. Trabajo, instantánea y asociación de clave se registran atómicamente antes del 202. Location identifica la ejecución.
4. Un 503 previo a la aceptación no crea trabajo, resultado o asociación exitosa. Si se corta la red después del envío no se supone que el servidor rechazó: se recupera la decisión repitiendo exactamente clave y cuerpo.
5. La asociación se conserva durante toda la retención del trabajo. No se elimina mientras pueda recuperarse o ejecutarse. El vencimiento de un tiempo en el navegador no relanza una operación.

Estados: **queued → running → succeeded**, o **failed / interrupted**. El servidor debe reconciliar interrupciones; no deja trabajos eternamente running. queued y running tienen result y error null. failed e interrupted incluyen error y fecha final. succeeded contiene resultado inmutable y no error.

Un reintento explícito tras fallo usa otra intención y clave, conserva el fallo previo y verifica la revisión actual. El transporte no reintenta mutaciones automáticamente. Si cambia el paquete activo, repetir una clave aceptada devuelve su trabajo y paquete originales, nunca otro cálculo silencioso.

## Resultado y motor ausente

GET /engine-status distingue not_configured, unavailable y ready. Es disponibilidad técnica; no demuestra utilidad clínica ni sustituye la comprobación al solicitar cálculo.

El resultado futuro incluye versión de esquema, ejecución, revisión de entrada, horizonte de 48 meses, probabilidad calibrada, clasificación, umbral y versiones de paquete, transformaciones y calibrador. No se fija un umbral numérico aquí ni se recibe desde el cliente. La clasificación debe concordar con el paquete autorizado. No se generan diagnósticos ni recomendaciones terapéuticas.

Sin paquete, el caso puede guardarse, pero no se presenta probabilidad. 503 ENGINE_NOT_AVAILABLE deja intacto el caso. El esquema de éxito describe una comunicación futura; no acredita ejecución ni que la calibración esté resuelta.

## Errores uniformes

JSON directo, sin wrapper data. Un error tiene esta estructura:

    {
      "error": {
        "code": "VERSION_CONFLICT",
        "message": "El caso cambió. Recupera su versión actual antes de guardar.",
        "requestId": "req-fixture-001",
        "fields": [],
        "retryable": false
      }
    }

fields contiene objetos con path, code y message. requestId coincide con **X-Request-Id**. retryable es informativo, sin reintentos automáticos. No se exponen trazas, rutas, credenciales ni datos completos del caso.

Estados previstos: 401 acceso requerido; 403 permiso/CSRF; 404 recurso no accesible; 409 estado o idempotencia; 412 versión discrepante; 428 versión omitida; 422 entrada inválida; 413 archivo demasiado grande; 415 tipo de solicitud; 503 componente necesario no disponible. Los errores inesperados conservan el mismo sobre.

## Verificación y límites

El JSON debe validar como OpenAPI 3.1: referencias locales resolubles, operationId únicos, parámetros declarados y ejemplos ficticios compatibles con sus esquemas. Mutaciones requieren cookie y CSRF; operaciones versionadas requieren If-Match; solicitud de cálculo requiere idempotencia.

Las pruebas de fase 2 verifican sesiones, permisos entre cuentas, ETag, creación idempotente, transacciones y recuperación con datos ficticios. La validez del archivo por sí sola no acredita esos controles: su implementación se comprueba en la suite del backend y del navegador. Las pruebas de preparación/QC e inferencia se incorporarán al implementar esas operaciones.

Fuentes: [OpenAPI 3.1.0](https://spec.openapis.org/oas/v3.1.0), [cookies en OpenAPI](https://swagger.io/docs/specification/v3_0/authentication/cookie-authentication/) y los documentos locales docs/00_ALCANCE_Y_TRAZABILIDAD.md, docs/04_ARQUITECTURA_REPRODUCIBLE.md y app/PLAN_CIERRE_PROTOTIPO.md. No se modificaron esas fuentes.
