# Cierre de la fase 2 del prototipo

Fecha: 7 de noviembre de 2026. Versión: 0.9.0. Rama: `codex/prototipo-fase2`.

La aplicación utiliza acceso real y conserva los datos basales de las evaluaciones ficticias en SQLite. Vue se comunica con FastAPI mediante la API v1. La contraseña y la sesión simuladas de la entrega anterior ya no conceden acceso.

**Entregado**

- Cuentas administradas por CLI, contraseña con Argon2, sesión opaca con vencimiento y revocación, cookie HttpOnly, comprobación de origen y CSRF, límite de intentos de acceso.
- Cada usuario solo consulta y modifica sus evaluaciones. Un identificador ajeno recibe el mismo 404 que uno inexistente.
- Creación, listado paginado, recuperación y edición de edad, sexo, IMC y lado. Los ausentes se conservan como null; el servidor asigna código, propietario, revisión y fecha.
- Control de edición mediante ETag e If-Match, revisiones inmutables y transacciones SQLite. Una edición antigua no sobrescribe una nueva.
- Creación idempotente: una respuesta perdida puede recuperarse con la misma clave y el mismo cuerpo sin crear otro caso. La asociación persiste tras reiniciar el servicio.
- Validación de respuestas en Vue, conservación de campos ante fallos y descarte de respuestas tardías de una sesión anterior. El cierre de sesión revoca el acceso en el servidor.
- SQLite con versión inicial de esquema, comprobación de integridad, WAL y recuperación a destino nuevo. Restaurar una copia invalida las sesiones anteriores.
- Herramientas locales para iniciar el servidor y preparar una cuenta de demostración con contraseña aleatoria privada. La base, credenciales, clave de sesión y registros están excluidos de Git.

**Verificación desde una copia limpia**

La exportación de los archivos preparados para Git se instaló con un entorno Python nuevo y `npm ci`. El backend y la interfaz de pruebas utilizaron puertos distintos de los del investigador y una base ficticia propia.

| Control | Resultado |
| --- | --- |
| Instalación con dependencias fijadas de backend y frontend | Correcta. |
| TypeScript y compilación de producción | Correctos. |
| Backend con bases SQLite temporales | 62 pruebas aprobadas. |
| Navegador Chromium: transporte e integración con FastAPI real | 29 pruebas aprobadas; cero fallos, omitidas o inestables. |
| OpenAPI y los 33 esquemas | Válidos; 99 comprobaciones de ejemplos ficticios. |
| Presentación de escritorio y móvil | Capturas revisadas; prueba móvil sin desbordamiento horizontal. |
| Acceso local por la dirección de Vue | Inicio de sesión 200, listado 200, motor not_configured y cierre 204. |

Las pruebas cubren propiedad, CSRF, rotación y vencimiento de sesión, edición concurrente real, respuestas inciertas, reintentos sin duplicados, rollback, copias y restauración, fallos de archivo al preparar credenciales y respuestas antiguas tras cambiar de cuenta o salir.

La prueba del motor comprueba el bloqueo; no realiza predicciones. Las cuentas y casos de prueba son ficticios. Las cifras anteriores son comprobaciones de software y no tamaños de muestra de investigación.

La suite antigua del acceso simulado y persistencia del navegador se conserva como historial v0.8. Se reemplaza por las pruebas actuales del servidor y no se suma a estos resultados.

**Qué se conserva y qué sigue pendiente**

SQLite guarda los datos basales y su historial. Las ilustraciones SVG y decisiones sobre sus recortes siguen siendo una simulación temporal: se reinician al recargar, cerrar sesión o cambiar la revisión basal. El servidor mantiene preparationId, reviewId y currentExecutionId en null; no convierte una aceptación de ejemplo en QC científico. Las pantallas lo indican expresamente.

La copia anterior de borradores del navegador permanece intacta y se avisa de su existencia; no se importa ni publica automáticamente. La carga de archivos y el guardado de preparación/QC reales corresponden a la fase 3. La instalación PWA y el alojamiento corresponden a sus siguientes fases. La inferencia continúa bloqueada.

GitHub Actions tiene preparado el flujo de backend, contrato, compilación y navegador. Su ejecución remota debe registrarse separadamente: la cuenta tenía una restricción externa que impedía iniciar trabajos en la fase anterior; no se presume una comprobación remota aprobada.

No se modificaron metodología, entrenamientos, paquetes científicos ni la prueba reservada. La entrega técnica comprende únicamente la aplicación y su flujo de comprobación.
