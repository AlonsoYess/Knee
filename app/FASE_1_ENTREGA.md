# Cierre de la fase 1 del prototipo

Fecha: 7 de noviembre de 2026. Versión: 0.8.0. Rama: `codex/prototipo-fase1`.

La fase 1 deja una versión reproducible del frontend y define la comunicación prevista con FastAPI. Las pantallas mantienen el recorrido ficticio y el guardado local; el transporte HTTP se prueba por separado con respuestas controladas. El contrato es una especificación para implementar el servidor, no acredita que exista autenticación o persistencia de servidor.

**Entrega**

- Frontend Vue completo, dependencias fijadas en `package-lock.json` y versión de Node de referencia en `.node-version`.
- Módulos ficticios movidos de `src/data/` a `src/demo/`, sin cambiar claves, versión o contenido del almacenamiento del navegador. Las exclusiones generales de datos privados permanecen intactas.
- `contracts/openapi.json`: API propuesta bajo `/api/v1`, con 14 operaciones y 33 esquemas; guía en `contracts/README.md`.
- Sesión y CSRF previstos; evaluaciones con control de versión; preparación y QC vinculados a su identidad; solicitudes de cálculo idempotentes sobre instantáneas y resultados versionados.
- Transporte JSON aislado en `frontend/src/api/http.ts`: rutas internas, cookies del mismo origen, sin caché ni reintentos automáticos, errores explícitos, conflictos y cancelación. No valida los datos particulares de cada respuesta en ejecución ni implementa todavía la carga multipart.
- Proxy de desarrollo hacia el backend local previsto, con configuración de ejemplo sin secretos. La publicación futura necesitará su propio enrutamiento.
- Validación del contrato y sus ejemplos; pruebas del recorrido y del transporte; flujo de GitHub Actions preparado para contrato, compilación y Chromium. Su ejecución remota se verifica por separado del resultado local.

**Comprobaciones realizadas**

| Comprobación | Resultado |
| --- | --- |
| Instalación desde una exportación de los archivos preparados para Git, sin módulos locales previos | Correcta con `npm ci --ignore-scripts --no-audit --no-fund`. |
| Compilación de producción y comprobación TypeScript en esa copia | Correctas. |
| Recorrido y transporte en Chromium independiente del navegador del investigador | 57 aprobadas, cero fallos, cero omitidas y cero inestables. |
| Validez OpenAPI 3.1 y referencias internas | Correcta. |
| Ejemplos ficticios contra los esquemas JSON | 98 comprobaciones correctas; no representan 98 casos reales ni únicos. |
| Inclusión de código y exclusión de dependencias y resultados generados | Fuente ficticio visible para Git; `node_modules`, `dist` y resultados de pruebas excluidos. |

Las pruebas de comunicación cubren cabeceras de sesión, CSRF, versiones e idempotencia, respuestas sin cuerpo, errores HTTP y de red, HTML inesperado, JSON inválido, rutas externas o codificadas ambiguamente, errores recibidos bajo HTTP exitoso y cancelación durante la lectura. Ninguna respuesta de error se convierte en una predicción o guardado ficticio.

**Límite del cierre**

Esta entrega no ejecuta el procedimiento científico, no carga modelos y no consulta la prueba reservada. Tampoco implementa todavía FastAPI, SQLite, cuentas reales, carga de DICOM, inferencia o instalación PWA. Los campos y estados contractuales no añaden reglas clínicas; las ausencias, incluido un IMC nulo, se resolverán según el paquete científico vigente.

La fase 2 implementará sesiones, permisos y guardado real con el contrato propuesto. La revisión de esquema en ejecución, la carga multipart y las respuestas binarias se incorporarán al conectar las operaciones correspondientes. El resultado predictivo exigirá posteriormente el paquete final, QC vigente y comprobación de equivalencia con el flujo experimental.
