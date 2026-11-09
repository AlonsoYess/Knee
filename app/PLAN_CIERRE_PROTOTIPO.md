# Plan de cierre del prototipo KneeOAI para la sustentación

Fecha de actualización: 9 de noviembre de 2026. Estado: fases 1 a 4 completadas en v0.8, v0.9, v0.10 y v0.11. Publicación e integración predictiva conservan sus condiciones de cierre. Registros: `FASE_1_ENTREGA.md`, `FASE_2_ENTREGA.md`, `FASE_3_ENTREGA.md` y `FASE_4_ENTREGA.md`.

Preparación de fase 5 en v0.12: paquete y configuración listos; construcción Linux, HTTPS Docker, recreación y recuperación en otro volumen comprobadas. No hay contratación ni publicación. Ver `FASE_5_PREPARACION.md` y `FASE_5_VERIFICACION_DOCKER.md`.

El objetivo es terminar la aplicación mientras se completa el trabajo científico, de modo que la última etapa consista en conectar y verificar el paquete predictivo aprobado. Habrá dos cierres: **aplicación preparada para integración** y **prototipo predictivo listo para sustentación**. El segundo requiere evidencia de funcionamiento con el paquete final; finalizar un entrenamiento no basta para declararlo alcanzado.

El trabajo de producto conserva las reglas científicas vigentes, los entrenamientos existentes y la prueba reservada. La revisión o aceptación del asesor se mantiene separada del cierre técnico de la aplicación.

**Punto de partida al definir el plan**

La interfaz v0.7 contiene acceso, listado, datos basales, radiografía y lado, revisión del recorte y resumen. Usa Vue, TypeScript y Vite. El guardado conserva evaluaciones ficticias en el navegador. La última entrega documentada registra compilación correcta y 43 pruebas de interacción aprobadas; no se volvieron a ejecutar para redactar este plan.

En ese punto inicial el acceso era simulado y no existían API, autenticación de servidor, base central ni carga real de DICOM. Las fases 2 y 3 incorporan esas funciones. PWA, despliegue y conexión predictiva mantienen sus fases propias.

Al iniciar la fase 1, el frontend estaba presente localmente y la regla general `data/` ocultaba `app/frontend/src/data/`. En v0.8 se resolvió moviendo esos módulos ficticios a `app/frontend/src/demo/`, conservando sus claves y versiones de almacenamiento. La entrega incorpora el frontend completo y su contrato de API, sin relajar las exclusiones de datos privados.

**Arquitectura propuesta para la primera entrega**

| Componente | Decisión propuesta | Responsabilidad |
| --- | --- | --- |
| Pantallas | Vue 3 y TypeScript actuales | Interacción del médico, validaciones de presentación y estados del recorrido. |
| Servidor | FastAPI modular, en el mismo repositorio | Sesiones, permisos, evaluaciones, archivos y coordinación de inferencia. |
| Datos | SQLite, detrás de FastAPI | Usuarios, sesiones, evaluaciones, revisiones y referencias a resultados. |
| Archivos | Almacenamiento privado persistente | Imágenes autorizadas, recortes y artefactos necesarios, fuera del directorio público. |
| Motor científico | Adaptador a proceso separado con `src/knee` | Reutilizar el procedimiento y paquete congelados, sin duplicarlos en Vue ni instalar dependencias científicas en FastAPI. |
| Distribución | Configuración reproducible y contenedor Docker | Ejecutar la misma versión en local y en un alojamiento compatible. |
| Acceso publicado | Dirección gratuita del proveedor con HTTPS; interfaz y API bajo el mismo origen | Empezar sin comprar un dominio y conservar una estructura sencilla. |

SQLite es adecuado para esta demostración acotada con un servidor y disco persistente. El navegador se comunica con FastAPI, nunca abre el archivo de base de datos. Tener dominio no exige PostgreSQL. Si se requieren varias instancias del servidor o muchas escrituras simultáneas, se resolverá PostgreSQL antes de implementar la persistencia; una migración posterior requeriría trabajo y pruebas propios. La base activa no se alojará en una carpeta sincronizada de Drive. [Criterios oficiales de SQLite](https://sqlite.org/whentouse.html).

Se supone una demostración con cuentas controladas y pocas sesiones simultáneas. El investigador eligió una dirección gratuita inicialmente y un presupuesto máximo de US$15 al mes para alojamiento. Quedan pendientes el proveedor y la dirección disponible; no se contempla comprar un dominio en esta primera entrega. El límite se aplicará al costo total de la infraestructura contratada, considerando servidor, almacenamiento, copias y cargos aplicables. La arquitectura no presupone un sistema hospitalario ni incorpora registro público de cuentas.

**Fases y condiciones para declararlas terminadas**

| Fase | Entrega concreta | Evidencia de cierre | Dependencia |
| --- | --- | --- | --- |
| 1. Base reproducible | Completada en v0.8: fuente completo, exclusiones resueltas, dependencias fijadas, contrato y ejecución documentada. | Copia limpia instalada y compilada; 57 pruebas aprobadas, OpenAPI y 98 ejemplos ficticios validados. Ver `FASE_1_ENTREGA.md`. | Cerrada técnicamente para su alcance. |
| 2. Acceso y guardado reales | Completada en v0.9: FastAPI, usuarios, sesiones, permisos y datos basales persistentes en SQLite. | Copia limpia: 62 pruebas de backend y 29 de navegador, incluidos aislamiento, reinicio, conflicto, respuesta perdida y recuperación. Ver `FASE_2_ENTREGA.md`. | Cerrada para acceso y basales; imagen/QC reales siguen en fase 3. |
| 3. Recorrido funcional completo | Completada en v0.10: DICOM privado, referencia sellada, preparación, QC y resumen persistentes; predicción bloqueada. | 110 pruebas de backend, 39 de navegador y 19 comprobaciones sintéticas del puente científico; respaldo completo y recuperación. Ver `FASE_3_ENTREGA.md`. | Cerrada para el recorrido de aplicación; no acredita integración predictiva ni validación clínica. |
| 4. PWA y ejecución local | Completada en v0.11: instalación, iconos, actualización explícita, estado de conexión y arranque unificado. | 185 pruebas de backend, 50 de navegador y 13 del núcleo PWA; instalación/cierre/reapertura real en Edge aislado y campos pendientes protegidos. Ver `FASE_4_ENTREGA.md`. | Cerrada para uso local; servidor necesario y predicción todavía bloqueada. |
| 5. Publicación y recuperación | Preparación v0.12 con paquete aislado y producción estricta; construcción Linux, HTTPS y recuperación Docker comprobadas el 9 de noviembre. | 9 pruebas de empaquetado y 23 comprobaciones HTTPS Docker correctas; recreación real y restauración desde respaldo externo a otro volumen. Evidencia previa: 242 pruebas backend y 23 HTTPS nativas. | Pendientes selección y aprobación de proveedor, recursos y costo, acceso público y persistencia del proveedor. No está cerrada la publicación. |
| 6. Integración científica | Paso 1 técnico preparado: FastAPI verifica los cinco pliegues clínicos guardados mediante un proceso separado; predicción clínica bloqueada. | 330 pruebas backend, carga real y quince logits ficticios coincidentes con las funciones originales. Ver `FASE_6_PASO_1_CONEXION_LOCAL.md`. | Pendientes paquete final formalizado, conexión radiográfica/multimodal, ejecución persistida, QC vigente y presentación de resultados. No está cerrada la integración predictiva. |
| 7. Entrega al jurado | Versión congelada, manuales, ensayo con el médico, guion y alternativa local. | Demostración completa repetida en las condiciones de la sustentación y registro de incidencias resueltas. | Fases anteriores y disponibilidad del paquete final. |

La numeración pertenece al prototipo y no sustituye las fases científicas ni las observaciones del jurado. Cada fase se cierra con su versión, pruebas pertinentes, pendientes explícitos y una demostración breve. Solo se reabre por un cambio o un fallo que afecte su aceptación.

**Detalle de la fase 1**

- Incorporar únicamente los cambios de aplicación; mantener separados los cambios científicos que ya existan en el repositorio.
- Corregir la exclusión de código ficticio mediante una regla específica o un nombre de carpeta inequívoco.
- Documentar configuración de ejemplo, instalación, arranque, pruebas y versión entregada. Mantener credenciales, bases de datos, imágenes y pesos fuera de Git.
- Definir cuatro contratos: entrada del caso; preparación y QC versionados; solicitud de predicción sobre una revisión vigente; resultado o error con versión del motor.
- Preparar verificaciones automáticas de la aplicación con datos sintéticos. La publicación de la versión se realiza después de comprobar el conjunto concreto de archivos.

**Detalle de la fase 2**

La fase 2 incluye usuarios, sesiones, evaluaciones basales, sus revisiones y asociaciones idempotentes de creación. La fase 3 añade referencias a imágenes y decisiones QC, con archivos fuera de la base y referencias comprobables. Ejecuciones y resultados predictivos se incorporarán durante la integración científica. Las actualizaciones basales tienen control de versión para detectar dos ediciones del mismo caso.

Las cuentas serán administradas. Se implementarán cambio de contraseña y recuperación mediante el administrador, sin añadir una dependencia de correo a esta primera entrega. Las contraseñas usarán un hash adecuado y las sesiones tendrán vencimiento y revocación en el servidor. La configuración publicada utilizará cookies protegidas y defensa CSRF; los permisos se comprobarán en cada operación de la API. [Contraseñas](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html) y [sesiones](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html) en OWASP.

El cierre exige probar aislamiento entre dos usuarios, sesión expirada, cierre de sesión, edición concurrente, pérdida de conexión y reinicio. Se definirá expresamente cómo conservar o importar los borradores ficticios del navegador antes de sustituir su almacenamiento; no se borrarán silenciosamente.

**Detalle de la fase 3**

El recorrido será: acceso → evaluaciones → datos basales → radiografía y lado → revisión de calidad → resumen → cálculo → resultado. Se conservarán los estados de borrador y la siguiente acción clara para el médico.

La fase 3 sustituye el catálogo SVG por carga privada de DICOM, ventanas nativas autenticadas y una referencia visual inmutable antes de la localización. El procedimiento compartido congelado prepara el recorte sin emplear esa referencia como entrada, y comprueba la observación completa antes de aceptar. Se conserva la fuente, el recorte, el QC y la trazabilidad tras recarga/reinicio; el respaldo incluye base y archivos. Las pruebas de aplicación usan píxeles y decisiones sintéticos; se distingue su alcance de la comprobación del puente científico.

Los estados necesarios son carga en curso, archivo inválido, preparación pendiente, revisión pendiente, rechazo con motivo, revisión desactualizada, motor no disponible, cálculo en curso, error recuperable y resultado disponible. Cambiar una entrada relevante invalidará la revisión o el resultado que corresponda; nunca se mostrará un resultado antiguo como si perteneciera a la edición nueva. Los cálculos llevarán un identificador y un estado persistido; reintentar no debe duplicar resultados. Una interrupción del servidor debe dejar un estado recuperable, no un cálculo eterno.

La pantalla de resultado y su vista imprimible pueden prepararse con marcadores claramente ficticios en el entorno de desarrollo. La versión demostrable sin paquete final mantendrá la predicción bloqueada. El resultado definitivo incluirá probabilidad calibrada de progresión estructural a 48 meses, clasificación con umbral congelado, versión y limitaciones. Las entradas clínicas seguirán siendo las autorizadas; la aplicación no añadirá un estimador KL, nuevas reglas clínicas ni recomendaciones terapéuticas.

**Detalle de las fases 4 y 5**

La PWA tendrá manifiesto, nombre, iconos, navegación y actualización verificadas. Se probará instalación y reapertura en Edge o Chrome en Windows y el recorrido adaptable en móvil. La instalación requiere HTTPS en la web o un origen local admitido; el comportamiento sin conexión se implementa y comprueba por separado. [Instalación de PWA en MDN](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Making_PWAs_installable).

La caché se limitará inicialmente a recursos estáticos públicos. No almacenará respuestas autenticadas, imágenes ni resultados. Si no hay conexión con el servidor se mostrará la limitación y se impedirá confirmar operaciones no realizadas. Una actualización pendiente no recargará automáticamente una edición sin guardar.

La PWA publicada seguirá necesitando acceso a su servidor. La alternativa local incluirá también el backend y, cuando esté autorizado, el paquete científico y sus dependencias. Su funcionamiento sin Internet deberá ensayarse: instalar un icono no lo garantiza. Las bases local y publicada serán independientes; esta entrega no incorpora sincronización entre ambas.

El dominio es la dirección de acceso; el alojamiento ejecuta la aplicación; el disco persistente conserva sus archivos. Render es un candidato a evaluar por su soporte de dominios y certificados HTTPS, no una contratación decidida. Su almacenamiento de servicio es efímero por defecto y requiere disco persistente de pago para conservar archivos locales entre reinicios y despliegues. [Dominios en Render](https://render.com/docs/custom-domains) y [discos persistentes](https://render.com/docs/disks).

Antes de contratar se presentarán la dirección gratuita disponible, la cuenta a cargo del investigador y el costo mensual total, dentro de US$15. La dirección gratuita no implica alojamiento gratuito. El tamaño final se elegirá midiendo memoria, tiempo de carga y latencia del paquete necesario. Si el predictor no cabe en ese presupuesto, se conservará el límite y se planteará la demostración predictiva local; la publicación con motor no disponible no acreditará el cierre predictivo en línea. Un aumento de presupuesto o un cambio de alcance requerirá una decisión expresa. Multiplicar procesos puede multiplicar la memoria de los modelos. [Memoria por proceso en FastAPI](https://fastapi.tiangolo.com/deployment/concepts/#memory-per-process).

El despliegue incluirá configuración privada de secretos, límites de carga, permisos de archivos, comprobación de disponibilidad, registro de errores sin contenido privado, versión identificable y retorno a una versión anterior. No se expondrá el servidor de desarrollo de Vite. Se respaldarán de forma coherente base y archivos, con copia fuera del servidor y restauración ensayada; para SQLite se empleará un mecanismo de copia consistente. [API de respaldo de SQLite](https://sqlite.org/backup.html).

La revisión previa al despliegue verificará que la distribución y sus capas no incluyan datos restringidos ni artefactos científicos privados. La demostración publicada empezará con ejemplos ficticios. El alojamiento de pesos o imágenes de investigación requiere la autorización correspondiente y conservación de los controles de acceso existentes.

**Detalle de la fase 6**

El paquete necesario debe identificar selección final, transformaciones clínicas y radiográficas, procesamiento y QC, pesos, calibrador, umbral y versiones compatibles. No se elegirá un pliegue ni se combinarán pesos desde la aplicación por conveniencia. La aplicación consumirá la regla final ya aprobada.

El servidor llamará a `src/knee`. El mismo caso y la misma preparación deberán dar la misma clasificación y una diferencia absoluta de probabilidad no mayor que `1e-5`, conforme al alcance vigente. También se comprobará que un QC ausente o desactualizado bloquee la probabilidad. Estas pruebas de equivalencia utilizarán casos de desarrollo autorizados; no requieren abrir la prueba reservada.

Se medirá el funcionamiento en el equipo y alojamiento previstos. El criterio operativo de duración aceptable se acordará antes del ensayo con el médico y se registrará junto con memoria máxima, arranque y errores. No se alterarán resolución, precisión numérica, combinación de modelos o transformaciones para abaratar el servidor sin evaluar y autorizar su impacto científico.

**Detalle de la fase 7**

La entrega contendrá versión identificable del código, configuración reproducible, guía de instalación y recuperación, manual breve del médico, diagrama de arquitectura, pruebas pertinentes, guion de demostración y grabación de respaldo. Se prepararán ejemplos ficticios o expresamente autorizados; no se usará la prueba reservada como catálogo de demostración.

El ensayo cubrirá ingreso, creación, guardado, recuperación, imagen válida e inválida, revisión, rechazo, cálculo, resultado y recuperación ante una interrupción. Se solicitará al médico una valoración descriptiva de comprensión y facilidad de uso; esa valoración no se presentará como validación clínica del predictor.

**Qué puede cerrarse mientras termina el entrenamiento**

Las fases 1 a 5 y la preparación del material de sustentación pueden avanzar ahora con datos ficticios. La fase 6 y el ensayo predictivo definitivo dependen del paquete científico formalizado. El diseño de pantallas, usuarios, persistencia, instalación y dominio podrá mantenerse si cambian artefactos del motor conservando su contrato. Un cambio aprobado en entradas o procedimiento exigirá revisar su impacto; no se promete que cualquier decisión posterior sea intercambiable sin trabajo.

El trabajo activo continúa en local con la fase 6, por decisión del investigador de aplazar alojamiento y dominio. El paso 1 técnico queda comprobado; la continuación debe verificar los artefactos y requisitos pendientes del paquete final antes de habilitar inferencia y resultados para el médico. La publicación de fase 5 conserva sus comprobaciones y su aprobación de proveedor/costo pendientes. La tesis y las respuestas al jurado se actualizarán por su procedimiento propio, cuando se indiquen; este plan no modifica esos documentos.
