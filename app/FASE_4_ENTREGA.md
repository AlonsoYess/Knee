# Fase 4 — PWA y arranque local

Interfaz 0.11.0; API 0.4.0; SQLite conserva esquema 2. Fecha: 7 de noviembre de 2026. Rama: `codex/prototipo-fase4`.

## Entrega

Se completa el alcance técnico local de fase 4: aplicación instalable, iconos propios, actualizaciones explícitas, comportamiento cuando el servidor no responde y arranque sencillo en Windows. Se conservan el acceso, la cuenta, las evaluaciones y el procedimiento de preparación de fase 3. Publicación HTTPS e integración predictiva siguen pendientes en sus fases correspondientes.

El médico puede abrir `app/IniciarKneeOAI.cmd` con doble clic. El iniciador sirve la interfaz compilada y la API en `http://127.0.0.1:8000/acceso`; ya no necesita mantener Vite abierto para usar esta versión. La primera preparación requiere dependencias e Internet. Una vez preparada, la misma distribución se reutiliza sin reconstruirla ni necesitar Internet para arrancar el servidor local. Si KneeOAI ya está abierto, el iniciador reutiliza el servicio sin recargar ventanas.

`app/DetenerKneeOAI.cmd` verifica identificador, fecha exacta de inicio, ejecutable y comando antes de detener exclusivamente su proceso registrado y descendientes. No detiene procesos por nombre ni por ocupar un puerto. Es compatible con la fecha JSON interpretada como texto en Windows PowerShell 5 y como fecha en PowerShell 7. Guardar el trabajo antes de detener el servidor. Las evaluaciones confirmadas se conservan.

## Instalación y conexión

El manifiesto declara nombre, identidad, inicio, alcance, modo de ventana e iconos PNG 192/512 y SVG. La interfaz ofrece instalar únicamente cuando el navegador proporciona su evento real de instalación; no simula una instalación exitosa. En Edge hay instrucciones alternativas cuando no se ofrece el botón. Se reconoce el modo instalado mediante el navegador y el modo de presentación.

La PWA guarda solo archivos públicos de la interfaz, verificados por SHA-256. No hay caché de sesión, respuestas API, evaluaciones, imágenes privadas ni resultados; tampoco sincronización o envío posterior de formularios. El servidor sigue comprobando cada operación y cada permiso.

Cuando no se puede conectar, se muestra que no existe confirmación de guardado. Una escritura fallida conserva los campos para reintentar. Reabrir sin servidor permite mostrar la interfaz pública, pero no inventa una sesión ni presenta casos privados recuperados de la caché. El indicador de Internet se distingue de la disponibilidad real del servidor: un servidor loopback puede funcionar aunque el navegador indique falta de Internet.

Las condiciones de promoción e instalación varían según navegador. El origen local loopback está admitido; un despliegue web necesita HTTPS. [Requisitos de instalación en MDN](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Making_PWAs_installable).

## Actualizaciones

Una nueva versión queda esperando. No hay recargas automáticas, `skipWaiting` automático ni apropiación de otras ventanas mediante `clients.claim`. El usuario elige cuándo actualizar. Se bloquea durante escrituras/cargas y se reutilizan los avisos de cambios pendientes del formulario. Acceso escrito y puntos QC todavía no añadidos tienen confirmación adicional de descarte. Cancelar conserva el formulario y la versión activa.

El worker rechaza la activación si hay otra ventana del origen abierta. Solo la ventana solicitante puede recargarse tras comprobar nuevamente que sea seguro. Los assets estáticos anteriores se conservan mientras pueda haber páginas usándolos; no contienen datos privados. El navegador puede desalojar su almacenamiento, por lo que la caché no sustituye un respaldo ni la disponibilidad del servidor.

El iniciador conserva copias públicas inmutables en una carpeta privada de ejecución. Publicar una versión local nueva requiere detener y volver a iniciar el servicio; no se sobrescribe una distribución en uso. El inventario y los bytes servidos son inmutables durante cada instancia. Las rutas API desconocidas nunca reciben el HTML como respuesta de respaldo.

## Evidencia

| Comprobación | Resultado |
| --- | --- |
| Instalación limpia del backend y sus regresiones, privacidad y entrega de archivos | 185 pruebas correctas. |
| Recorrido existente, transporte, sesiones y preparación/revisión | 42 pruebas de navegador correctas. |
| PWA con dos distribuciones compiladas: instalación, caché, desconexión, actualización, varias ventanas, guardados, acceso y punto QC pendientes | 8 pruebas de navegador correctas. |
| Núcleo de caché, identidad de builds y ciclo de actualización | 13 pruebas correctas. |
| Contrato API sin cambios de operaciones | 16 operaciones y 141 ejemplos ficticios validados. |
| Compilación de producción | Correcta. |
| Instalación real en Edge para Windows, en perfil nuevo aislado | Ventana independiente; cierre completo y reapertura sin reinstalar; sesión y evaluación ficticia recuperadas. |
| Iniciadores en Windows PowerShell | Inicio, reutilización del servicio, parada y reinicio correctos. |
| Acceso local existente después de actualizar | Inicio/listado 200, cierre 204, interfaz/manifiesto 200; contraseña conservada. |

No hubo fallos, pruebas omitidas ni pruebas inestables en los dos conjuntos finales de navegador. Las pruebas emplearon bases y DICOM sintéticos; no se cargaron modelos predictivos ni se abrieron datos de prueba reservada. La comprobación de Edge instaló exclusivamente en un perfil de prueba, seleccionó abrir como ventana mediante la preferencia de ese perfil y desinstaló esa instalación al terminar. No instaló en el perfil personal del investigador.

El script reproducible es `frontend/tests/support/verify-pwa-install.mjs`. Usa un servidor ficticio dedicado en un puerto 8020–8099, nunca el 8000 del usuario. Para asociar inequívocamente la ventana, compara el `windowId` de los targets CDP `tab` y `page`; además exige que el navegador informe `standalone`. Las capturas y los informes completos quedan fuera de Git.

La ejecución remota de GitHub debe distinguirse de estos resultados locales: la cuenta venía bloqueada por facturación antes de iniciar pasos. Se verificará el estado de esta rama al publicarla; no se modificará facturación.

## Límites y continuación

La versión local requiere el servidor activo para leer o guardar evaluaciones. La PWA no inicia automáticamente Python al abrir el icono y no incorpora una base offline ni sincronización. Para usarla, iniciar primero KneeOAI; cerrar la ventana instalada no detiene el servidor.

No cambian metodología, entrenamientos, transformaciones científicas ni prueba reservada. La predicción sigue bloqueada. Sigue fase 5: dirección gratuita, HTTPS, alojamiento dentro del presupuesto y recuperación. La conexión del paquete predictivo y su validación pertenecen a fase 6.
