# Fase 3 — DICOM, preparación y revisión persistentes

Versión de interfaz 0.10.0; API 0.3.0. Fecha: 7 de noviembre de 2026. Rama de aplicación: `codex/prototipo-fase3`.

## Alcance

Se completa el recorrido de aplicación desde acceso y datos basales hasta carga privada de DICOM, referencia visual sellada, preparación, revisión y resumen recuperables. La predicción de progresión permanece bloqueada. Este cierre técnico no constituye validación clínica ni el cierre de la integración predictiva de fase 6.

La interfaz Vue conserva SQLite detrás de FastAPI. El médico ve los estados, las observaciones pendientes y una acción para continuar. Puede volver al resumen, recargar o iniciar otra sesión sin perder lo confirmado por el servidor. Se avisa al abandonar una edición pendiente; un error no se presenta como guardado correcto.

## Comportamiento entregado

- Carga de un único DICOM, con límites reales de bytes, nombre interno opaco y archivos privados. Las vistas PNG requieren sesión, propiedad del caso, revisión exacta e integridad SHA/tamaño.
- Original bilateral y mitad en ventanas 1–99 y 5–95. El centro visual, la inspección, el cegamiento y una nota se sellan antes de ejecutar o mostrar el recorte. La referencia no se puede editar posteriormente y nunca guía el localizador.
- Preparación por el procedimiento congelado existente, mediante proceso separado y sin instalar dependencias en los entornos científicos. Se mantienen matrices y píxeles nativos, sin corrección manual de geometría.
- Observación QC completa, evaluada por el procedimiento aprobado. No basta marcar una confirmación genérica. Aceptación requiere evaluación favorable; rechazo administrativo exige motivo y permanece distinguido de una aceptación científica.
- Decisiones con revisor, fechas, versiones e historial. Editar edad, sexo o IMC conserva el QC radiográfico; cambiar imagen o lado invalida su asociación activa. Cambiar la versión del preparador o política evita reutilizar una aceptación como vigente.
- Control de concurrencia y solicitudes idempotentes. Repetir una respuesta perdida no duplica archivos, referencias ni decisiones, ni reactiva una preparación histórica. La nueva selección explícita de los mismos bytes puede recuperar la preparación anterior sin recalcular ni resellar.
- Interrupción persistida y reintento sobre la misma fuente y referencia. La abstención del localizador se distingue de un fallo inesperado y no produce aceptación.
- Migración SQLite 1→2, conservando cuentas y revisiones. Respaldo y restauración conjuntos de base y archivos, con manifiesto SHA-256/tamaños; una restauración revoca sesiones copiadas y usa destino nuevo.

## Verificación

La entrega se comprueba con dependencias propias de aplicación en una copia exportada del código. Las evidencias detalladas, bases temporales, capturas y archivos sintéticos se conservan privadamente fuera de la publicación.

| Verificación | Resultado |
| --- | --- |
| Backend: acceso, permisos, migración, archivos, referencia, QC, conflictos, reintentos y restauración | 110 pruebas correctas. |
| Navegador: acceso y transporte, recorrido completo, recarga, rechazo, referencia no resuelta, coordenadas, PNG inválido y respuestas perdidas | 39 pruebas correctas. |
| OpenAPI 3.1 | 16 operaciones y 141 ejemplos ficticios validados. |
| Compilación de producción | Correcta. |
| Puente científico con píxeles sintéticos | 19 comprobaciones correctas, seis PNG y conservación/paridad de píxeles y geometría. |
| Preflight de activos de preparación | Diez modelos del localizador cargados en CPU/eval; sin predicción en ese preflight. |
| API con localizador real sobre el dibujo sintético | Cuatro vistas correctas; abstención explícita, referencia conservada y ninguna aprobación. |

Las pruebas de aplicación usan un procesador ficticio inyectado únicamente por sus harnesses. La comprobación de paridad utiliza puntos sintéticos conocidos mediante Python interno; el protocolo IPC de producción no permite inyectarlos. Ambas evidencias comprueban software y conservación, no rendimiento anatómico o clínico en pacientes. La comprobación neural de la API sobre el dibujo tiene alcance separado y no sustituye ensayos posteriores con casos autorizados.

Se revisó el diseño de escritorio y móvil, incluidos los tres pasos con imágenes y ausencia de desbordamiento horizontal a 375 px. Las dimensiones PNG deben coincidir con las declaradas antes de permitir inspección/aceptación. El desplazamiento de coordenadas del recorte se comprueba con su caja XYXY exacta.

## Uso local y límites

Abrir `http://127.0.0.1:5173/acceso`, con la misma cuenta local. El servidor escucha en `127.0.0.1:8000`; ambos procesos deben estar activos. La actualización conserva la contraseña y el archivo privado de acceso. Se creó un respaldo previo a la migración. El inicio de sesión, listado y cierre de sesión locales se verificaron tras actualizar.

Las ubicaciones del entorno/checkout del preparador están en configuración privada ignorada. Una copia nueva del código necesita esos activos aprobados para preparar imágenes; sin ellos muestra indisponibilidad y conserva el acceso/borrador. No descarga pesos ni activa un sustituto. El almacén no es público; en Windows se deben conservar los permisos de la cuenta del sistema. No se afirma anonimización de píxeles ni aislamiento mediante ACL que no se haya configurado.

La metodología, los entrenamientos, sus pesos predictivos y la prueba reservada no se modifican por esta entrega. No se conectan modelos de progresión, calibradores ni umbrales de clasificación. La cuenta de GitHub tiene un bloqueo de facturación que impidió iniciar la última ejecución remota; ese estado se debe distinguir de las pruebas locales y comprobar al publicar esta rama, sin modificar facturación.

## Siguiente fase

Fase 4: PWA instalable, iconos, actualización y comportamiento sin conexión, con arranque local sencillo. Fase 5 mantiene publicación HTTPS y recuperación en alojamiento dentro del presupuesto. Fase 6 conectará el paquete predictivo aprobado y comprobará su paridad y recursos antes de habilitar el cálculo.
