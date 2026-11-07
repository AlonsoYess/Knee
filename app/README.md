# Prototipo

## Interfaz local v0.8: Base reproducible y contrato con el servidor

La fase 1 del prototipo incorpora el frontend al control de versiones y prepara su comunicación con FastAPI. `contracts/openapi.json` define la API v1 prevista; `contracts/README.md` explica sus estados y reglas. `frontend/src/api/http.ts` proporciona el transporte HTTP comprobado con respuestas ficticias. Las pantallas todavía no llaman a ese transporte: continúan usando la sesión simulada y el guardado local descritos abajo. No hay un servidor FastAPI ejecutándose ni una conexión a modelos.

`frontend/` contiene las pantallas en Vue 3, TypeScript, Vite y Vue Router. Se ejecutan localmente y parten de siete casos completamente ficticios. Las evaluaciones guardadas se conservan en el almacenamiento local del navegador, incluidas las que están incompletas; se recuperan al recargar y tras cerrar la sesión simulada. Los campos modificados sin pulsar un botón de guardado permanecen provisionales. No hay conexión a datos OAI, servicios, base de datos de servidor, pesos ni predicciones. El acceso es una simulación visible; la marca de sesión del navegador no constituye autenticación ni autorización de servidor.

- **Acceso:** validación de campos y credenciales ficticias, mostrar/ocultar contraseña, ayuda de recuperación y botón para completar los datos de ejemplo. Ninguna contraseña se conserva en el almacenamiento del navegador.
- **Mis evaluaciones:** búsqueda por código, filtro por estado, contador y estado sin coincidencias. La búsqueda y el filtro se conservan en memoria al volver desde un caso y se reinician al cerrar sesión o recargar. Nueva evaluación abre el formulario; Retomar abre un borrador navegable. Los ejemplos precargados sin recorrido editable ofrecen Ver detalle. El código de cualquier caso abre su resumen de lectura. Una revisión aceptada que llegó al paso 4 ofrece Ver resumen. Los ejemplos antiguos conservan sus ausencias explícitas, sin asociaciones de imagen inventadas.
- **Nueva evaluación:** código ficticio generado, edad basal, sexo registrado, IMC basal y rodilla. Completar ejemplo rellena solo campos vacíos y conserva lo escrito por el usuario. Permite guardar un borrador incompleto o comprobar que los datos requeridos estén listos. La validación comprueba formato numérico, edad entera positiva e IMC positivo; no establece criterios clínicos nuevos. Guardar y salir conserva el borrador y vuelve al listado; Guardar y continuar abre Radiografía y lado.
- **Radiografía y lado:** catálogo de dos ilustraciones SVG sintéticas, vista ampliada y confirmación explícita del lado del paciente. No se reciben archivos reales ni se simula una lectura DICOM. La imagen debe cargar correctamente antes de confirmar, ampliar o continuar. Cambiar la imagen o el lado elimina la confirmación anterior y cualquier decisión de revisión asociada; también se invalida al cambiar y volver a la selección original dentro de una misma edición. Un lado discrepante requiere corregir la selección o volver a los datos basales. Guardar y salir conserva un borrador de preparación, incluido uno sin imagen o sin confirmar el lado. Guardar y continuar abre Revisión del recorte.
- **Revisión del recorte:** comparación del original y una vista fija del mismo SVG y lado; ambas vistas se pueden ampliar. Los marcos de `demoCrops.ts` son ejemplos de presentación, no parámetros de localización científica. La aceptación exige revisar explícitamente el ejemplo; el rechazo exige un motivo descriptivo de demostración. Aceptar o rechazar selecciona una decisión provisional; Guardar y salir o Guardar y ver resumen la conserva en este navegador, vinculada a imagen, lado y versión. El avance al resumen guarda revisión y paso final en una sola operación. El foco pasa a la acción siguiente después de seleccionar la decisión. La decisión guardada requiere reapertura explícita para modificarse. Si falla alguna vista, se bloquea la aceptación. Las revisiones pendientes y las decisiones aceptadas o rechazadas se pueden retomar desde el listado. El rechazo bloquea el avance al resumen desde esta pantalla; la aceptación permite abrirlo.
- **Resumen del caso:** vista de lectura del estado actual de datos basales, imagen, lado y revisión. Muestra los campos o asociaciones pendientes, el motivo del rechazo o una revisión desactualizada cuando la identidad no coincide. Una acción señala el primer paso pendiente y los enlaces de edición respetan los requisitos de cada pantalla. No fabrica una imagen o decisión para los ejemplos antiguos. Las vistas se pueden ampliar cuando han cargado; un fallo visual se muestra separado de la decisión guardada. La predicción permanece bloqueada y el botón Calcular predicción deshabilitado en todos los estados, incluso con preparación completa; no hay porcentajes, categorías de riesgo ni respuestas de un motor.
- **Navegación:** acceso simulado requerido, menú de la sección resaltado durante todo el recorrido, retorno al paso guardado del borrador y confirmación al abandonar cambios pendientes, incluido el cierre de sesión. La protección incluye los cambios de imagen o lado aunque luego vuelvan al valor inicial. Los diálogos admiten teclado y las pantallas se adaptan a escritorio y móvil.

### Ejecutar localmente

Desde `app/frontend`, con Node.js compatible con el campo `engines` de `package.json` (la versión usada en la verificación se fija en `.node-version`):

```powershell
npm ci --ignore-scripts
npm run test:contracts
npm run dev
```

Abrir `http://127.0.0.1:5173/acceso` y pulsar **Completar datos de ejemplo** y después **Entrar a la demostración**. El servidor solo escucha en la computadora local. La interfaz usa recursos locales una vez instaladas sus dependencias.

```powershell
npm run build
```

Este comando comprueba los tipos y genera `dist/`. `node_modules/`, `dist/` y los resultados de pruebas quedan excluidos de Git. Las versiones se fijan en `package-lock.json`.

Las pruebas del recorrido se ejecutan con `npm run test:e2e` usando Microsoft Edge instalado en Windows. En CI u otros sistemas se utiliza Chromium, instalable con `npx playwright install chromium` (en Linux CI, `npx playwright install --with-deps chromium`). `E2E_PORT` permite un puerto diferente y `CI=true` exige un servidor nuevo para la prueba. Comprueban validaciones, conservación de filtros y valores capturados, navegación, borradores, selección de imágenes, carga lenta, confirmación del lado, revisión pendiente, aceptación, rechazo, reapertura, foco tras decidir, invalidación de decisiones, error de carga, salida con cambios, resumen actual, orientación al primer pendiente, bloqueo de predicción y presentación móvil exclusivamente con ejemplos ficticios. También se verifica el transporte HTTP sin contactar un backend real.

`npm run test:contracts` valida OpenAPI, sus referencias internas y los ejemplos ficticios contra sus esquemas. `.github/workflows/prototype.yml` prepara las comprobaciones de contrato, compilación y recorrido para GitHub Actions, sin ejecutar el código científico.

### Comunicación prevista con FastAPI

Vue solicitará rutas relativas bajo `/api/v1`. En desarrollo, Vite reenvía `/api` a `http://127.0.0.1:8000`; `.env.example` documenta cómo cambiar ese destino mediante una configuración local excluida de Git. No es necesario crear ese archivo para ver la demostración actual. En producción, la misma dirección pública deberá servir la interfaz y enrutar `/api` al backend; el proxy de desarrollo no configura un despliegue.

Las sesiones futuras se gestionarán en el servidor con cookies y protección CSRF. Cada evaluación tendrá una versión para detectar ediciones desactualizadas. La revisión quedará vinculada a imagen, lado y preparación. El cálculo referenciará una revisión vigente y devolverá un identificador de ejecución, su estado y el resultado versionado cuando exista. El contrato distingue sesión vencida, conflicto, entrada no válida y motor no disponible. El transporte no reintenta escrituras automáticamente ni convierte errores en predicciones ficticias.

El frontend compartido no contiene claves de base de datos o del servidor. La sesión simulada actual sigue identificada como tal y no concede permisos de API. La autenticación, la persistencia central y la aplicación efectiva de estas reglas corresponden a la fase 2.

### Guardado local de demostración

1. Crear o retomar una evaluación ficticia.
2. Pulsar **Guardar y salir**, **Guardar y continuar** o **Guardar y ver resumen**. Seleccionar una aceptación o un rechazo por sí solo no guarda la decisión.
3. Recargar la página: se recupera el último estado guardado, con el mismo código y paso. Cerrar sesión no elimina las evaluaciones.

La persistencia usa `localStorage` bajo una clave propia de la demostración, con esquema y catálogo versionados. Se conservan los datos ficticios del caso y los identificadores de las ilustraciones y revisiones. No se guardan contraseñas, archivos, píxeles clínicos ni predicciones. Las preferencias de búsqueda siguen siendo temporales.

La restauración valida estructura, identidad, tipos y asociaciones. Una revisión de una versión anterior conserva su identidad y se presenta como no vigente, sin adjudicarle una aceptación nueva. Si la copia es ilegible o incompatible, se avisa y se impide sobrescribirla con los ejemplos iniciales. Si falla una escritura, se conserva la copia anterior y la edición sigue pendiente. Antes de escribir se comprueba si otra pestaña ya cambió la copia; un cambio detectado bloquea la escritura desactualizada. Esta comprobación es optimista: no hay combinación automática ni exclusión entre guardados exactamente simultáneos. La demostración se utiliza en una sola pestaña de edición.

El alcance es el mismo navegador, perfil y dirección local (`http://127.0.0.1:5173`). Limpiar los datos del sitio elimina esta copia; no es una copia de seguridad ni sincronización con Drive. La etapa actual conserva exclusivamente los ejemplos ficticios del prototipo.

### Organización del código fuente

- `frontend/src/views/`: pantallas Vue (`LoginView.vue`, `EvaluationsView.vue`, `NewEvaluationView.vue`, `RadiographView.vue`, `CropReviewView.vue`, `CaseSummaryView.vue`).
- `frontend/src/components/`: componentes visuales compartidos; `DemoCropPreview.vue` presenta vistas del SVG sintético, sin procesamiento radiográfico.
- `frontend/src/demo/`: casos inventados, estado del recorrido y persistencia local versionada en `demoEvaluationPersistence.ts`. El nombre evita que la exclusión de datos privados `data/` oculte código necesario.
- `frontend/src/api/`: transporte futuro de la API, independiente del estado ficticio de las pantallas.
- `contracts/`: especificación de comunicación con el servidor y ejemplos sintéticos.
- `frontend/public/demo/`: dos ilustraciones vectoriales originales de interfaz, marcadas como sintéticas y no clínicas; no proceden de OAI ni de pacientes.
- `frontend/src/router.ts`: navegación y protección del recorrido de demostración.
- `frontend/tests/`: pruebas de interacción.

La interfaz se mantiene en el mismo repositorio Knee. El futuro backend será FastAPI/Python y reutilizará `src/knee`; Vue se ocupa de las pantallas. Las versiones de código y los artefactos privados conservan destinos distintos; esta entrega no copia datos a Git ni sincroniza las evaluaciones con Drive.

### Siguientes entregas

El recorrido visual puede revisarse desde el acceso hasta el resumen, con la predicción aún bloqueada. Acceso real, persistencia centralizada y permisos del backend corresponden a entregas posteriores, al igual que la instalación PWA y la integración predictiva. Esta entrega no registra un service worker ni guarda imágenes o resultados en caché.

Las opciones y decisiones de revisión son simulaciones de interacción. No ejecutan el QC científico de `src/knee`, no crean una lista nueva de criterios clínicos y no constituyen aceptación de una imagen real. La futura integración deberá conservar el procedimiento científico vigente.

## Integración científica prevista

El servicio FastAPI y la interfaz importarán el paquete de inferencia y el preprocesamiento compartido desde `src/knee`, una vez formalizado el paquete final; no duplicarán transformaciones dentro de la API o de la interfaz. Las pruebas funcionales exigirán la misma salida que el pipeline experimental para entradas idénticas.

El prototipo recibirá DICOM basal, lateralidad, edad, sexo e IMC y mostrará la probabilidad calibrada de progresión estructural a 48 meses y su clasificación mediante el umbral congelado. Conforme a `RES-KL-001`, no estimará ni mostrará un grado Kellgren–Lawrence. Los pesos, radiografías, predicciones individuales y demás artefactos privados no se incluirán en este repositorio.
