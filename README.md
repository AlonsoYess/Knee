# Predicción de progresión estructural de artrosis de rodilla

Proyecto de tesis de Ingeniería de Sistemas: desarrollo y evaluación de un sistema multimodal que integra una radiografía basal de rodilla y variables clínicas basales para estimar la progresión estructural radiográfica en la cohorte OAI.

## Estado

La Fase 0 está cerrada técnicamente: la cohorte tabular fue auditada en local y en Colab, y sus evidencias privadas fueron verificadas en Drive. La Fase 1 ha comenzado y su paso 1 está cerrado: el piloto DICOM fue reconciliado y auditado con diez adquisiciones únicas. El paso 2 está en curso: el inventario maestro de 1,916 adquisiciones fue ejecutado y verificado, y los dos primeros lotes de 100 paquetes fueron descargados, auditados y promovidos a Drive. El inventario registra 210 adquisiciones disponibles y 1,706 pendientes en los lotes 3 al 20. La inspección completa de radiografías, los recortes, las particiones definitivas, el entrenamiento y la evaluación del prototipo siguen pendientes. No se reportan métricas predictivas hasta completar los experimentos.

## Plan de trabajo

Ver [Plan de implementación por fases](docs/PLAN_DE_IMPLEMENTACION.md). Cada fase incluye entregables, criterios de cierre y dependencias. El plan sigue el Capítulo III de la metodología y se actualizará con las decisiones y resultados realmente ejecutados.

El [contrato de alcance y matriz de trazabilidad](docs/00_ALCANCE_Y_TRAZABILIDAD.md) vincula los problemas, objetivos e hipótesis de los capítulos I y II con las variables, modelos, métricas, evidencias y entregables definidos en el capítulo III. Toda mejora metodológica deberá pasar por el control de cambios descrito allí antes de ejecutarse.

Las [reglas metodológicas invariables](docs/01_REGLAS_INVARIABLES.md) convierten ese alcance en un catálogo auditable. Su versión legible por código está en [`configs/governance/scope_contract.json`](configs/governance/scope_contract.json) y cuenta con pruebas para detectar cambios silenciosos. El paso 2 está aprobado como versión 1.0, pero no autoriza entrenamiento.

El [inventario documental](docs/INVENTARIO_DOCUMENTAL.md) es el resumen público de las versiones y respaldos verificados. El [acta pública de cierre de la Fase 0](docs/05_CIERRE_FASE_0.md) registra los controles superados sin revelar datos privados. Las ubicaciones exactas, identificadores, enlaces y huellas se conservan únicamente en el inventario privado de Drive.

La [estructura maestra de Drive y el inventario de fuentes](docs/02_ESTRUCTURA_DRIVE_Y_FUENTES.md) distinguen las fuentes académicas rectoras, el contexto histórico y los archivos técnicos privados. El [registro público resumido](configs/governance/source_registry.json) conserva roles, autoridad, sensibilidad y estado de verificación sin exponer nombres exactos, rutas, tamaños, huellas ni enlaces privados.

El [control de cambios metodológicos](docs/03_CONTROL_CAMBIOS_METODOLOGICOS.md), aprobado como versión 1.0, exige registrar y aprobar cualquier propuesta antes de aplicarla. Incluye una [plantilla controlada](docs/templates/SOLICITUD_CAMBIO_METODOLOGICO.md), un [registro JSON](configs/governance/methodology_change_log.json) y una validación automática que bloquea propuestas incompletas, no aprobadas o influenciadas por la prueba reservada.

La [estrategia de modelos](docs/06_ESTRATEGIA_MODELOS.md) y su [registro ejecutable](configs/model_registry.json) conservan las líneas base del Capítulo III y preespecifican los candidatos modernos, sus puertas de elegibilidad y su selección exclusivamente en desarrollo. Su aprobación prepara el modelado, pero no adelanta el entrenamiento ni abre la prueba.

El [paso 1 de la Fase 1](docs/07_FASE_1_PASO_1.md) define cómo reconciliar el manifiesto piloto con diez adquisiciones DICOM realmente únicas, auditar su integridad por contenido y conservar la evidencia individual únicamente en Drive. Su [acta pública de cierre](docs/08_CIERRE_FASE_1_PASO_1.md) registra la ejecución verificada. La libreta [`01_auditoria_piloto_dicoms.ipynb`](notebooks/01_auditoria_piloto_dicoms.ipynb) se limita a orquestar ese control.

El [paso 2 de la Fase 1](docs/09_FASE_1_PASO_2.md) verifica el contrato de 1,916 adquisiciones basales, distingue una descarga pendiente de una ausencia real, audita por contenido los paquetes ya disponibles y genera lotes privados para descarga selectiva. La libreta [`02_inventario_adquisiciones.ipynb`](notebooks/02_inventario_adquisiciones.ipynb) orquesta el módulo versionado sin incluir datos privados ni lógica científica duplicada.

El [avance verificado del paso 2](docs/10_AVANCE_FASE_1_PASO_2.md) documenta el inventario inicial y el mecanismo de descarga exacta con el cliente oficial de NDA. La libreta [`03_descarga_selectiva_lote.ipynb`](notebooks/03_descarga_selectiva_lote.ipynb) conserva la ejecución controlada de un lote. La libreta [`04_descarga_lotes_pendientes.ipynb`](notebooks/04_descarga_lotes_pendientes.ipynb) reanuda el inventario y procesa automáticamente todos los pendientes, manteniendo validación, evidencia y cierre independientes por lote y detención inmediata ante el primer fallo.

## Empezar la fase 0

La [guía de arranque](docs/FASE_0.md) indica cómo organizar Drive y ejecutar la [libreta de Colab](notebooks/00_arranque_colab.ipynb). La libreta solo orquesta una revisión versionada del repositorio: monta Drive, instala el paquete, ejecuta sus pruebas y llama a los comandos de auditoría. La lógica científica permanece en `src/knee`. La auditoría compara el CSV de 21 columnas con el manifiesto y guarda recuentos agregados, huellas SHA-256 y el commit ejecutado. El [registro de decisiones](docs/DECISIONES.md) conserva los asuntos pendientes.

## Alcance del experimento

- Unidad de análisis: rodilla con KL inicial 2 o 3; particiones agrupadas por participante.
- Entrada: radiografía bilateral PA con flexión fija de V00, recortada para la rodilla correspondiente, más edad, sexo e IMC de V00.
- Desenlace: incremento de al menos un grado Kellgren-Lawrence entre V00 y V06, visita del horizonte operacional de 48 meses.
- Comparación central: modelos clínico, radiográfico y multimodal sobre la misma cohorte y prueba reservada.
- La salida del prototipo será una probabilidad de progresión y su clasificación según un umbral fijado en desarrollo. Es un prototipo académico, no un diagnóstico ni una recomendación terapéutica.

## Datos y acceso

El repositorio público contendrá código, configuraciones y documentación sin identificadores individuales. Los DICOM, manifiestos con identificadores, CSV de cohorte, credenciales, pesos del modelo y metadatos exactos de fuentes privadas se mantendrán fuera de GitHub, en almacenamiento autorizado. Google Drive será el almacenamiento de trabajo para Colab Pro. No pegar claves, identificadores o rutas privadas en notebooks, documentación pública o incidencias.

## Estructura prevista

```text
docs/       protocolo y decisiones
notebooks/  ejecución reproducible en Colab
src/knee/   extracción, calidad, recortes, particiones, entrenamiento e inferencia
configs/    configuraciones de experimentos
tests/      verificaciones de integridad y funciones críticas
app/        servicio FastAPI e interfaz
```

Los directorios se crearán cuando comience la fase correspondiente. Los notebooks y la aplicación llamarán a funciones de `src/knee`; no duplicarán preprocesamiento, entrenamiento ni inferencia. La separación entre GitHub, Colab, Drive y el futuro prototipo se documenta en la [arquitectura reproducible](docs/04_ARQUITECTURA_REPRODUCIBLE.md), aprobada como versión 1.1.
