# Predicción de progresión estructural de artrosis de rodilla

Proyecto de tesis de Ingeniería de Sistemas: desarrollo y evaluación de un sistema multimodal que integra una radiografía basal de rodilla y variables clínicas basales para estimar la progresión estructural radiográfica en la cohorte OAI.

## Estado

La cohorte tabular está auditada. La inspección y descarga completa de radiografías, los recortes, las particiones definitivas, el entrenamiento y la evaluación del prototipo están pendientes. No se reportan métricas predictivas hasta completar los experimentos.

## Plan de trabajo

Ver [Plan de implementación por fases](docs/PLAN_DE_IMPLEMENTACION.md). Cada fase incluye entregables, criterios de cierre y dependencias. El plan sigue el Capítulo III de la metodología y se actualizará con las decisiones y resultados realmente ejecutados.

## Empezar la fase 0

La [guía de arranque](docs/FASE_0.md) indica cómo organizar Drive y ejecutar la [libreta de Colab](notebooks/00_arranque_colab.ipynb). La auditoría compara el CSV de 21 columnas con el manifiesto y guarda recuentos agregados y huellas SHA-256. El [registro de decisiones](docs/DECISIONES.md) conserva los asuntos pendientes.

## Alcance del experimento

- Unidad de análisis: rodilla con KL inicial 2 o 3; particiones agrupadas por participante.
- Entrada: radiografía bilateral PA con flexión fija de V00, recortada para la rodilla correspondiente, más edad, sexo e IMC de V00.
- Desenlace: incremento de al menos un grado Kellgren-Lawrence entre V00 y V06, visita del horizonte operacional de 48 meses.
- Comparación central: modelos clínico, radiográfico y multimodal sobre la misma cohorte y prueba reservada.
- La salida del prototipo será una probabilidad de progresión y su clasificación según un umbral fijado en desarrollo. Es un prototipo académico, no un diagnóstico ni una recomendación terapéutica.

## Datos y acceso

El repositorio público contendrá código, configuraciones y documentación sin identificadores individuales. Los DICOM, manifiestos con identificadores, CSV de cohorte, credenciales y pesos del modelo se mantendrán fuera de GitHub, en almacenamiento autorizado. Google Drive será el almacenamiento de trabajo para Colab Pro. No pegar claves ni rutas privadas en notebooks o incidencias.

## Estructura prevista

```text
docs/       protocolo y decisiones
notebooks/  ejecución reproducible en Colab
src/knee/   extracción, calidad, recortes, particiones, entrenamiento e inferencia
configs/    configuraciones de experimentos
tests/      verificaciones de integridad y funciones críticas
app/        servicio FastAPI e interfaz
```

Los directorios se crearán cuando comience la fase correspondiente. Los notebooks llamarán a funciones de `src/knee` para que entrenamiento e inferencia compartan el mismo preprocesamiento.
