# Plan de implementación por fases

Este plan convierte el Capítulo III de la tesis en trabajo ejecutable con Codex, GitHub, Google Drive y Colab Pro. Las fechas son las del cronograma metodológico y deberán ajustarse a los tiempos reales de descarga y al calendario académico. La cohorte tabular auditada al 23 de septiembre de 2026 contiene 2 778 rodillas de 1 916 participantes (473 progresoras y 2 305 no progresoras); esas cifras son el punto de partida, no el tamaño definitivo después del control de imágenes.

## Reglas que rigen todas las fases

1. Una fila representa una rodilla, pero ninguna persona puede aparecer a la vez en desarrollo y prueba ni en dos pliegues incompatibles.
2. Solo los datos de V00 son predictores. KL de V06 y cualquier información futura sirven para construir o comprobar el desenlace, jamás como entrada.
3. La comparación principal utilizará las mismas rodillas para los modelos clínico, radiográfico y multimodal.
4. La prueba reservada no se utilizará para escoger recortes, arquitecturas, hiperparámetros, calibradores ni umbrales.
5. Los archivos de OAI, los identificadores, las rutas privadas, las credenciales, las predicciones por caso y los pesos se guardarán fuera de este repositorio público. En GitHub quedarán código, configuraciones y reportes agregados que se puedan publicar conforme a las condiciones de uso.
6. Cada experimento registrará versión de datos, código, particiones, semilla, dependencias, GPU, parámetros, tiempo y artefactos. Los notebooks ejecutarán el código compartido; la inferencia del prototipo reutilizará las mismas transformaciones.
7. Si alguna decisión cambia por evidencia de la auditoría, se documentará antes de abrir la prueba, con motivo y efecto sobre la metodología.

## Fase 0. Base del proyecto y trazabilidad

**Cuándo:** inicio inmediato, en paralelo con la obtención de imágenes.

**Trabajo:** preparar la estructura `src/knee/`, `notebooks/`, `configs/`, `tests/` y `app/`; definir dependencias reproducibles y la configuración de rutas de Drive sin valores privados; registrar la procedencia de la extracción Oracle/miNDAR (paquete 1249038, consultas, versión de cohorte); establecer un inventario de archivos y una plantilla de bitácora experimental. Agregar controles de Git para impedir subir datos o credenciales por accidente.

**Entrega y cierre:** repositorio que ejecuta comprobaciones básicas en Colab y localmente, README actualizado, configuración de ejemplo y mapa entre CSV, manifiesto y DICOM. No es necesario subir a GitHub los CSV reales ni los DICOM.

## Fase 1. Descarga selectiva y control radiográfico

**Cuándo:** 24 de septiembre al 9 de octubre en el cronograma; pendiente de disponibilidad real de los archivos.

**Trabajo:** corregir primero la auditoría piloto: `00829603` se registró dos veces como `.tar` y `.tar.gz`, mientras que falta auditar `00839603`. Confirmar los diez estudios únicos. Descargar selectivamente los archivos indicados por las 1 916 rutas únicas del manifiesto, sin trasladar el paquete OAI completo. Registrar disponibilidad, tamaño, huella de integridad, lectura DICOM, número de cuadros, profundidad, fotometría, dimensiones, modalidad y motivo de cualquier fallo.

Diseñar con casos de desarrollo una regla verificable para separar las dos rodillas de cada radiografía bilateral y localizar la región tibiofemoral. Verificar la lateralidad con anatomía y marcadores, pues el campo DICOM suele estar vacío. Revisar recortes y casos de baja confianza sin mirar la etiqueta de progresión. Conservar el original, el recorte, los parámetros usados y un registro de aceptación o rechazo. La operación de inferencia deberá poder reproducir el recorte sin una intervención manual privilegiada.

**Entrega y cierre:** auditoría de las 1 916 adquisiciones o de las efectivamente obtenidas; manifiesto de recortes aceptados por rodilla; causas de exclusión; ejemplos visuales de control de calidad sin identificadores para uso permitido. No comenzar el entrenamiento final hasta conocer la cohorte común utilizable.

## Fase 2. Cohorte analítica y particiones congeladas

**Cuándo:** 10 al 14 de octubre en el cronograma, después de la fase 1.

**Trabajo:** unir el CSV con el manifiesto de imágenes aceptadas por `SRC_SUBJECT_ID`, `SIDE` y `BARCODE_BASE`. Comprobar unicidad, correspondencia, fecha V00 anterior a V06 y etiqueta definida como KL de V06 menos KL de V00 mayor o igual a uno. Documentar el tamaño final y las diferencias frente a las 2 778 rodillas auditadas inicialmente.

Con semilla 2026, asignar participantes a cinco bloques aproximadamente equilibrados; reservar el bloque 0 (aproximadamente 20 %) como prueba. Crear cinco pliegues agrupados dentro de los otros bloques de desarrollo. Los sujetos del piloto utilizados para ajustar recortes permanecerán en desarrollo. Comprobar sujetos, rodillas, positivos y negativos en cada división, sin probar semillas hasta obtener una partición favorable.

**Entrega y cierre:** manifiesto de particiones versionado en almacenamiento restringido, recuentos agregados publicables y comprobaciones automáticas de ausencia de cruces. Las etiquetas de prueba quedan apartadas hasta la fase 6.

## Fase 3. Modelos clínicos y referencia

**Cuándo:** 15 al 21 de octubre en el cronograma.

**Trabajo:** implementar pipelines con edad, sexo e IMC de V00 para regresión logística regularizada, XGBoost y perceptrón multicapa. Imputar los tres IMC ausentes con la mediana calculada en cada entrenamiento; codificar y escalar dentro del pliegue, nunca antes de dividir. Evaluar la referencia clínica que añade KL inicial y, de forma complementaria, una variante con cirugía previa y WOMAC, sin incorporarlas retroactivamente al contraste principal.

**Entrega y cierre:** predicciones fuera de pliegue y métricas de desarrollo de todos los candidatos, configuración ganadora según PR-AUC definida como `average_precision_score`, transformaciones guardadas y registro de costos. Las métricas de desarrollo no serán el resultado final de la tesis.

## Fase 4. Modelo radiográfico

**Cuándo:** 15 al 31 de octubre en el cronograma; puede solaparse con la fase clínica.

**Trabajo:** hacer un ensayo corto en Colab Pro para medir tiempo, VRAM y tamaño de lote. Entrenar DenseNet121 y ViT-B/16 con pesos preentrenados, recortes de 224 × 224, primero con codificador congelado y después con ajuste fino acotado. Aplicar aumentos moderados solo en entrenamiento. Limitar la búsqueda a ocho configuraciones por codificador, máximo 60 épocas y parada temprana de ocho épocas, sujeto al presupuesto real registrado. Seleccionar por PR-AUC media agrupada en desarrollo; entrenar la configuración neuronal seleccionada con semillas 2026, 2027 y 2028.

**Entrega y cierre:** pesos y predicciones fuera de pliegue guardados fuera de GitHub, curvas, parámetros, consumo de recursos y verificación de que cada predicción de validación proviene de un codificador que no vio a ese participante.

## Fase 5. Fusión multimodal y contrastes

**Cuándo:** 1 al 6 de noviembre en el cronograma.

**Trabajo:** implementar la fusión intermedia de representación radiográfica y rama densa clínica (edad, sexo e IMC). Reutilizar por pliegue solo pesos visuales entrenados sin los participantes de validación de ese pliegue. Comparar, como análisis complementario, una fusión tardía con promedio fijo 0,5/0,5 de las probabilidades clínica y radiográfica. Mantener la misma cohorte y particiones en los tres escenarios principales. Limitar la búsqueda de fusión intermedia a ocho configuraciones y documentar si el presupuesto obliga a reducirla.

**Entrega y cierre:** predicciones fuera de pliegue, selección del multimodal basada solo en desarrollo, comparación provisional y registro de todos los candidatos. Un resultado inferior del multimodal también se conserva y se reporta.

## Fase 6. Calibración, prueba reservada y análisis

**Cuándo:** 7 al 12 de noviembre en el cronograma.

**Trabajo:** ajustar calibrador y umbral usando predicciones fuera de pliegue de desarrollo; fijar el umbral por índice de Youden según el protocolo. Congelar arquitectura, entradas, recortes, pesos, calibrador, umbral, código y secuencia de inferencia antes de abrir el bloque 0. Entrenar las versiones finales con todo desarrollo durante el número de épocas fijado a partir de los pliegues.

Evaluar una sola vez las predicciones finales de prueba con PR-AUC (AP) como comparación principal; reportar ROC-AUC, sensibilidad, especificidad, precisión, F1, matriz de confusión, Brier y calibración. Estimar intervalos y diferencias pareadas con bootstrap de 2 000 remuestreos por participante. Completar los análisis de sensibilidad e interpretabilidad previstos en el capítulo III, dejando claro que una revisión de mapas no demuestra causalidad ni utilidad clínica.

**Entrega y cierre:** tablas, curvas y conclusiones basadas en prueba; paquete de inferencia congelado; bitácora de apertura de prueba. No reajustar el sistema en respuesta al desempeño observado allí.

## Fase 7. Prototipo en Python y valoración

**Cuándo:** desarrollo inicial desde octubre; integración y pruebas hasta el 18 de noviembre.

**Trabajo:** implementar un servicio de inferencia con FastAPI y una interfaz web. Recibir DICOM basal, lateralidad, edad, sexo e IMC; validar entradas, reproducir la selección y el recorte de rodilla, aplicar el mismo procesamiento clínico, promediar predicciones, calibrar y clasificar con el umbral congelado. Mostrar probabilidad de progresión estructural a 48 meses, clasificación, procedencia OAI, versión y limitaciones. Si el recorte o la lateralidad no son confiables, devolver un error claro sin inventar una probabilidad. Las pruebas funcionales compararán la salida del servicio con la del pipeline experimental para casos idénticos.

Revisar el prototipo con el traumatólogo mediante los casos autorizados y la ficha prevista. Esta revisión trata comprensión y pertinencia de la interfaz; no constituye validación externa del desempeño predictivo. No afirmar transportabilidad a pacientes peruanos sin una cohorte externa con desenlace comparable.

**Decisión de alcance antes de implementar la interfaz:** el capítulo I menciona mostrar un grado KL estimado, mientras que el capítulo III no define ni evalúa un modelo para producirlo. Resolver la discrepancia de los capítulos antes de prometer esa salida. El modelo principal de progresión y sus comparaciones no dependen de esa función adicional.

**Entrega y cierre:** servicio e interfaz ejecutables, pruebas críticas aprobadas, paquete de modelo versionado y registro descriptivo de la valoración profesional.

## Fase 8. Resultados, tesis y entrega

**Cuándo:** redacción progresiva; cierre previsto para el 23 de noviembre, seguido de preparación de exposición en diciembre.

**Trabajo:** actualizar el capítulo de desarrollo con lo realmente implementado; consignar exclusiones, cohortes finales, recursos, configuraciones, decisiones y fallos. Redactar resultados de comparación, incertidumbre, calibración, errores, prototipo y limitaciones. Ajustar el capítulo III solo cuando lo ejecutado requiera una corrección explícita y trazable. Preparar figuras de flujo, arquitectura y resultados, anexos de reproducibilidad y demostración.

**Entrega y cierre:** tesis coherente con el código y los resultados, prototipo demostrable y material de sustentación. Confirmar las fechas institucionales de entrega y exposición antes de tratarlas como definitivas.

## Primer bloque de trabajo con Codex

1. Crear la estructura del proyecto, configuración de ejemplo y verificaciones de integridad del CSV y manifiesto.
2. Corregir el recuento del piloto y ejecutar de nuevo la auditoría con `00839603`.
3. Preparar un notebook de Colab que monte Drive, compruebe dependencias y rutas, y audite una muestra de DICOM sin exponer claves.
4. Revisar juntos los recortes y la lateralidad antes de procesar masivamente las imágenes.

**Responsabilidades de acceso:** el investigador mantiene el acceso autorizado a NDA, obtiene los DICOM seleccionados y los almacena en Drive; Codex prepara, revisa y mantiene el código y la documentación en GitHub. Las credenciales no se incorporan a este repositorio.

## Riesgos y puntos de decisión

| Punto | Acción antes de seguir |
| --- | --- |
| Descarga incompleta o DICOM ilegibles | Registrar disponibilidad y exclusiones; recalcular la cohorte común antes de partirla. |
| Lateralidad o recortes dudosos | Revisar a ciegas del desenlace; fijar una regla reproducible o rechazar casos irresolubles. |
| Recursos de Colab Pro menores a lo previsto | Medir con un ensayo; reducir búsquedas dentro de desarrollo y documentar el cambio. |
| Resultado multimodal sin mejora | Reportar el contraste y las limitaciones sin cambiar modelos usando la prueba. |
| Ausencia de datos peruanos longitudinales | Mantener la evaluación interna OAI y la valoración del especialista separadas; no llamarlas validación externa. |
| KL estimado en el capítulo I | Acordar si se elimina de esa promesa o se diseña y evalúa como tarea auxiliar independiente. |
