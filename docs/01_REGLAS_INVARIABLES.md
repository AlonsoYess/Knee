# Reglas metodológicas invariables

## Control del documento

| Campo | Valor |
| --- | --- |
| Documento | Reglas metodológicas invariables y mecanismos de cumplimiento |
| Estado | Aprobado por el investigador; vigente |
| Versión | 1.4 |
| Fecha de aprobación | 1 de octubre de 2026 |
| Paso | Fase 0, paso 2 |
| Contrato rector | `docs/00_ALCANCE_Y_TRAZABILIDAD.md`, versión 1.3 |
| Contrato ejecutable | `configs/governance/scope_contract.json` |
| Destino de respaldo | Drive privado autorizado; ubicación exacta en el inventario privado |
| Estado del respaldo | Versiones anteriores preservadas en Drive; reglas 1.4 y contrato JSON 1.4 pendientes de sincronización |

## 1. Finalidad

Este documento traduce el alcance aprobado y el Capítulo III en reglas con identificadores estables. Su propósito es que cada fase futura pueda demostrar, mediante configuración, pruebas, bitácoras y artefactos, que continúa dentro de la tesis. La versión 1.4 conserva la ampliación de modelos de `MCR-2026-002` y registra `MCR-2026-004`, aprobada y aún no aplicada, que sustituye a `MCR-2026-003`; no autoriza entrenamiento.

Las reglas se dividen en tres clases:

1. **Invariable:** compromiso académico o experimental que no puede cambiarse por conveniencia técnica ni por resultados observados.
2. **Optimizable con registro:** decisión que puede mejorarse dentro del protocolo, siempre antes de abrir la prueba y dejando evidencia.
3. **Pendiente bloqueante:** ambigüedad que debe resolverse expresamente antes de implementar la parte afectada.

## 2. Catálogo de reglas invariables

| ID | Regla obligatoria | Motivo y evidencia de cumplimiento |
| --- | --- | --- |
| `INV-DIS-001` | El estudio permanece cuantitativo, no experimental, longitudinal, retrospectivo y de análisis secundario. | Define el diseño del Capítulo III. Se comprobará en protocolo, reportes y tesis final. |
| `INV-DAT-001` | La fuente de datos principal es OAI; no se atribuirán los resultados a otra población. | Delimita procedencia y validez externa. La versión y procedencia se registrarán en el inventario de datos. |
| `INV-UNIT-001` | La unidad de análisis es la rodilla y toda partición, validación y remuestreo se agrupa por participante. | Evita fuga entre las dos rodillas de una persona. Habrá pruebas automáticas sobre manifiestos y particiones. |
| `INV-POP-001` | La población principal incluye rodillas con KL basal 2 o 3 que satisfagan los criterios de elegibilidad y calidad. | Mantiene la población definida; toda exclusión deberá llevar causa trazable. |
| `INV-TMP-001` | Los predictores proceden de V00; V06 se usa para construir o comprobar el desenlace a 48 meses y nunca como predictor. | Preserva la precedencia temporal. El esquema de variables y las pruebas rechazarán entradas posteriores al basal. |
| `INV-OUT-001` | `PROGRESION_48M = 1` cuando `KL_V06 - KL_V00 >= 1`; en otro caso vale `0`. | Fija el desenlace primario. La auditoría recalculará la etiqueta en lugar de confiar únicamente en el CSV. |
| `INV-CLIN-001` | Los predictores clínicos principales son edad, sexo e IMC de V00. | Mantiene la comparación prometida. Agregar variables al análisis principal requiere control de cambios. |
| `INV-RAD-001` | La entrada radiográfica principal es el recorte tibiofemoral de la rodilla correspondiente obtenido de la radiografía bilateral PA con flexión fija de V00. | Conserva modalidad, visita, región anatómica y lateralidad. El manifiesto de recortes documentará su trazabilidad. |
| `INV-COH-001` | Clínico, radiográfico y multimodal se comparan sobre la misma cohorte común y las mismas particiones. | Impide ventajas por muestras distintas. Cada tabla comparativa incluirá el mismo identificador de cohorte y partición. |
| `INV-MOD-001` | El escenario clínico evalúa regresión logística, XGBoost y MLP. El radiográfico conserva DenseNet121 y ViT-B/16 como líneas base obligatorias y agrega ConvNeXt V2 Tiny, DINOv3 ViT-S/16 y SKELEX como candidatos modernos registrados. | `MCR-2026-002` amplía candidatos sin cambiar cohorte, entrada, desenlace ni prueba. Ninguna arquitectura no registrada puede agregarse por resultados favorables. |
| `INV-FUS-001` | La fusión intermedia es el enfoque multimodal principal y compara concatenación con compuertas o FiLM. La fusión tardía fija es complementaria. | Evita reemplazar el contraste principal por el resultado complementario más favorable y preespecifica las variantes principales. |
| `INV-KLR-001` | La variante clínica que agrega KL basal es solo una referencia complementaria; no reemplaza al escenario clínico principal ni se incorpora silenciosamente al multimodal principal. | Preserva la pregunta sobre imagen más edad, sexo e IMC y evita una comparación circular o alterada. |
| `INV-SPLIT-001` | La semilla de partición es 2026; el bloque 0 contiene aproximadamente 20 % de participantes y queda reservado como prueba. | Hace reproducible la separación y protege la evaluación final. El manifiesto deberá probar disyunción por participante. |
| `INV-CV-001` | La selección se realiza exclusivamente en desarrollo mediante validación cruzada estratificada y agrupada de cinco pliegues. El bloque 0 no participa en ella. | Evita fuga y optimismo. Las asignaciones de pliegue serán persistidas y auditadas. |
| `INV-TEST-001` | La prueba se abre una sola vez después de congelar cohorte, preprocesamiento, modelos, calibración, umbral, código y secuencia de inferencia. No se reajusta por sus resultados. | Protege la inferencia confirmatoria. Se exigirá una lista de congelamiento y una bitácora de apertura. |
| `INV-MET-001` | La comparación y selección principales usan PR-AUC definida como precisión promedio no interpolada mediante `average_precision_score`. | Evita ambigüedad entre AP y el área trapezoidal. El nombre de implementación deberá constar en configuraciones y reportes. |
| `INV-STAT-001` | La incertidumbre confirmatoria usa 2 000 remuestras bootstrap por participante; las diferencias son pareadas y los dos contrastes principales se ajustan con Holm. | Respeta dependencia entre rodillas, emparejamiento y multiplicidad. Las unidades remuestreadas se comprobarán en pruebas. |
| `INV-CAL-001` | Calibración y umbral se ajustan solo con predicciones fuera de pliegue de desarrollo; el umbral se fija con Youden y preferencia por sensibilidad en empates. | Evita calibrar con prueba. El calibrador y el umbral conservarán procedencia de sus predicciones. |
| `INV-IMB-001` | Las redes tratan el desbalance con BCE ponderada calculando pesos solo en el entrenamiento correspondiente. No se balancea artificialmente la validación o la prueba ni se aplica SMOTE a imágenes. | Evita alterar la distribución de evaluación y crear muestras radiográficas sintéticas impropias. |
| `INV-SEED-001` | Las configuraciones neuronales seleccionadas se ejecutan con semillas 2026, 2027 y 2028. | Permite evaluar variabilidad de optimización sin usar prueba para escoger semilla. |
| `INV-TRACE-001` | Toda ejecución relevante registra revisión de código, versión/huella de datos, cohorte, particiones, semillas, configuración, entorno y artefactos. | Hace auditable la cadena completa. Una métrica sin procedencia no será aceptada como resultado de tesis. |
| `INV-RES-001` | Se conservan y reportan resultados negativos, incluido que el multimodal no supere a uno o ambos unimodales. | Impide selección oportunista y mantiene válidas HG1 y HG0. |
| `INV-PROT-001` | El prototipo es académico, no diagnóstico; entrega probabilidad calibrada y clasificación, declara OAI y KL 2/3, y no recomienda tratamientos. | Mantiene el límite funcional y ético de la tesis. |
| `INV-PRIV-001` | DICOM, identificadores, manifiestos individuales, credenciales, predicciones por caso y pesos no se publican en GitHub. | Protege datos restringidos. Se almacenarán en Drive autorizado y se reforzará con exclusiones y controles. |

## 3. Decisiones optimizables, no invariables

Estas decisiones pueden mejorarse sin cambiar la pregunta de investigación, pero cada cambio deberá registrarse **antes** de ejecutarse y sin consultar la prueba:

- parámetros exactos del localizador y del recorte, siempre que se conserve la región tibiofemoral y se valide su calidad;
- tasas de aprendizaje, tamaño de lote, regularización, número de capas descongeladas y eficiencia de ejecución;
- hiperparámetros dentro del presupuesto predefinido y reducción justificada del espacio de búsqueda por capacidad de cómputo;
- organización interna del código, automatización, formatos de artefactos y precisión mixta;
- elección del candidato ganador dentro del registro cerrado, aplicando la regla de selección de desarrollo;
- precisión mixta, acumulación de gradiente, calentamiento, programación cosenoidal, descongelamiento gradual y decaimiento de tasa por capas conforme al registro de modelos.

Estas libertades no autorizan agregar otra arquitectura, cambiar entradas principales, pérdida, estrategia de partición, métrica principal, calibración, umbral o contrastes. La sensibilidad a 384 × 384 está aprobada solo para candidatos promovidos dentro de desarrollo. Cualquier otra resolución o familia entra al proceso de aprobación de la sección 7.

## 4. Decisiones resueltas

### 4.1 Estrategia de modelos

| ID | Decisión | Fundamento | Consecuencia obligatoria |
| --- | --- | --- | --- |
| `RES-MOD-001` | Ampliar de forma cerrada los candidatos radiográficos y la fusión intermedia mediante `MCR-2026-002`. | Existen pesos y métodos modernos reproducibles, y el presupuesto permite evaluarlos; el tamaño de la cohorte exige limitar la búsqueda y proteger la prueba. | Aplicar `configs/model_registry.json`, mantener las líneas base, promover como máximo dos candidatos modernos solo con desarrollo y revisar licencias antes de cargar pesos. |

La actualización del archivo Word del capítulo III permanece pendiente y será realizada posteriormente por el investigador. El código y la tesis final deberán describir únicamente los candidatos realmente ejecutados.

### 4.2 Salida de KL estimado

| ID | Decisión | Fundamento | Consecuencia obligatoria |
| --- | --- | --- | --- |
| `RES-KL-001` | Retirar del apartado 1.5.2 del capítulo I la promesa de mostrar un grado KL estimado. | La metodología investiga la progresión estructural a 48 meses. Estimar KL exigiría otra tarea, otro procedimiento de entrenamiento y una evaluación no definidos en el capítulo III. | El prototipo solo mostrará la probabilidad calibrada de progresión y su clasificación. No se desarrollará un estimador KL. |

La decisión fue tomada expresamente por el investigador el 27 de septiembre de 2026. Cierra `PEND-KL-001` sin ampliar el alcance. La actualización del archivo Word del capítulo I permanece como acción documental necesaria y será realizada posteriormente por el investigador.

### 4.3 Recorte proporcional y revisión técnica

`MCR-2026-004`, aprobada el 1 de octubre de 2026, sustituye a `MCR-2026-003`, retirada sin aplicación. `INV-RAD-001` se desarrolla mediante una adaptación determinista versionada de knee-crop, ROI proporcional y revisión técnica por el investigador. No requiere pesos ni anotaciones expertas; no se declarará precisión anatómica, validación radiológica o garantía poblacional. Las secciones 4.1–4.4 de la [MCR vigente](25_MCR_2026_004_RECORTE_ROI_Y_REVISION_TECNICA.md) fijan algoritmo, coordenadas, defectos críticos, advertencias y umbrales.

Se conservan separación bilateral congelada y píxeles nativos. El piloto histórico y los veinte participantes nuevos de confirmación pertenecen exclusivamente al desarrollo; estos últimos solo se seleccionan tras superar la regresión. Se prohíben reemplazos de fallos y ajustes guiados por la confirmación, desenlace o prueba. Las transformaciones aprendidas de los modelos mantienen el ajuste exclusivo en el entrenamiento correspondiente.

La actualización académica en Word sigue pendiente, sin autorización para sobrescribir originales. El [acta 26](26_APROBACION_MCR_2026_004_Y_CIERRE_PENDIENTE.md) acredita actualización contractual, no implementación: el CSV de v0.4 no contiene respuestas y sus dos registros de cierre faltan. Verificar el cierre es requisito previo a integrar. Los criterios nuevos no reinterpretan el rechazo 8/20 de v0.4.

## 5. Asuntos pendientes que bloquean implementación

| ID | Asunto | Trabajo permitido mientras esté pendiente | Trabajo bloqueado |
| --- | --- | --- | --- |
| `PEND-DATA-001` | La cohorte efectiva depende de disponibilidad, lectura, lateralidad, región anatómica y calidad de todas las radiografías requeridas. | Descargar, inventariar, auditar y documentar exclusiones. | Congelar particiones definitivas o iniciar entrenamiento final antes de cerrar la cohorte común. |
| `PEND-PILOT-001` | La muestra piloto actual no representa diez estudios únicos: contiene dos empaquetados de una adquisición y omite otra prevista. | Reconstruir una muestra piloto de diez estudios únicos y repetir la inspección. | Usar la inspección actual como evidencia de cobertura o tasa de calidad del universo. |

## 6. Mecanismos de cumplimiento

| Mecanismo | Función | Estado en este paso |
| --- | --- | --- |
| `configs/governance/scope_contract.json` | Fuente legible por código de los compromisos, decisiones y bloqueos. | Versión 1.4; incluye MCR-2026-004 aprobada y no aplicada, con cierre de v0.4 pendiente. |
| `configs/model_registry.json` | Define candidatos, elegibilidad, promoción, técnicas y prohibiciones del modelado. | Aprobado como versión 1.0 bajo `MCR-2026-002`; no autoriza entrenamiento. |
| `tests/test_scope_contract.py` | Detecta eliminación o cambio silencioso de valores críticos. | Creado; debe aprobarse antes de cerrar el paso. |
| Manifiesto privado de cohorte/particiones | Prueba unidad, lateralidad, elegibilidad, exclusiones y ausencia de cruce entre grupos. | Se construirá en fases 1 y 2. |
| Registro de ejecución | Une código, datos, configuración, entorno, semillas y artefactos. | Estructura inicial existente; se ampliará antes de entrenar. |
| Lista de congelamiento y bitácora de apertura | Demuestra que la prueba no intervino en decisiones. | Se preparará antes de la fase 6. |
| Registro de decisiones | Conserva cambios, motivos y aprobaciones sin reescribir el pasado. | Activo en `docs/DECISIONES.md`. |

Las pruebas de este paso validan el **contrato**, no demuestran todavía la calidad del conjunto de datos ni la ausencia real de fuga. Esas evidencias solo existirán cuando se generen y auditen los manifiestos privados.

## 7. Procedimiento obligatorio de cambio

Antes de ejecutar una desviación o mejora que afecte una regla, se abrirá un registro con:

1. identificador y fecha;
2. regla afectada;
3. problema observado con evidencia;
4. alternativa propuesta y alternativas descartadas;
5. impacto sobre capítulos I, II y III, hipótesis y comparabilidad;
6. impacto sobre sesgo, fuga, calibración, inferencia y reproducibilidad;
7. costo en datos, tiempo y cómputo;
8. decisión explícita del investigador;
9. archivos y versiones modificados;
10. momento de aplicación y confirmación de que la prueba no fue consultada.

Si el cambio altera el protocolo escrito, primero deberá actualizarse de manera trazable la metodología. Ningún resultado posterior puede utilizarse como justificación retrospectiva de una decisión previa.

## 8. Condición de cierre de la Fase 0, paso 2

El paso se cerrará únicamente cuando:

1. el investigador apruebe este catálogo, la decisión resuelta y los asuntos aún pendientes;
2. el contrato JSON represente los mismos compromisos sin contradicciones;
3. las pruebas automáticas del contrato sean satisfactorias;
4. README y registro de decisiones enlacen los artefactos;
5. las copias versionadas estén verificadas y privadas en Google Drive.

Las cinco condiciones se cumplieron el 27 de septiembre de 2026. La instrucción del investigador de continuar al siguiente paso se registra como aprobación y la Fase 0, paso 2 queda **cerrada en la versión 1.0**. Esta aprobación tampoco autoriza entrenamiento.
