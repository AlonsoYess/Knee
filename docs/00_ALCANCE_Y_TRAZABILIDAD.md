# Alcance y trazabilidad de la tesis

## Control del documento

| Campo | Valor |
| --- | --- |
| Documento | Contrato de alcance y matriz de trazabilidad de la tesis |
| Estado | Aprobado por el investigador; vigente como contrato rector |
| Versión | 1.1 |
| Fecha de aprobación | 27 de septiembre de 2026 |
| Documento rector del repositorio | `docs/00_ALCANCE_Y_TRAZABILIDAD.md` |
| Destino de respaldo | Copia versionada en el Drive privado autorizado; ubicación exacta en el inventario privado |
| Estado del respaldo | Versión 1.1 sincronizada, verificada y privada en Google Drive |
| Fuentes académicas primarias | `SRC-THESIS-001` para los capítulos I y II; `SRC-THESIS-002` para el capítulo III |
| Datos de referencia | Cohorte tabular OAI consolidada y documentación entregada al 23 de septiembre de 2026 |

Este documento convierte los capítulos I, II y III en un contrato verificable para el desarrollo técnico. No reemplaza a la tesis ni modifica sus enunciados. Su función es impedir desviaciones de alcance, vincular cada objetivo con evidencia concreta y exigir que cualquier mejora metodológica se evalúe y apruebe antes de ejecutarse.

## 1. Reglas de autoridad y uso

1. Los capítulos I y II fijan el problema, los objetivos, las hipótesis, la justificación y las delimitaciones de la investigación.
2. El Capítulo III fija la operación de los datos, el desarrollo, la selección, la evaluación y la integración del prototipo.
3. Una implementación no podrá contradecir ninguno de los tres capítulos. Cuando exista una discrepancia entre ellos, se registrará como decisión pendiente y no se resolverá silenciosamente en el código.
4. Los resultados experimentales no podrán utilizarse para reescribir retrospectivamente una decisión como si hubiese sido adoptada antes del entrenamiento o antes de abrir la prueba.
5. Una propuesta técnicamente superior podrá reemplazar una decisión de implementación solo después de documentar su evidencia, efecto metodológico, costo, riesgo y compatibilidad con los objetivos. Si afecta el protocolo escrito, requerirá aprobación explícita del investigador y actualización trazable de la tesis antes de ejecutarse.
6. Los documentos normativos y metodológicos tendrán una copia versionada en Google Drive. GitHub conservará la versión de trabajo del código y de la documentación pública; Drive conservará el respaldo documental y todos los datos o artefactos privados.
7. Ningún identificador individual, DICOM, manifiesto privado, predicción por caso, credencial o peso de modelo se publicará en el repositorio.

## 2. Contrato central de investigación

### 2.1 Problema general

Determinar en qué medida un sistema multimodal basado en aprendizaje profundo que integre radiografías de rodilla y datos clínicos de la evaluación inicial permite predecir la progresión estructural de la artrosis de rodilla, en comparación con modelos que utilizan cada fuente por separado.

### 2.2 Objetivo general

Desarrollar y evaluar el sistema multimodal descrito y determinar su desempeño frente a un modelo clínico y un modelo radiográfico evaluados bajo las mismas condiciones.

### 2.3 Hipótesis general

- **HG1:** el sistema multimodal presenta desempeño predictivo superior al de ambos modelos unimodales.
- **HG0:** el sistema multimodal no supera a ambos modelos unimodales.

La investigación no presupone que HG1 deba cumplirse. Un resultado en el que el multimodal no supere a uno o a ambos comparadores será un resultado válido y deberá conservarse y explicarse.

### 2.4 Unidad, población y horizonte

| Elemento | Regla comprometida |
| --- | --- |
| Fuente | Osteoarthritis Initiative (OAI) |
| Diseño | No experimental, longitudinal, retrospectivo, cuantitativo y de análisis secundario |
| Unidad de análisis | Rodilla |
| Unidad de agrupación | Participante |
| Evaluación inicial | V00 |
| Seguimiento | V06, horizonte operacional de 48 meses |
| Población principal | Rodillas con KL basal 2 o 3 |
| Entrada temporal permitida | Información disponible en V00 |
| Uso de V06 | Construcción y comprobación del desenlace; nunca predictor |
| Generalización permitida | Cohorte OAI bajo sus protocolos; no se afirmará transportabilidad a población peruana |

### 2.5 Desenlace principal

Para cada rodilla elegible `i`:

```text
PROGRESION_48M(i) = 1, si KL_V06(i) - KL_V00(i) >= 1
PROGRESION_48M(i) = 0, en caso contrario
```

La etiqueta representa progresión radiográfica estructural. No equivale a progresión del dolor, necesidad de tratamiento, empeoramiento clínico global ni indicación quirúrgica.

## 3. Estado de los datos que condiciona el alcance

La cohorte tabular auditada constituye el punto de partida, no la cohorte analítica definitiva.

| Indicador | Estado consolidado |
| --- | ---: |
| Rodillas | 2 778 |
| Participantes | 1 916 |
| Adquisiciones radiográficas bilaterales vinculadas | 1 916 |
| Rodillas progresoras | 473 (17,03 %) |
| Rodillas no progresoras | 2 305 (82,97 %) |
| KL basal 2 | 1 891 |
| KL basal 3 | 887 |
| IMC ausente | 3 rodillas |

El control completo de los 1 916 DICOM y de sus regiones articulares permanece pendiente. Por ello:

- no se congelará todavía el tamaño analítico final;
- no se crearán las particiones definitivas antes de cerrar el control radiográfico;
- no se ejecutará el entrenamiento definitivo;
- cualquier exclusión técnica deberá tener un motivo trazable;
- los modelos principales se compararán sobre la misma cohorte común utilizable.

## 4. Contrato de variables y escenarios

### 4.1 Variables permitidas

| Rol | Variables o fuente | Uso |
| --- | --- | --- |
| Desenlace principal | `KL_BASE`, `KL_48`, `PROGRESION_48M` | Construcción y verificación de la etiqueta |
| Modalidad radiográfica principal | Región tibiofemoral de la radiografía bilateral PA con flexión fija de V00 | Modelo radiográfico y rama visual multimodal |
| Modalidad clínica principal | Edad, sexo e IMC de V00 | Modelo clínico principal y rama clínica multimodal |
| Referencia radiográfica convencional | KL inicial | Solo modelo clínico más KL, como análisis complementario |
| Variante clínica ampliada | Cirugía previa y WOMAC total de V00, además de edad, sexo e IMC | Análisis complementario |
| Sensibilidad | Reemplazo temprano de rodilla | Desenlace compuesto complementario con modelos congelados |

### 4.2 Entradas prohibidas en los modelos principales

- KL de V06.
- Variables obtenidas después de V00.
- Etiqueta de progresión o derivados de ella.
- Reemplazo posterior utilizado como predictor.
- Identificadores o rutas capaces de actuar como atajos del desenlace.
- Datos de la rodilla contralateral colocados en una partición incompatible.
- Resultados de prueba utilizados para escoger recortes, variables, arquitectura, calibrador, hiperparámetros o umbral.

### 4.3 Escenarios comprometidos

| Escenario | Entradas | Familias previstas | Papel |
| --- | --- | --- | --- |
| Clínico principal | Edad, sexo, IMC | Regresión logística regularizada, XGBoost y perceptrón multicapa | Comparador principal |
| Radiográfico | Región articular V00 | DenseNet121 y ViT-B/16 con transferencia | Comparador principal |
| Multimodal principal | Imagen V00, edad, sexo, IMC | Fusión intermedia de representaciones | Escenario central y base del prototipo |
| Clínico más KL | Edad, sexo, IMC, KL inicial | Familia clínica seleccionada | Referencia complementaria |
| Clínico ampliado | Edad, sexo, IMC, cirugía previa, WOMAC | Familia clínica seleccionada | Análisis complementario |
| Fusión tardía | Probabilidades clínica y radiográfica | Promedio fijo 0,5/0,5 y calibración en desarrollo | Contraste complementario |

La comparación confirmatoria corresponde al clínico principal, radiográfico y multimodal principal. Los análisis complementarios no podrán sustituir retrospectivamente a estos escenarios por producir un resultado más favorable.

## 5. Matriz de trazabilidad académica y técnica

### 5.1 OE1, PE1 y HE1: conjunto longitudinal

| Componente | Definición operacional |
| --- | --- |
| Pregunta | Cómo estructurar radiografías y datos clínicos manteniendo participante, rodilla y visita |
| Objetivo | Construir y preparar el conjunto longitudinal correctamente vinculado |
| Hipótesis | Los registros OAI permiten conformarlo con correspondencia verificable |
| Entradas | CSV de cohorte, manifiesto, DICOM V00, lecturas KL V00/V06, fechas y lateralidad |
| Trabajo exigido | Auditoría tabular; inventario DICOM; control de lectura; lateralidad; localización articular; trazabilidad de exclusiones; cierre de cohorte común |
| Evidencia | Clave participante-rodilla única; fechas coherentes; etiqueta reproducible; imagen y región aceptadas; informe de exclusiones; huellas de archivos |
| Entregables | Cohorte analítica versionada, manifiesto radiográfico, reporte de calidad y diagrama de flujo de selección |
| Criterio de aceptación | Ninguna rodilla utilizada carece de vínculo explicable entre V00, V06, lado, datos clínicos e imagen; toda exclusión tiene motivo |

### 5.2 OE2, PE2 y HE2: modelo radiográfico

| Componente | Definición operacional |
| --- | --- |
| Pregunta | Cómo predecir progresión usando la radiografía inicial |
| Objetivo | Desarrollar el modelo unimodal radiográfico |
| Hipótesis | Capacidad discriminativa superior al azar |
| Entradas | Región articular V00 procesada de forma reproducible |
| Familias | DenseNet121 y ViT-B/16 con transferencia de aprendizaje |
| Evidencia de desarrollo | Predicciones fuera de pliegue, curvas, configuraciones, semillas, consumo de recursos y ausencia de fuga por participante |
| Evidencia confirmatoria | ROC-AUC de prueba frente a 0,5, intervalo bootstrap por participante y contraste ajustado según el protocolo |
| Entregables | Pipeline radiográfico, configuración seleccionada, pesos privados, predicciones y reporte de evaluación |
| Criterio de aceptación | El pipeline reproduce el mismo preprocesamiento en desarrollo e inferencia y la evaluación usa participantes no vistos |

### 5.3 OE3, PE3 y HE3: modelo clínico

| Componente | Definición operacional |
| --- | --- |
| Pregunta | Cómo predecir progresión usando datos clínicos iniciales |
| Objetivo | Desarrollar el modelo unimodal clínico |
| Hipótesis | Capacidad discriminativa superior al azar |
| Entradas principales | Edad, sexo e IMC de V00 |
| Familias | Regresión logística regularizada, XGBoost y perceptrón multicapa |
| Tratamiento | Imputación, codificación y escalamiento ajustados solo con el entrenamiento de cada pliegue |
| Evidencia confirmatoria | ROC-AUC de prueba frente a 0,5 bajo el mismo procedimiento de HE2 |
| Entregables | Pipeline clínico, transformaciones serializadas, configuración seleccionada, predicciones y reporte |
| Criterio de aceptación | No existe ajuste de transformaciones con validación o prueba; el modelo usa exclusivamente predictores autorizados |

### 5.4 OE4, PE4 y HE4: integración multimodal

| Componente | Definición operacional |
| --- | --- |
| Pregunta | Cómo integrar representaciones radiográficas y clínicas |
| Objetivo | Desarrollar el modelo multimodal |
| Hipótesis | Capacidad discriminativa superior al azar |
| Entradas | Región articular V00, edad, sexo e IMC de V00 |
| Estrategia principal | Fusión intermedia de representación visual y rama clínica densa |
| Control de fuga | En cada pliegue, los pesos adaptados a OAI solo pueden proceder de participantes de entrenamiento |
| Evidencia confirmatoria | ROC-AUC de prueba frente a 0,5 bajo el mismo procedimiento de HE2 y HE3 |
| Entregables | Pipeline multimodal, arquitectura, pesos privados, predicciones, explicaciones y paquete de inferencia |
| Criterio de aceptación | La fusión comparte cohorte, particiones, desenlace y evaluación con los dos modelos unimodales |

### 5.5 OE5, PE5, HE5 y HG: comparación principal

| Componente | Definición operacional |
| --- | --- |
| Pregunta | Qué diferencias de desempeño existen entre los tres enfoques |
| Objetivo | Comparar el multimodal con ambos unimodales |
| Hipótesis | El multimodal seleccionado supera al radiográfico y al clínico en la misma prueba |
| Comparaciones confirmatorias | Multimodal menos clínico; multimodal menos radiográfico |
| Métrica principal | PR-AUC como precisión promedio no interpolada, mediante `average_precision_score` |
| Incertidumbre | 2 000 remuestras bootstrap por participante, incorporando juntas sus rodillas |
| Multiplicidad | Ajuste secuencial de Holm para los dos contrastes principales |
| Regla de superioridad | Diferencia de PR-AUC con IC 95 % completamente positivo y valor p ajustado menor de 0,05 |
| Regla para HG1 | Debe cumplirse la superioridad frente a los dos comparadores; superar solo a uno no basta |
| Entregables | Tabla comparativa, diferencias pareadas, intervalos, valores p ajustados, curvas y análisis de errores |
| Criterio de aceptación | Las predicciones están congeladas, corresponden a las mismas rodillas y la prueba no participó en ninguna decisión |

## 6. Contrato de partición, selección y evaluación

### 6.1 Partición

- La asignación se realizará por participante, no por rodilla.
- Se utilizará semilla 2026.
- El bloque 0, aproximadamente 20 %, será la prueba reservada.
- El resto constituirá desarrollo.
- Dentro de desarrollo se realizarán cinco pliegues agrupados y aproximadamente estratificados.
- Los participantes usados para ajustar el preprocesamiento radiográfico permanecerán en desarrollo.
- No se probarán semillas sucesivas para escoger una división más conveniente.
- Todos los escenarios compartirán las mismas asignaciones.

### 6.2 Selección de configuraciones

- La selección ocurrirá exclusivamente dentro de desarrollo.
- La métrica de selección será la PR-AUC media de los cinco pliegues.
- Una diferencia menor de 0,01 en PR-AUC se desempatará por ROC-AUC y luego por Brier.
- Se conservarán todos los candidatos, no solo el ganador.
- Las redes seleccionadas se ejecutarán con semillas 2026, 2027 y 2028; no se elegirá la mejor semilla según prueba.
- El número final de épocas se fijará a partir de desarrollo, sin detener por prueba.

### 6.3 Desbalance, calibración y umbral

- Las redes utilizarán entropía cruzada binaria ponderada con peso calculado solo en el entrenamiento correspondiente.
- No se alterará artificialmente la frecuencia de validación o prueba.
- No se aplicará SMOTE a radiografías.
- La calibración se ajustará con predicciones fuera de pliegue de desarrollo.
- Las redes usarán escalado por temperatura sobre logits; los modelos clínicos convencionales, calibración logística.
- El umbral se fijará en desarrollo maximizando el índice de Youden, con preferencia por sensibilidad en caso de empate.
- Calibrador y umbral quedarán congelados antes de abrir la prueba.

### 6.4 Métricas y evidencia secundaria

Además de la PR-AUC principal, se reportarán según corresponda:

- ROC-AUC;
- sensibilidad y especificidad;
- precisión, F1 y exactitud balanceada;
- matriz de confusión;
- Brier;
- curva, pendiente e intercepto de calibración;
- intervalos de confianza por participante;
- estabilidad entre pliegues y semillas;
- tiempo, memoria, parámetros y tamaño de artefactos.

Una mejora estadística no se presentará automáticamente como utilidad clínica. La calibración, los falsos negativos, la procedencia OAI y la ausencia de validación externa deberán acompañar la interpretación.

## 7. Interpretabilidad y prototipo

### 7.1 Interpretabilidad

- DenseNet utilizará Grad-CAM.
- ViT utilizará attention rollout si resulta seleccionado.
- La rama clínica utilizará SHAP.
- Las explicaciones visuales y clínicas se presentarán por separado.
- Ninguna explicación se interpretará como causalidad o confirmación de una lesión.
- También se conservarán explicaciones de errores, no únicamente ejemplos favorables.

### 7.2 Prototipo comprometido

El prototipo será académico y demostrativo. Deberá:

- recibir una radiografía basal, lateralidad, edad, sexo e IMC;
- reutilizar exactamente el preprocesamiento congelado;
- mostrar probabilidad calibrada de progresión estructural a 48 meses y clasificación según el umbral fijado;
- advertir la procedencia OAI, la población KL 2/3 y las limitaciones;
- rechazar entradas cuya lateralidad o región articular no puedan resolverse con confianza;
- reproducir la predicción experimental con diferencia absoluta máxima de `1e-5`.

No emitirá diagnósticos, recomendaciones terapéuticas ni decisiones autónomas, ni se presentará como dispositivo médico o sistema hospitalario en producción.

### 7.3 Decisión cerrada sobre el KL estimado

El Capítulo I, apartado 1.5.2, prometía mostrar adicionalmente un grado KL estimado. El Capítulo III no define datos, arquitectura, entrenamiento, validación ni métricas para esa tarea. El 27 de septiembre de 2026 el investigador decidió **retirar esa promesa** en lugar de agregar un segundo problema de modelado.

En consecuencia:

- el prototipo no estimará ni mostrará un grado KL;
- su salida se limitará a la probabilidad calibrada de progresión estructural a 48 meses y a la clasificación derivada del umbral fijado en desarrollo;
- no se entrenará, evaluará ni integrará un clasificador o regresor de KL;
- la decisión no modifica el desenlace, los objetivos, las hipótesis ni las comparaciones principales de la metodología;
- la frase correspondiente deberá eliminarse del apartado 1.5.2 del documento de tesis antes de consolidar la siguiente versión del capítulo I.

Esta decisión cierra la discrepancia metodológica. La corrección material del archivo Word queda como acción documental pendiente y no autoriza alterar otras delimitaciones del capítulo I.

## 8. Exclusiones explícitas del alcance

Quedan fuera del estudio principal:

- diagnóstico inicial de artrosis en población general;
- rodillas KL 0, KL 1 o KL 4 al inicio;
- predicción de dolor, limitación funcional o decisión quirúrgica como desenlace principal;
- predicción causal o explicación causal de la progresión;
- validación externa no confirmada;
- afirmaciones de aplicabilidad directa a pacientes peruanos;
- uso clínico autónomo, certificación de dispositivo médico o integración hospitalaria productiva;
- ajuste de decisiones con información del conjunto de prueba;
- sustitución del contraste principal por un análisis complementario más favorable.

## 9. Clasificación de decisiones y control de cambios

### 9.1 Compromisos congelados

Requieren modificación académica explícita para cambiar:

- problema, objetivos e hipótesis;
- unidad de análisis y agrupación por participante;
- población KL 2/3;
- horizonte V00-V06;
- definición del desenlace;
- modalidades principales y comparadores;
- cohorte común;
- separación desarrollo-prueba;
- PR-AUC (AP) como métrica principal;
- comparaciones confirmatorias y regla de superioridad;
- carácter académico y no clínico del prototipo.

### 9.2 Decisiones optimizables dentro del protocolo

Pueden mejorarse en desarrollo, con registro previo y sin acceder a prueba:

- parámetros exactos de localización y recorte;
- hiperparámetros dentro del presupuesto;
- número de capas descongeladas;
- tasas de aprendizaje, regularización y lote;
- implementación eficiente y precisión mixta;
- organización del código, pruebas, formatos de artefactos y automatización;
- reducción razonada del espacio de búsqueda por limitaciones de cómputo.

### 9.3 Cambios metodológicos que requieren aprobación previa

Entre otros:

- sustituir DenseNet121 o ViT-B/16;
- cambiar la resolución 224 x 224;
- agregar nuevas variables al modelo principal;
- modificar la pérdida, la estrategia de balance, la calibración o el umbral;
- cambiar el esquema de partición o las semillas comprometidas;
- introducir aprendizaje auto-supervisado, ensambles adicionales o modelos fundacionales;
- modificar las comparaciones confirmatorias;
- agregar el estimador KL al prototipo.

Cada propuesta deberá registrarse antes de ejecutarse con:

1. problema que intenta resolver;
2. alternativa propuesta;
3. evidencia técnica o científica;
4. efecto sobre capítulos, hipótesis y comparabilidad;
5. efecto sobre tiempo, cómputo y datos;
6. riesgos de sesgo o fuga;
7. decisión del investigador;
8. versión desde la que aplica.

## 10. Gestión documental en Google Drive

Todos los documentos de dirección y ejecución tendrán una copia versionada en el Drive privado autorizado. La ubicación, los nombres exactos, los identificadores y las huellas de cada versión se mantienen exclusivamente en el inventario privado.

Para evitar versiones divergentes:

- GitHub conservará el Markdown editable y su historial técnico;
- Drive conservará una copia versionada de cada versión aprobada;
- una versión aprobada no se sobrescribirá: se generará `v0.2`, `v1.0`, etc.;
- el inventario privado deberá registrar nombre, versión, fecha, estado, ubicación Git, ubicación Drive y SHA-256;
- los documentos que incluyan identificadores o rutas privadas existirán únicamente en Drive;
- los notebooks temporales de Colab no serán la única copia de ninguna decisión.

La conexión autorizada de Google Drive se verificó el 27 de septiembre de 2026. Las versiones superadas se conservan en un historial privado sin sobrescribirse. Cada sincronización se comprueba por metadatos para confirmar tipo, ubicación y estado privado. La ausencia de una unidad local montada no impide el respaldo mediante la conexión autorizada.

## 11. Condición de cierre de la Fase 0, paso 1

Este paso podrá marcarse como cerrado cuando:

1. el investigador revise y apruebe este contrato;
2. las discrepancias identificadas permanezcan registradas, sin decisiones implícitas;
3. el documento esté enlazado desde el README;
4. exista una copia versionada y verificada en Google Drive;
5. la regla de control de cambios esté aceptada para las fases siguientes.

Las cinco condiciones se cumplieron el 27 de septiembre de 2026 y este paso quedó **cerrado en la versión 1.0**. La instrucción del investigador de continuar al siguiente paso se registra como aprobación del contrato y de su regla de control de cambios. Esta aprobación no autoriza todavía el entrenamiento.
