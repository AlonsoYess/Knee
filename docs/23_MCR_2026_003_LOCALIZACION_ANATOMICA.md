# MCR-2026-003 — Localización tibiofemoral mediante puntos anatómicos

> Estado vigente: retirada sin aplicación por aprobación expresa de MCR-2026-004. La aprobación del 30 de septiembre y el contenido técnico siguiente son históricos, no autorización vigente. Véase el [acta 26](26_APROBACION_MCR_2026_004_Y_CIERRE_PENDIENTE.md).

## 1. Identificación

| Campo | Valor |
| --- | --- |
| ID | `MCR-2026-003` |
| Clase | `M2`: modificación del procedimiento escrito en el capítulo III |
| Versión documental | 1.2, retirada por sustitución; contenido técnico 1.0 preservado |
| Registro | 30 de septiembre de 2026 |
| Proponente | Codex, por solicitud del investigador |
| Estado | `retirada`, sustituida por MCR-2026-004 el 1 de octubre de 2026 |
| Decisión del investigador | Aprobada expresamente el 30 de septiembre de 2026, bajo las condiciones G0–G6 |
| Paso | Fase 1, paso 4: localización tibiofemoral y recorte |
| Fuente académica contrastada | `SRC-THESIS-002`, apartado 3.2.1, control de calidad y preparación radiográfica |
| Reglas relacionadas | `INV-RAD-001`, `INV-UNIT-001`, `INV-COH-001`, `INV-SPLIT-001`, `INV-CV-001`, `INV-TEST-001`, `INV-TRACE-001`, `INV-PRIV-001` |
| Registro verificable | `configs/governance/methodology_change_log.json` |

M2 es la **clase de impacto**, no el número de solicitud: ya existen `MCR-2026-001` y `MCR-2026-002`. La versión 1.0 se preparó sin autorización de implementación. La versión documental 1.1 registra la aprobación posterior sin cambiar sus alternativas, geometría, muestra, umbrales ni condiciones. Los contratos registran ahora el cambio autorizado; el código del localizador y los Word originales no se modificaron. El [acta de comprobaciones previas](24_APROBACION_M2_Y_COMPROBACIONES_PREVIAS.md) distingue autorización de disponibilidad para integrar.

Se propone sustituir la búsqueda basada en perfiles por un localizador anatómico aprendido **preentrenado y congelado**, con KNEEL como candidato preferente sujeto a elegibilidad. El alcance propuesto comprende auditar sus pesos, integrar una inferencia reproducible y evaluar recortes con referencias humanas y una muestra técnica adicional. No incluye entrenamiento o ajuste fino, ni un cambio en los modelos de predicción de progresión. Actualmente no existe un checkpoint declarado elegible para este proyecto.

## 2. Problema observado

El capítulo III, apartado 3.2.1, prescribe localizar la articulación mediante perfiles de intensidad y gradiente en la mitad unilateral. Su ventana proporcional debe contener espacio articular y extremos óseos adyacentes; los parámetros se fijan en desarrollo y los casos dudosos se revisan sin mostrar el desenlace. El campo físico de **140 × 140 mm** es una concreción del protocolo operativo del paso 4, no una cifra impuesta por ese párrafo del Word.

Las cuatro implementaciones evaluadas sobre las mismas diez adquisiciones no superaron el criterio integral:

| Versión | Aceptables | Rechazados | Fuente local |
| --- | ---: | ---: | --- |
| v0.1 | 9/20 | 11/20 | [Acta v0.1](15_REVISION_FASE_1_PASO_4_V01.md) |
| v0.2 | 8/20 | 12/20 | [Acta v0.2](16_REVISION_FASE_1_PASO_4_V02.md) |
| v0.3 | 0/20 | 20/20 | [Acta v0.3](19_REVISION_FASE_1_PASO_4_V03.md) |
| v0.4 | 8/20 | 12/20 | [Acta v0.4](22_REVISION_FASE_1_PASO_4_V04.md) |

En v0.4 hubo nueve errores de centrado y seis recortes con puntos de la regla; tres presentaron ambos defectos. No se registraron pérdidas anatómicas ni exclusiones técnicas. La ejecución del piloto se identifica con `c8460c1e4feea45e02d955254a05afcfa0198f09`. La revisión está documentada; **el cierre operativo sigue pendiente de ejecutar y contrastar la libreta 12**. M2 no transforma esa preparación en una ejecución realizada.

La evidencia demuestra insuficiencia de estas implementaciones en el piloto. No demuestra que todos los métodos deterministas fallen, que una red sea necesariamente superior, ni que v0.4 tenga una tasa poblacional de éxito del 40 %. Las veinte rodillas fueron reutilizadas para desarrollo y proceden de una selección inicial enriquecida por desenlace; no son una muestra independiente representativa. El cegamiento de las vistas no cambia cómo se seleccionó inicialmente el piloto.

## 3. Evidencia científica y alternativas

Revisión dirigida de fuentes primarias consultadas el 30 de septiembre de 2026; no se presenta como revisión sistemática. Se buscaron métodos sobre radiografía de rodilla y reutilización verificable de código, no únicamente trabajos cuyo título incluyera «tesis».

| Fuente | Evidencia verificada | Aplicabilidad y límite |
| --- | --- | --- |
| [Tiulpin et al., 2019, progresión multimodal](https://www.nature.com/articles/s41598-019-56527-3) | Empleó BoneFinder para puntos anatómicos, recortes de 140 mm y alineación del platillo tibial en OAI/MOST. | Antecedente directamente relacionado con progresión. Su horizonte, variables y modelo difieren de esta tesis; no se importan sus resultados ni su protocolo completo. |
| [KNEEL, Tiulpin et al., ICCVW 2019](https://arxiv.org/pdf/1907.12237) | Arquitectura hourglass con soft-argmax, localización gruesa y 16 puntos. Entrenamiento/selección sobre 748 rodillas OAI; evaluación en dos conjuntos de Oulu. El artículo señala que el BoneFinder comparador fue entrenado con 500 imágenes OAI sin disponer de su listado. | Fundamenta el candidato anatómico y obliga a auditar solapamiento. No acredita que el checkpoint actualmente distribuido coincida con el del artículo. |
| [Repositorio oficial KNEEL](https://github.com/imedslab/KNEEL) y [ficha de pesos](https://huggingface.co/imedslab/kneel) | Existe paquete de inferencia y acceso a pesos condicionado a aceptar términos. El README restringe uso comercial; la ficha indica pesos actualizados y una etiqueta de licencia. | Código visible no equivale a permisos de redistribución de pesos ni a trazabilidad completa de entrenamiento. No se aceptaron términos, solicitaron accesos ni descargaron pesos en esta revisión. |
| [BoneFinder, sitio del proveedor](https://bone-finder.com/) | Herramienta aprendida mediante regresión por bosques; licencia de investigación no comercial solicitada por formulario. El paquete por defecto corresponde al fémur proximal; otros modelos requieren contacto. | Comparador anatómico sólido, pero modelo de rodilla, integración, derechos y procedencia siguen pendientes. No se asume disponible en Colab. |
| [Chavoshi et al., 2026](https://link.springer.com/article/10.1007/s10278-026-01961-9) y [knee-crop oficial](https://github.com/Emory-HITI/knee-crop) | El resumen compara un método por intensidad con SVM y aprendizaje profundo; el repositorio ofrece recorte determinista sobre PNG y declara licencia MIT. | Alternativa relevante frente al supuesto de que mayor complejidad siempre es mejor. Solo se accedió al resumen y secciones públicas del artículo: la declaración de datos menciona simulaciones mientras el resumen menciona OAI/MRKR. Esa inconsistencia requiere aclaración antes de usar sus cifras como evidencia fuerte. |

### Alternativas evaluadas y decisión técnica propuesta

| Alternativa | Ventaja | Coste o riesgo | Disposición en M2 |
| --- | --- | --- | --- |
| Seguir afinando v0.4 sobre veinte rodillas | Implementación conocida | Reutilización del mismo piloto y falta de mejora integral | No seleccionada; conservar como comparador histórico |
| KNEEL congelado | Puntos inspeccionables y candidato específico de rodilla | Pesos entrenados con OAI o procedencia desconocida; integración y licencia | Preferente **solo si supera la puerta de elegibilidad** |
| BoneFinder congelado | Antecedente directo en progresión | Acceso al modelo de rodilla y procedencia no resueltos | Alternativa documentada; no sustituto automático si KNEEL falla |
| knee-crop publicado | Sin entrenamiento local; código público | Diferente canal de entrada, geometría y normalización; evidencia editorial por aclarar | Alternativa reproducible para otra decisión comparativa; no incorporada silenciosamente como v0.5 |
| Detector genérico de cajas, por ejemplo YOLO | Localización global rápida | Una caja no valida por sí sola interlínea ni conservación de ambos compartimentos | No seleccionado para este objetivo de puntos y centro físico |
| Entrenar localizador propio de puntos | Control sobre sujetos de entrenamiento | Anotación experta, muestra auxiliar, entrenamiento y validación adicionales | Contingencia que exigiría nueva versión de la solicitud y presupuesto previo |
| Recorte manual de toda la cohorte | Referencia humana accesible | No reproduce el procedimiento automático del prototipo | Solo anotación de referencia; no producción de entradas privilegiadas |

La recomendación se basa en adecuación anatómica, auditabilidad y coste de reutilización. No se declara que KNEEL sea la mejor solución hasta evaluarlo bajo los mismos criterios. Si no resulta elegible, se informa ese resultado y se revisa la propuesta; no se promueve otro candidato por continuidad implícita.

## 4. Cambio propuesto

### 4.1 Puerta previa: procedencia y condiciones de uso

Antes de inferir sobre las radiografías del proyecto se deberá registrar:

1. Revisión inmutable del código, archivo de pesos, SHA-256, dependencias, licencia exacta de código y pesos y condiciones aplicables al prototipo académico. La ficha pública no sustituye el texto de licencia.
2. Cohortes, sujetos y visitas usados para entrenamiento, ajuste, selección y cualquier actualización del checkpoint, incluidos modelos previos de la cadena. No basta comprobar el conjunto del artículo de 2019.
3. Cruce privado de sujetos y, cuando existan, huellas de imágenes, contra toda la cohorte principal y el universo candidato de sensibilidad por reemplazo. Comparar por participante y todas sus visitas, no solo por rodilla o archivo V00.
4. Evidencia de **ausencia de solapamiento**. Si faltan identificadores o una garantía de procedencia verificable suficiente, se registra `provenance_unresolved` y ese checkpoint no se declara elegible. Ausencia de información no significa ausencia de fuga. No se eliminan participantes de la tesis para acomodar pesos externos.
5. Ejecución en infraestructura privada controlada, sin enviar DICOM a un servicio público. Las credenciales de descarga no se guardan en notebooks, Git ni logs.

Exigir disyunción frente a toda la cohorte es una decisión conservadora de esta propuesta para mantener un preprocesador fijo en todos los pliegues. Evita usar anotaciones de sujetos de validación para aprender el localizador. Un preentrenamiento anatómico solapado no equivale automáticamente a haber visto el desenlace, pero impide la independencia que se exige aquí. Si no puede cumplirse, cualquier estrategia distinta deberá definir su validación por pliegues antes de autorizarla.

### 4.2 Flujo y geometría

La cadena propuesta es: DICOM V00 → normalización de trabajo documentada → separación/lateralidad ya congeladas → inferencia anatómica → centro y caja físicos → controles de calidad → salida aceptada o abstención.

- Se conserva el original de 16 bits y la transformación por imagen vigente. La entrada interna que requieran los pesos tendrá su transformación propia, reversible para coordenadas y separada de la imagen que recibirá el predictor. No se aplica dos veces normalización ni se cambia la entrada clínica.
- El adaptador debe utilizar el campo unilateral del paso 3. Si el paquete externo solo permite otra separación bilateral, se estudia esa incompatibilidad y se detiene la integración hasta resolverla; no se reemplaza el paso 3 sin documentarlo.
- Se versiona la correspondencia de los 16 puntos con su definición anatómica y su transformación al DICOM. Un experto debe verificar los índices, el reflejo izquierda/derecha y las curvas antes de utilizarlos; no se adivina la semántica de un tensor.
- Se propone construir perfiles tibial y femoral por interpolación lineal de los puntos de sus superficies articulares, con dominio común. En posiciones de 25 % y 75 % del ancho del platillo tibial se obtienen los puntos medios entre ambas superficies; el centro de recorte es el promedio de esos dos puntos. Esta es una **regla geométrica propuesta para la tesis**, no un resultado atribuido a KNEEL. Si sus puntos no soportan esas superficies, se declara incompatibilidad; no se cambia la regla durante la evaluación.
- Se calcula en milímetros, respetando los dos espaciados y el orden fila/columna. La caja final mantiene 140 × 140 mm, alineada con los ejes del DICOM. No se añade rotación, escala anatómica variable, relleno negro ni enmascaramiento. El error máximo por redondeo del tamaño será un píxel de origen por eje y se registrará.
- La línea roja del visor será un indicador del nivel del centro; no se presentará como segmentación exacta de una interlínea curva. La región debe conservar ambos compartimentos y los extremos óseos adyacentes acordados en el manual de anotación.
- La salida del predictor permanece en 224 × 224. La resolución interna del localizador debe corresponder al checkpoint y quedar registrada; no es una nueva resolución experimental del clasificador.

### 4.3 Factibilidad del campo y artefactos

Antes de integrar el candidato se comprueba sobre las veinte rodillas, usando referencias humanas, si la caja de 140 mm centrada por la regla anterior conserva anatomía y excluye artefactos. Esta prueba usa anotaciones para evaluar **factibilidad**, no para generar los recortes que luego se darán al predictor.

Si el campo invade la regla incluso con centro de referencia correcto, el problema es geométrico y no se resuelve cambiando pesos. Se detiene M2 en esa puerta y se propone por separado una enmienda de tamaño, máscara o criterio de aceptación. No se reduce el campo por caso ni se relajan los requisitos para declarar éxito.

Las puertas independientes verifican integridad, lateralidad, puntos finitos dentro del campo, orden tibia/fémur, cobertura de ambos compartimentos, tamaño físico, ausencia de texto/regla/rectángulos/bordes y confianza suficiente. Un fallo produce abstención y motivo trazable. La confianza del modelo no sustituye estas puertas; no se interpreta una salida soft-argmax como probabilidad calibrada de localización correcta.

## 5. Referencia humana y validación preespecificada

### 5.1 Anotación

Un lector con experiencia en radiografía de rodilla y un segundo lector capacitado realizarán anotaciones independientes; las discrepancias las adjudicará un experto. Esta disponibilidad es un recurso **por confirmar**, no un compromiso ya obtenido. El investigador de sistemas puede preparar el visor y la trazabilidad; una valoración automática del asistente no sustituye al lector anatómico.

El manual identificará puntos, superficies, márgenes anatómicos, tratamiento de superposiciones y los casos no evaluables. Se registrarán coordenadas originales, visibilidad, confianza del lector, caja anatómica de referencia y máscaras de artefactos periféricos. Los puntos no visibles se marcarán como tales, sin inventar coordenadas. Se documentarán desacuerdos en milímetros antes de adjudicar. Los lectores verán V00 sin desenlace, datos futuros, predicción de progresión ni superposición del candidato durante la anotación inicial; el orden de revisión de recortes se aleatorizará y no mostrará la versión.

Las anotaciones son referencia de evaluación. No entrenan los pesos congelados ni corrigen individualmente los recortes de producción. No se descargarán imágenes privadas para servicios externos de anotación.

### 5.2 Conjuntos y orden

| Conjunto | Tamaño propuesto | Función | Restricción |
| --- | --- | --- | --- |
| Piloto histórico | 10 participantes, 20 rodillas | Definir manual, comprobar factibilidad, integrar adaptador y regresión | Ya utilizado repetidamente; desarrollo permanente; no estimar generalización |
| Calificación técnica nueva | 60 participantes adicionales, 120 rodillas previstas | Evaluar una versión y puertas previamente congeladas | Participantes distintos del piloto, sin ajuste tras ver resultados; desarrollo permanente |
| Cohorte principal restante | La que corresponda tras calidad | Procesamiento posterior, comparación predictiva | No se procesa con el nuevo método mediante la mera preparación o aprobación documental de M2 |

Los 60 participantes se seleccionarán una sola vez del inventario V00 restante mediante muestreo aleatorio por participante con semilla 2026, sobre lista privada ordenada y versionada, sin consultar V06 ni progresión. Se revisarán ambas rodillas para evaluar la transformación bilateral; se marcará por separado cuáles son elegibles para el análisis principal. Los fallos y no evaluables permanecen en el denominador técnico; no se sustituyen sujetos para mejorar aceptación. Si ya existiese un manifiesto de prueba congelado, solo se muestreará del desarrollo y se demostrará la disyunción.

Se propone reservar desde la selección esos 60 participantes, además de los 10 del piloto, para desarrollo. Es una reserva de uso técnico, no la creación del bloque 0 ni de los pliegues definitivos. El capítulo III ya exige que los sujetos usados para ajustar preprocesamiento queden en desarrollo. Se comprobará posteriormente el efecto sobre tamaño, representatividad y balance; no se redibujan semillas para favorecer resultados.

Se congelan código, pesos, manual, umbrales y decisión de candidato antes de mostrar sus resultados en los 60 participantes. Si se usa esa muestra para modificar el método, pasa a ser desarrollo exploratorio y pierde su condición de evaluación no utilizada para ajuste. Repetir una calificación requiere nueva versión y muestra autorizadas. La separación interna no convierte esta evaluación OAI en validación externa.

### 5.3 Métricas y umbrales propuestos

Estos umbrales son criterios de ingeniería **propuestos para aprobación**, no estándares clínicos ni resultados obtenidos. Se fijarán antes de la ejecución y no se elegirán para que las salidas conocidas aprueben.

| Medición | Regla propuesta |
| --- | --- |
| Error del centro | Distancia euclídea en mm frente al centro derivado de referencia; ≤ 3 mm por rodilla |
| Puntos anatómicos | Mediana de error ≤ 2 mm y percentil 95 ≤ 5 mm en puntos visibles adjudicados; informar además porcentaje a 2 y 5 mm y faltantes |
| Integridad y lateralidad | Cero errores de lectura atribuibles al pipeline, intercambio de lados o transformaciones de coordenadas |
| Recorte integral | Centro admisible, anatomía completa y ausencia de artefactos periféricos; doble revisión y adjudicación |
| Regresión histórica | 20/20 recortes integralmente aceptables; sin excluir retrospectivamente fallos del localizador |
| Calificación nueva | 120/120 recortes integralmente aceptables y 60/60 participantes con ambas rodillas aceptadas |
| Seguridad del rechazo | Cero recortes incorrectos aceptados por las puertas automáticas; registrar falsos rechazos y abstenciones |
| Reproducibilidad | Mismos datos/configuración/pesos reproducen puntos y cajas dentro de tolerancia numérica declarada; máxima diferencia de centro 0,1 mm |

Se reportan todos los fallos; una abstención no cuenta como recorte correcto. Se informa tanto cobertura automática como calidad entre los aceptados, evitando que rechazar todo parezca buen desempeño. Para avanzar no bastará aprobar la revisión humana si el prototipo se abstiene en parte del conjunto: debe producir automáticamente los recortes aceptables exigidos. IoU de cajas se reportará como indicador secundario, no sustituto de centrado y cobertura anatómica.

Con 60 éxitos de 60 participantes, el límite inferior binomial unilateral del 95 % sería aproximadamente 95,1 % (`0.05 ** (1/60)`), bajo un modelo de participantes independientes. Esto justifica el tamaño como demostración técnica acotada; no garantiza esa tasa en toda OAI. Las dos rodillas no se tratarán como 120 observaciones independientes. Se reportarán intervalos descriptivos por participante y resultados por modalidad/espaciado/lateralidad cuando haya tamaños suficientes; no se presentará potencia para subgrupos pequeños ni superioridad predictiva.

## 6. Impacto académico y metodológico

| Dimensión | Impacto propuesto | Actualización necesaria tras aprobación |
| --- | --- | --- |
| Capítulo I, problemas, objetivos e hipótesis | Sin cambio: sigue predicción multimodal de progresión a 48 meses | Comprobar coherencia, sin nueva promesa clínica |
| Capítulo II | Incorporar antecedentes de localización anatómica y sus limitaciones | Añadir solo fuentes utilizadas y verificadas |
| Capítulo III, 3.2.1 | Sustituir perfiles por localizador congelado, control de procedencia, anotación y evaluación técnica ampliada | Enmienda trazable antes de ejecutar M2 |
| Cronograma y presupuesto | Añadir revisión experta, auditoría de pesos e integración | Estimar con mediciones iniciales; no asumir disponibilidad gratuita |
| Cohorte y exclusiones | Mantener elegibilidad; distinguir fallo del método de imagen no evaluable | Mismo manifiesto de calidad para los tres escenarios; informar exclusiones |
| Predictores, desenlace y horizonte | Sin cambio; puntos usados para recortar, no nuevos predictores | Preservar `scope_contract.json` en esos campos |
| Particiones | Reservar los sujetos técnicos para desarrollo antes de la partición definitiva | Registro privado de uso y auditoría de disyunción; bloque 0 y cinco pliegues intactos en definición |
| Transformaciones aprendidas | Preprocesador externo congelado y disjunto de toda la cohorte | Documentar la excepción al ajuste dentro del pliegue de 3.2.1; si se ajusta localmente deja de cumplir M2 |
| Comparadores y métricas predictivas | Sin cambio de familias ni de AP, calibración, umbral o contrastes | No usar PR-AUC para escoger localizador |
| Inferencia estadística | Mantener bootstrap por participante y Holm del análisis predictivo | Métricas técnicas separadas de las confirmatorias |
| Interpretabilidad y prototipo | Reutilizar el mismo localizador y abstención; auditar artefactos | Conservar trazabilidad de coordenadas; no ofrecer diagnóstico o KL |
| Documentos rectores | Especificar nuevo procedimiento de preprocesamiento y su estado | `00_ALCANCE_Y_TRAZABILIDAD`, `01_REGLAS_INVARIABLES`, `14_FASE_1_PASO_4`, plan y contrato ejecutable |

La clase propuesta es M2 porque modifica un procedimiento de 3.2.1 y sus controles, manteniendo población, objetivo e inferencia. Si la solución requiere modificar la cohorte, aprovechar sujetos de prueba o cambiar el análisis confirmatorio, deberá reevaluarse la clase y tramitarse una solicitud distinta.

### Texto de enmienda propuesto para 3.2.1

Texto para revisar e incorporar **después de aprobación**, no modificación ya realizada del Word:

> La localización tibiofemoral se realizará mediante un modelo preentrenado de puntos anatómicos, cuyos pesos permanecerán congelados. Su utilización estará condicionada a comprobar las condiciones de uso, la procedencia y la ausencia de solapamiento de participantes con la cohorte de estudio y el universo candidato del análisis de sensibilidad. Los puntos se expresarán en el sistema de coordenadas del DICOM y se emplearán exclusivamente para construir el recorte tibiofemoral de la rodilla correspondiente, mediante una regla geométrica versionada. El campo candidato de 140 × 140 mm se someterá a comprobación de factibilidad anatómica y exclusión de artefactos, sin correcciones manuales de los recortes de producción.
>
> La evaluación técnica empleará anotaciones anatómicas independientes con adjudicación experta y revisión ciega al desenlace. El piloto histórico se utilizará para desarrollo y regresión; una muestra adicional de 60 participantes se reservará para evaluar una versión previamente congelada. Todos los participantes utilizados en este proceso permanecerán en desarrollo. Se reportarán errores en milímetros, aceptación integral, abstenciones y fallos por participante. Los umbrales y la muestra se fijarán antes de la evaluación adicional. Las imágenes no localizables producirán un estado explícito de rechazo y el prototipo aplicará el mismo procedimiento automático. La prueba predictiva reservada permanecerá cerrada.
>
> Este localizador externo congelado no se ajustará con imágenes ni anotaciones de la cohorte. Las demás transformaciones aprendidas seguirán estimándose exclusivamente dentro del entrenamiento de cada pliegue. Un ajuste fino o entrenamiento propio del localizador requerirá un protocolo adicional de datos y validación antes de ejecutarse.

## 7. Riesgos y recursos

- **Pesos y solapamiento:** mayor riesgo de elegibilidad. Revisar el checkpoint real; solicitar documentación a sus autores requiere gestionar ese contacto por separado. Esta revisión no contactó a terceros ni obtuvo garantías de ausencia de cruce.
- **Referencia humana:** contar con especialista y segundo lector antes de recoger anotaciones. Comprobar concordancia en el piloto; una referencia ambigua bloquea conclusiones sobre milímetros.
- **Geometría:** un centro correcto puede seguir incluyendo la regla. La prueba de factibilidad antecede al gasto de integración y a la apertura de la muestra adicional.
- **Sobreajuste:** máximo un candidato aprendido elegible y una configuración geométrica final llevados a calificación; no búsqueda repetida en los mismos 60 participantes.
- **Dominio:** PA con flexión fija, fotometría, espaciado, contraste y versiones del modelo pueden diferir de sus datos de origen. Evaluar errores por condiciones técnicas y mantener abstención; no extrapolar a radiografías clínicas locales.
- **Exclusiones y comparabilidad:** informar rechazos con denominador original, incluyendo pérdidas de rodillas elegibles. No adaptar criterios de exclusión para fabricar aceptación o mejorar el modelo multimodal.
- **Reproducibilidad y acceso:** fijar dependencias y huellas; pesos privados, licencia registrada y transformación de coordenadas verificable. Docker en el repositorio externo no garantiza compatibilidad con Colab; el adaptador debe probar la ejecución disponible.
- **Anotación estimada:** 140 rodillas previstas, dos lectores. A 5–10 minutos por rodilla y lector serían unas 23–47 horas, más adjudicación y manual. Es una hipótesis de planificación a medir con el piloto, no una tarifa ni disponibilidad confirmada.
- **Cómputo:** inferencia CPU/GPU y conversión DICOM, sin entrenamiento local en M2. Medir latencia, VRAM, RAM, almacenamiento y coste con las veinte rodillas antes de proyectar a la cohorte; no prometer una GPU específica.
- **Contingencia:** si ningún peso es elegible, una nueva propuesta puede evaluar entrenamiento anatómico con datos auxiliares disjuntos, tamaño/anotaciones/licencia conocidos y separación por participante. No se autoriza esa ampliación en este documento.

## 8. Plan de implementación y verificación posterior a la aprobación

| Puerta | Trabajo y evidencia de salida | Condición para continuar |
| --- | --- | --- |
| G0 — cierre y aprobación | Contrastar libreta 12; registrar decisión exacta sobre MCR-2026-003 y enmienda académica | Cierre v0.4 verificable y aprobación registrada; contratos actualizados antes de aplicar |
| G1 — referencia y campo | Manual, dos lectores, anotaciones privadas del piloto y factibilidad de 140 mm | Referencia interpretable y campo capaz de cumplir el criterio; si falla, revisar M2 |
| G2 — elegibilidad | Auditoría de checkpoint, licencia, cadena de entrenamiento, sujetos y dependencias | Pesos elegibles y ejecutables; procedencia desconocida bloquea la promoción |
| G3 — adaptador y regresión | Módulo reutilizable en `src/knee`, configuración, pruebas sintéticas y ejecución controlada sobre piloto | 20/20 recortes válidos, coordenadas correctas y puertas verificadas |
| G4 — congelamiento técnico | Huellas de código/pesos/configuración; manifiesto adicional y reserva de desarrollo | Congelamiento anterior a consultar resultados del candidato en la muestra nueva |
| G5 — calificación | Anotaciones/revisión privada, informe agregado, métricas completas y consumo | Cumplimiento de todos los umbrales de 5.3; revisión del investigador |
| G6 — cierre del paso 4 | Acta de aceptación o rechazo, paquete congelado y limitaciones | Solo una aceptación verificable permite proponer el procesamiento del paso 5 |

Las pruebas necesarias para la implementación incluirán transformación ida/vuelta de coordenadas y espaciado anisótropo, lateralidad y reflejos internos, puntos ausentes/fuera de imagen, orden óseo inválido, recorte contra límites, artefactos, abstención, repetibilidad y separación de sujetos. No se ejecutaron estas pruebas del futuro localizador al preparar M2.

Artefactos privados previstos: auditoría de procedencia, manifiesto técnico, anotaciones independientes/adjudicadas, CSV de revisión, coordenadas/cajas, logs de calidad y pesos. Artefactos públicos: código, configuración sin datos individuales, informe agregado, referencias, pruebas sintéticas y acta. Cada fase registrará su revisión y huellas sin sobrescribir v0.1–v0.4.

Reversión: conservar evidencias, marcar candidato rechazado o no elegible y mantener abierto el paso 4. No volver a utilizar v0.4 como método aceptado. Un cambio de pesos después de congelamiento, geometría distinta, entrenamiento propio o acceso a otra cohorte exige revisión versionada previa.

## 9. Decisión del investigador

| Campo | Valor |
| --- | --- |
| Decisión | `aprobada` por el investigador |
| Fecha de aprobación | 30 de septiembre de 2026 |
| Evidencia de aprobación | Respuesta directa «si apruebo, hazlo bien» a la confirmación del alcance de MCR-2026-003 |
| Alcance aprobado | Adoptar la enmienda M2 e iniciar comprobaciones previas; tras cumplir G0–G2, integrar/evaluar un único localizador congelado en las muestras técnicas descritas |
| Condiciones aprobadas | Pesos elegibles, campo factible, referencia experta, muestra adicional reservada para desarrollo, puertas G0–G6 |
| Fuera del alcance aprobado | Entrenamiento/ajuste fino, procesamiento masivo, particiones definitivas, prueba reservada y modificación de los modelos predictivos |

Esta aprobación es propia de MCR-2026-003, no heredada de v0.4. No elimina las condiciones previas ni declara elegibles los pesos. El texto de enmienda de la sección 6 queda aprobado documentalmente; su incorporación a la versión académica de la tesis sigue pendiente. No se sobrescribieron los Word entregados.

## 10. Instantánea de preparación y seguimiento posterior

| Campo | Estado al preparar el documento |
| --- | --- |
| Documento y registro MCR | Preparados para revisión |
| Word de la tesis y contratos aprobados | Sin aplicar M2 |
| Checkpoint elegible | Ninguno confirmado |
| Anotaciones expertas y muestra adicional | No creadas |
| Implementación y fecha | No aplicada; sin fecha |
| Experimentos del candidato | No ejecutados |
| Procesamiento masivo / entrenamiento / particiones / apertura de prueba | No ejecutados por esta propuesta |
| Estado final de M2 | `esperando_aprobacion`; no `implementada_y_verificada` |

La revisión documental usa evidencia del piloto y las fuentes enlazadas; no consultó datos del bloque de prueba. Sus resultados técnicos futuros y la decisión del investigador se incorporarán con fecha, revisión de código/configuración, desviaciones y evidencia de verificación.

### Actualización posterior: aprobación del 30 de septiembre de 2026

La tabla anterior conserva el estado de preparación de la versión 1.0. El estado vigente es **`aprobada_no_aplicada`**, con contratos actualizados y comprobaciones previas documentadas en el acta 24. No existe integración ni evaluación del candidato. G0 está incompleta por falta de cierre operativo verificable e incorporación académica; G1 y G2 aún no están acreditadas. El paso 4 permanece abierto.
