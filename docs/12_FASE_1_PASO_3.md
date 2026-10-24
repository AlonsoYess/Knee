# Fase 1, paso 3: separación bilateral y verificación de lateralidad

## Propósito y límite

Este paso diseña y valida, con las diez adquisiciones del piloto, una regla reproducible para dividir cada radiografía bilateral basal en dos campos y asignar correctamente la lateralidad. No localiza todavía la interlínea tibiofemoral, no produce el recorte definitivo, no procesa las 1,916 adquisiciones, no entrena modelos y no accede a la prueba reservada.

La ejecución es ciega al desenlace. Utiliza la auditoría privada cerrada del paso 1, que contiene únicamente las referencias canónicas y huellas necesarias para volver a abrir los diez DICOM. No lee el CSV histórico del piloto porque ese archivo también contiene variables clínicas y de progresión que no son necesarias para esta decisión de imagen.

## Decisión técnica del piloto

El punto medio geométrico se conserva como referencia, pero no se acepta de forma automática. La regla candidata busca una línea dentro del 40 %–60 % del ancho y combina:

1. un perfil robusto de intensidad dentro del 15 %–85 % de la altura;
2. un perfil de gradiente horizontal para penalizar cortes que atraviesen estructuras;
3. suavizado proporcional a la resolución, sin coordenadas absolutas comunes;
4. una penalización leve por alejarse del centro;
5. controles de equilibrio entre los anchos resultantes, prominencia del valle y confianza.

La primera ejecución del piloto mostró una separación visualmente aceptable cuyo punto candidato quedó prácticamente en el límite superior de la banda de búsqueda. Aunque la anatomía no fue cortada, la versión inicial la clasificó como confianza alta. Antes de congelar la regla se incorporó una guarda del 1 % del ancho total en ambos límites de la banda: todo corte dentro de esa zona se marca como `REVIEW_REQUIRED_BOUNDARY` y confianza `LOW`, sin modificar automáticamente la línea ni excluir el estudio. Esta mejora fortalece el control de incertidumbre previsto en la metodología; no cambia la población, la entrada, el desenlace ni la tarea predictiva.

La imagen nativa y su profundidad se conservan. Para estimar la línea y generar la vista de control se crea una copia de trabajo normalizada entre los percentiles 1 y 99. `MONOCHROME1` se invierte solo en esa copia; `MONOCHROME2` no se invierte. El resultado de este paso son dos campos bilaterales potenciales, no los recortes articulares del paso 4.

La literatura confirma que dividir radiografías bilaterales OAI por el centro es una referencia utilizada en trabajos previos, pero también muestra que la localización anatómica supervisada exige modelos y anotaciones adicionales. Un estudio reciente informa que una localización determinista basada en intensidad puede ser robusta y no requiere anotaciones ni entrenamiento. Estas fuentes respaldan la comparación técnica, pero no sustituyen nuestra validación piloto:

- Thomas et al., *Applying Densely Connected Convolutional Neural Networks for Staging Osteoarthritis Severity from Plain Radiographs*, describen la división por el centro antes de localizar la articulación: <https://pubmed.ncbi.nlm.nih.gov/30306418/>.
- Chavoshi et al., *An Intensity-Based Cropping Approach for Fast, Interpretable, and Robust Localization of the Knee Joint in Radiographs*, comparan un procedimiento determinista sin entrenamiento con alternativas SVM y profundas: <https://pubmed.ncbi.nlm.nih.gov/42115493/>.
- Tiulpin et al., *KNEEL: Knee Anatomical Landmark Localization Using Hourglass Networks*, muestran que una alternativa neuronal de puntos anatómicos requiere anotaciones y entrenamiento específicos: <https://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Tiulpin_KNEEL_Knee_Anatomical_Landmark_Localization_Using_Hourglass_Networks_ICCVW_2019_paper.pdf>.

Por ello, una red de localización no se incorpora silenciosamente en este paso. Si la regla determinista no supera el piloto, se documentará la evidencia y se evaluará una alternativa para el paso 4 mediante el control de cambios metodológicos antes de entrenarla o usarla.

## Lateralidad

La correspondencia candidata es:

| Posición en la matriz mostrada | Rodilla propuesta |
| --- | --- |
| Mitad izquierda de la imagen | Derecha del participante (`SIDE=1`) |
| Mitad derecha de la imagen | Izquierda del participante (`SIDE=2`) |

No se congela esta correspondencia hasta revisar los marcadores y la anatomía de las diez vistas. El campo DICOM `Laterality` se registra como evidencia secundaria, pero no decide la asignación: en la auditoría cerrada estaba vacío en nueve adquisiciones y contenía `L` en una radiografía bilateral.

## Flujo reproducible

El comando `knee-bilateral-pilot`:

1. exige la auditoría cerrada del piloto y verifica las huellas del paquete, DICOM y píxeles antes de analizar cada caso;
2. crea alias `case_001` a `case_010` para que la revisión no muestre identificadores;
3. estima una línea de separación con parámetros versionados;
4. genera vistas PNG con la línea candidata y la correspondencia propuesta;
5. clasifica la confianza como `HIGH` o `LOW`; una línea cercana al límite de búsqueda se deriva obligatoriamente a revisión, sin convertir una confianza baja en exclusión automática;
6. genera una planilla de revisión ciega y conserva por separado el mapa privado entre alias y adquisición;
7. registra de forma explícita que no leyó el desenlace, no procesó masivamente, no entrenó y no abrió la prueba.

La libreta [`05_validacion_separacion_bilateral.ipynb`](../notebooks/05_validacion_separacion_bilateral.ipynb) solo monta Drive, obtiene una revisión identificable del repositorio, ejecuta las pruebas, llama al comando y muestra las diez vistas seudonimizadas.

## Evidencias privadas esperadas en Drive

La ejecución escribe en `outputs/auditorias/fase_1/paso_3_separacion_bilateral`:

- `resumen_separacion_publico.json`: recuentos y distribución agregada de confianza;
- `resultados_separacion_privados.csv`: métricas y trazabilidad individual;
- `mapa_casos_privado.csv`: relación separada entre alias y huella;
- `previews_ciegas/`: diez vistas sin identificadores con la línea candidata;
- `revision_visual_ciega.csv`: registro que debe completar el revisor;
- `parametros_candidatos.json`: parámetros aún no congelados;
- `registro_preparacion.json`: commit y bloqueos de seguridad de la ejecución.

Ninguno de los archivos individuales ni las vistas se publica en GitHub.

## Protocolo de revisión visual

Para cada alias se documentará:

1. si la línea roja separa las dos rodillas sin cortar anatomía relevante;
2. si marcadores visibles o referencias anatómicas respaldan la correspondencia derecha/izquierda propuesta;
3. qué evidencia se observó, sin consultar la etiqueta de progresión;
4. si existe un defecto técnico que obligue a excluir el caso y su motivo;
5. confirmación expresa de que la revisión fue ciega al desenlace.

Las vistas de confianza baja deben revisarse con especial atención. La intervención humana valida la regla general; no autoriza dibujar recortes manuales privilegiados que luego no puedan reproducirse en la aplicación.

## Cierre reproducible

Después de completar la revisión, el comando `knee-bilateral-close` compara cada valor automático del registro con los resultados privados originales, exige decisiones para los diez alias, confirma que la revisión fue ciega y detiene el cierre ante un rechazo, exclusión, edición de los valores automáticos o violación de los bloqueos. Si todas las condiciones se cumplen, genera `parametros_congelados.json` y `cierre_paso_3_publico.json`. La libreta [`06_cierre_revision_separacion_bilateral.ipynb`](../notebooks/06_cierre_revision_separacion_bilateral.ipynb) orquesta esta operación desde una revisión identificable del repositorio.

## Criterio de cierre

El paso 3 podrá cerrarse solo cuando:

- las diez adquisiciones se procesen sin fallos de integridad ni lectura;
- cada línea de separación tenga una revisión documentada;
- la correspondencia derecha/izquierda esté respaldada en el piloto y no dependa del campo DICOM;
- los casos de baja confianza y cualquier conflicto tengan una resolución explícita;
- los parámetros aceptados queden versionados y congelados antes de procesar la cohorte;
- se confirme que no se usó el desenlace, no se ejecutó el procesamiento masivo, no se entrenó y no se abrió la prueba reservada.

## Estado

**Cerrado técnicamente.** `bilateral_split_v0.2_pilot` procesó las diez adquisiciones sin fallos: nueve casos quedaron en confianza alta y un caso fue derivado correctamente a revisión por la guarda de límites. Las diez separaciones y lateralidades fueron aceptadas, sin exclusiones y sin consultar el desenlace. La libreta de cierre verificó el registro y congeló los parámetros sobre el commit `2489a7ec6253fb4aaa12818416f249780d26ea24`. El resultado agregado y la autorización limitada para el paso 4 constan en [`13_CIERRE_FASE_1_PASO_3.md`](13_CIERRE_FASE_1_PASO_3.md). No se ejecutó procesamiento masivo, entrenamiento ni apertura de la prueba reservada.
