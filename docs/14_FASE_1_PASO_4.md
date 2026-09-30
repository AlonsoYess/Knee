# Fase 1, paso 4: localización tibiofemoral y recorte por rodilla

## Propósito y límite

Este paso diseña y valida, con los diez estudios ya utilizados en el piloto, una regla reproducible para localizar la articulación tibiofemoral y generar un recorte por cada rodilla. Produce veinte candidatos seudonimizados y conserva las coordenadas necesarias para repetir exactamente cada recorte.

No procesa todavía las 1,916 adquisiciones, no crea la cohorte radiográfica definitiva, no genera particiones, no entrena modelos y no abre la prueba reservada. La revisión continúa ciega al desenlace de progresión.

## Decisión técnica candidata

La primera alternativa será determinista, interpretable y sin entrenamiento adicional. Combina tres señales calculadas en cada campo unilateral ya separado:

1. oscuridad relativa de la interlínea dentro de una banda vertical anatómicamente plausible;
2. contraste entre la interlínea candidata y las regiones óseas superior e inferior;
3. gradiente vertical robusto alrededor de los bordes articulares.

La posición horizontal se estima mediante un centro ponderado de la anatomía de alta intensidad, excluyendo los bordes laterales. La posición vertical se selecciona por una función multiseñal suavizada y una penalización leve respecto de la región esperada; no se acepta automáticamente el centro geométrico.

El campo de visión candidato es cuadrado y mide 140 × 140 mm. Sus dimensiones en píxeles se calculan a partir de `ImagerPixelSpacing` y, si no existe, `PixelSpacing`. La ausencia de un espaciado válido es un fallo técnico explícito: no se sustituye silenciosamente por una escala arbitraria. El recorte nativo conserva la profundidad original; la normalización percentilar se usa únicamente para localizar y revisar. El redimensionamiento a 224 o 384 píxeles corresponderá al pipeline del modelo y no altera este campo de visión anatómico.

La selección de 140 mm es coherente con trabajos de OAI que han utilizado regiones físicas alrededor del centro articular. Una alternativa neuronal como KNEEL puede localizar puntos anatómicos con mayor flexibilidad, pero requiere anotaciones, pesos y una evaluación separada. No se incorporará silenciosamente: si el método determinista no supera el piloto, se documentará el fallo y se evaluará una propuesta de cambio antes de entrenar o integrar un localizador supervisado.

Referencias técnicas principales:

- Chavoshi et al., *An Intensity-Based Cropping Approach for Fast, Interpretable, and Robust Localization of the Knee Joint in Radiographs*: <https://pubmed.ncbi.nlm.nih.gov/42115493/>.
- Tiulpin et al., *Automatic Knee Osteoarthritis Diagnosis from Plain Radiographs: A Deep Learning-Based Approach*: <https://pmc.ncbi.nlm.nih.gov/articles/PMC5789045/>.
- Tiulpin, Melekhov y Saarakkala, *KNEEL: Knee Anatomical Landmark Localization Using Hourglass Networks*: <https://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Tiulpin_KNEEL_Knee_Anatomical_Landmark_Localization_Using_Hourglass_Networks_ICCVW_2019_paper.pdf>.

## Dependencias verificables

El comando `knee-joint-pilot` exige antes de operar:

- la auditoría cerrada de las diez adquisiciones del paso 1;
- el resultado privado de separación bilateral del paso 3;
- el cierre público del paso 3;
- los parámetros `bilateral_split_v0.2_pilot` congelados;
- coincidencia de huellas entre el resultado bilateral y su cierre.

Esto impide recalcular silenciosamente la separación o modificar la lateralidad durante el recorte.

## Salidas privadas del piloto

La ejecución escribe en `outputs/auditorias/fase_1/paso_4_localizacion_tibiofemoral`:

- `resumen_localizacion_publico.json`: recuentos agregados y estado del piloto;
- `resultados_localizacion_privados.csv`: coordenadas, espaciado, confianza, trazabilidad y huellas individuales;
- `recortes_nativos/`: matrices exactas de los veinte candidatos en su profundidad original;
- `previews_ciegas/`: vistas por rodilla con el campo unilateral, la caja candidata, la interlínea y el recorte;
- `revision_visual_ciega.csv`: decisiones que deberá completar el revisor;
- `parametros_candidatos.json`: parámetros aún no congelados;
- `registro_preparacion.json`: commit y bloqueos de la ejecución.

Los archivos individuales permanecen en Drive privado y no se publican en GitHub.

## Protocolo de revisión visual

Para cada una de las veinte rodillas se documentará:

1. si la línea candidata está centrada en la articulación tibiofemoral;
2. si el recorte conserva completa la anatomía tibiofemoral necesaria;
3. si excluye texto, bordes, rectángulos de anonimización y la regla central;
4. si el recorte es aceptable sin correcciones manuales privilegiadas;
5. si existe un defecto técnico y su motivo;
6. que la revisión se realizó sin consultar el desenlace.

Las candidatas de confianza baja se revisarán con especial atención. La revisión valida o rechaza la regla; no permite desplazar manualmente una caja individual para mejorar un caso.

## Criterio de cierre

El paso 4 podrá cerrarse únicamente cuando:

- los diez estudios y veinte rodillas se procesen sin fallos de integridad;
- exista una decisión completa para cada recorte;
- la articulación esté centrada y la anatomía relevante permanezca completa;
- los artefactos periféricos y la regla central queden fuera del recorte;
- todos los casos de confianza baja tengan resolución explícita;
- los parámetros aceptados queden congelados antes del paso 5;
- se confirme que no se consultó el desenlace, no se procesó masivamente, no se crearon particiones, no se entrenó y no se abrió la prueba reservada.

## Estado

**Preparación reproducible implementada; ejecución piloto pendiente.** El siguiente acto autorizado es ejecutar [`07_validacion_localizacion_tibiofemoral.ipynb`](../notebooks/07_validacion_localizacion_tibiofemoral.ipynb) en Colab y revisar las veinte vistas. El procesamiento masivo continúa bloqueado hasta cerrar este paso.
