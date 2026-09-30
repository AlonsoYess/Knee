# Fase 1, paso 4: localización tibiofemoral y recorte por rodilla

## Propósito y límite

Este paso diseña y valida, con los diez estudios ya utilizados en el piloto, una regla reproducible para localizar la articulación tibiofemoral y generar un recorte por cada rodilla. Produce veinte candidatos seudonimizados y conserva las coordenadas necesarias para repetir exactamente cada recorte.

No procesa todavía las 1,916 adquisiciones, no crea la cohorte radiográfica definitiva, no genera particiones, no entrena modelos y no abre la prueba reservada. La revisión continúa ciega al desenlace de progresión.

## Decisión técnica candidata

La primera alternativa es determinista, interpretable y sin entrenamiento adicional. Combina tres señales calculadas en cada campo unilateral ya separado:

1. oscuridad relativa de la interlínea dentro de una banda vertical anatómicamente plausible;
2. contraste entre la interlínea candidata y las regiones óseas superior e inferior;
3. gradiente vertical robusto alrededor de los bordes articulares.

La versión inicial `tibiofemoral_crop_v0.1_pilot` resumía la intensidad en una banda central. Su revisión ciega demostró que esa banda podía confundir la espina tibial o estructuras inferiores con la interlínea y que una caja físicamente correcta podía conservar puntos de la regla central. Por ello `v0.1` quedó rechazada y sus parámetros no se congelaron.

La corrección `tibiofemoral_crop_v0.2_pilot` conserva el enfoque determinista, pero estima perfiles independientes en los compartimentos medial y lateral, excluye la zona intercondílea central de la decisión primaria y exige concordancia espacial entre ambos perfiles. La confianza incorpora la distancia entre los dos máximos candidatos; una discordancia obliga a revisión. La posición horizontal continúa derivándose de la anatomía y el ajuste de la caja añade una separación física mínima respecto del borde interno donde se ubica la regla central. No se desplaza manualmente una rodilla individual.

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

La ejecución inicial se conserva en `outputs/auditorias/fase_1/paso_4_localizacion_tibiofemoral`. La repetición `v0.2` escribe en el subdirectorio `v0_2_piloto`, sin sobrescribir la evidencia rechazada de `v0.1`:

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

**Paso abierto; primera versión rechazada y repetición corregida preparada.** `tibiofemoral_crop_v0.1_pilot` procesó las veinte rodillas, pero la revisión ciega aceptó integralmente 9/20 y rechazó 11/20. Nueve líneas no quedaron centradas y siete recortes conservaron la regla o sus puntos; ambos defectos se superponen en algunos casos. No se atribuyeron exclusiones técnicas a los DICOM y no se consultó el desenlace.

La revisión rechazada quedó documentada sin congelar parámetros. El siguiente acto autorizado es ejecutar [`08_repeticion_localizacion_tibiofemoral_v02.ipynb`](../notebooks/08_repeticion_localizacion_tibiofemoral_v02.ipynb), que valida primero el acta de `v0.1` y luego genera las veinte vistas con `v0.2`. Esta corrección refina el mismo procedimiento durante el piloto y no cambia población, entrada, desenlace, horizonte ni modelos. El procesamiento masivo continúa bloqueado hasta que una versión complete y supere la revisión.
