# Fase 1, paso 4: localización tibiofemoral y recorte por rodilla

## Propósito y límite

Este paso diseña y valida, con los diez estudios ya utilizados en el piloto, una regla reproducible para localizar la articulación tibiofemoral y generar un recorte por cada rodilla. Produce veinte candidatos seudonimizados y conserva las coordenadas necesarias para repetir exactamente cada recorte.

No procesa todavía las 1,916 adquisiciones, no crea la cohorte radiográfica definitiva, no genera particiones, no entrena modelos y no abre la prueba reservada. La revisión continúa ciega al desenlace de progresión.

## Enmienda M2 aprobada; código preparado, evaluación pendiente

El 1 de octubre de 2026 se aprobó [MCR-2026-004](25_MCR_2026_004_RECORTE_ROI_Y_REVISION_TECNICA.md), que sustituye a MCR-2026-003 sin borrar su aprobación histórica. Autoriza el candidato determinista fijado, ROI proporcional y revisión técnica sin lector anatómico adicional. El [acta 26](26_APROBACION_MCR_2026_004_Y_CIERRE_PENDIENTE.md) conserva la instantánea inicial pendiente y documenta su resolución posterior: el investigador ejecutó la libreta 12 en Colab y los dos JSON de cierre de v0.4 fueron contrastados como rechazo 8/20.

Las secciones siguientes conservan el protocolo y evidencias históricas de v0.1–v0.4. Para el candidato nuevo rigen la guía y todas las puertas de 4.3–4.4 de MCR-2026-004: regresión histórica y confirmación posterior en veinte participantes nuevos de desarrollo. No se aplican retrospectivamente a v0.4. El código adaptador, sus pruebas sintéticas y la libreta 13 están preparados localmente, pero el candidato aún no se ha aplicado a las radiografías ni publicado. Los Word académicos no se han modificado.

## Decisión técnica histórica: versiones deterministas

La primera alternativa es determinista, interpretable y sin entrenamiento adicional. Combina tres señales calculadas en cada campo unilateral ya separado:

1. oscuridad relativa de la interlínea dentro de una banda vertical anatómicamente plausible;
2. contraste entre la interlínea candidata y las regiones óseas superior e inferior;
3. gradiente vertical robusto alrededor de los bordes articulares.

La versión inicial `tibiofemoral_crop_v0.1_pilot` resumía la intensidad en una banda central. Su revisión ciega demostró que esa banda podía confundir la espina tibial o estructuras inferiores con la interlínea y que una caja físicamente correcta podía conservar puntos de la regla central. Por ello `v0.1` quedó rechazada y sus parámetros no se congelaron.

La corrección `tibiofemoral_crop_v0.2_pilot` conservó el enfoque determinista, pero estimó perfiles independientes en los compartimentos medial y lateral, excluyó la zona intercondílea central de la decisión primaria y exigió concordancia espacial entre ambos perfiles. La confianza incorporó la distancia entre los dos máximos candidatos y el ajuste de la caja añadió una separación física mínima respecto del borde interno. Su revisión posterior mostró que la concordancia podía reforzar un máximo equivocado y que el margen fijo no excluía todos los puntos de la regla, por lo que `v0.2` también quedó rechazada.

La corrección `v0.3`, definida inicialmente en [`17_PROPUESTA_FASE_1_PASO_4_V03.md`](17_PROPUESTA_FASE_1_PASO_4_V03.md), se implementó según [`18_IMPLEMENTACION_FASE_1_PASO_4_V03.md`](18_IMPLEMENTACION_FASE_1_PASO_4_V03.md). Usa pares de bordes dirigidos, consenso de centro y ancho, una banda periférica adaptativa y puertas obligatorias de confianza. Se ejecutó sobre el mismo piloto y quedó rechazada tras la revisión ciega documentada en [`19_REVISION_FASE_1_PASO_4_V03.md`](19_REVISION_FASE_1_PASO_4_V03.md).

El análisis posterior mostró que la banda adaptativa acopló indebidamente la exclusión de artefactos con el desplazamiento horizontal y que la puerta de bordes verticales no discriminó el centrado real. La propuesta [`20_PROPUESTA_FASE_1_PASO_4_V04.md`](20_PROPUESTA_FASE_1_PASO_4_V04.md) separó conservación anatómica, validación periférica y consenso vertical entre dos familias de señal; [`21_IMPLEMENTACION_FASE_1_PASO_4_V04.md`](21_IMPLEMENTACION_FASE_1_PASO_4_V04.md) registra su implementación. `v0.4` se ejecutó sobre el mismo piloto y quedó rechazada tras aceptar 8/20 recortes: la anatomía permaneció completa, pero nueve líneas no quedaron centradas y seis campos conservaron puntos de la regla. La revisión se documenta en [`22_REVISION_FASE_1_PASO_4_V04.md`](22_REVISION_FASE_1_PASO_4_V04.md) y su cierre reproducible queda preparado en la libreta 12.

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

La ejecución inicial se conserva en `outputs/auditorias/fase_1/paso_4_localizacion_tibiofemoral`. Las repeticiones `v0.2`, `v0.3` y `v0.4` escriben respectivamente en `v0_2_piloto`, `v0_3_piloto` y `v0_4_piloto`, sin sobrescribir las evidencias rechazadas:

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

**Paso abierto; `v0.1`, `v0.2`, `v0.3` y `v0.4` rechazadas.** `tibiofemoral_crop_v0.1_pilot` aceptó integralmente 9/20 recortes. La repetición `tibiofemoral_crop_v0.2_pilot` aceptó 8/20 y rechazó 12/20. `tibiofemoral_crop_v0.3_pilot` procesó las mismas veinte rodillas sin fallos, pero aceptó 0/20: ocho líneas no quedaron centradas, once recortes presentaron anatomía incompleta y catorce conservaron borde, fondo negro o puntos de la regla. `tibiofemoral_crop_v0.4_pilot` recuperó la anatomía completa, pero solo aceptó 8/20 recortes: nueve líneas no quedaron centradas y seis campos conservaron puntos de la regla. No hubo exclusiones técnicas ni consulta del desenlace.

Las revisiones están documentadas en [`15_REVISION_FASE_1_PASO_4_V01.md`](15_REVISION_FASE_1_PASO_4_V01.md), [`16_REVISION_FASE_1_PASO_4_V02.md`](16_REVISION_FASE_1_PASO_4_V02.md), [`19_REVISION_FASE_1_PASO_4_V03.md`](19_REVISION_FASE_1_PASO_4_V03.md) y [`22_REVISION_FASE_1_PASO_4_V04.md`](22_REVISION_FASE_1_PASO_4_V04.md), sin congelar parámetros. La libreta [`10_cierre_revision_localizacion_tibiofemoral_v03.ipynb`](../notebooks/10_cierre_revision_localizacion_tibiofemoral_v03.ipynb) ya consolidó y verificó el cierre rechazado de la tercera revisión. La libreta [`12_cierre_revision_localizacion_tibiofemoral_v04.ipynb`](../notebooks/12_cierre_revision_localizacion_tibiofemoral_v04.ipynb) queda preparada para validar el CSV privado y consolidar el rechazo de la cuarta revisión. El procesamiento masivo continúa bloqueado hasta que una versión complete y supere la revisión; cualquier nueva familia técnica requiere control de cambios previo.
