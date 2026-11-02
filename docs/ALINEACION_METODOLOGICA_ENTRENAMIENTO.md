# Concordancia entre metodología y ejecución

La entrada vigente es `configs/modeling_release_v1_2.json`, bajo MCR-2026-015. Esta versión restaura componentes ya comprometidos en el Capítulo III y en el contrato de alcance. Su aprobación permite implementar y verificar el software; **no permite entrenar ni abrir la prueba reservada**. La búsqueda de arquitecturas permanece cerrada.

## Escenarios y entradas

| Escenario | Entradas clínicas basales | Papel |
|---|---|---|
| `clinical_main` | Edad, sexo, IMC | Comparación principal y rama del multimodal |
| `clinical_plus_baseline_kl` | Edad, sexo, IMC, KL inicial | Referencia complementaria |
| `clinical_plus_baseline_surgery_and_womac` | Edad, sexo, IMC, cirugía previa, WOMAC total | Referencia complementaria |

Los tres algoritmos clínicos registrados —regresión logística, XGBoost y MLP— utilizan los mismos cinco pliegues en cada escenario. No se cambia la cohorte. La mediana del IMC y WOMAC, la moda de cirugía y las estadísticas de escalado se calculan exclusivamente en el entrenamiento de cada pliegue. Un empate de la moda de cirugía se resuelve en cero. KL inicial observado se limita a 2 o 3 y se representa como variable ordinal escalada; no se imputa. Cirugía es binaria y no se escala. WOMAC se comprueba dentro de 0–96. No se añaden indicadores de ausencia ni predictores de seguimiento.

Las columnas canónicas auditadas son `KL_BASE`, `CIRUGIA_PREVIA` y `WOMAC_TOTAL`. La unión siempre incluye participante y lado. Las variables complementarias viajan en una entrada adicional privada vinculada por hash al paquete de desarrollo; el paquete principal verificado en la libreta 32 permanece intacto.

## Preparación sin entrenamiento

`scripts/prepare_methodology_inputs.py` verifica el paquete de desarrollo y su referencia de la libreta 32. Recibe una proyección JSON basal **solo de los participantes y rodillas de desarrollo**, obtenida de la fuente V00 auditada, con hash externo. Cada fila contiene exactamente `src_subject_id`, `side`, `AGEYEARS`, `SEX`, `BMI`, `KL_BASE`, `CIRUGIA_PREVIA`, `WOMAC_TOTAL`, con números tipados y `null` para ausencias admitidas. `side` es `LEFT` o `RIGHT`; en la fuente original `1=RIGHT`, `2=LEFT`. Los tres predictores principales deben coincidir con el paquete existente. No se acepta el CSV completo de cohorte, claves desconocidas, etiquetas futuras ni un paquete de prueba.

El comando requiere `--development-package-dir`, `--development-package-sha256`, `--input-reference`, `--baseline-development-projection`, `--baseline-projection-sha256` y `--output-dir`. Prepara 15 estados clínicos, 30 matrices y 45 solicitudes de trabajo, sin autoridades de ejecución ni ajuste predictivo. Las salidas deben ir a una carpeta privada nueva, separada de las fuentes. La preparación de datos reales sigue pendiente hasta disponer de la proyección auditada y comprobarla en desarrollo; las pruebas de software no la sustituyen.

El ejecutor `scripts/run_methodology_training.py` admite las nuevas solicitudes y `--baseline-sidecar` para los dos escenarios complementarios. Exige autorización separada vinculada a la solicitud, la versión, la fuente adicional y el pliegue. El MLP conserva capas de 16 y 8 unidades y adapta su entrada a 3, 4 o 5 variables. No hay ajuste de hiperparámetros adicional en esta corrección.

## Fusión tardía

`scripts/combine_late_fusion_development.py` utiliza predicciones de desarrollo de trabajos terminados y verificados. Combina **0,5 × probabilidad clínica + 0,5 × probabilidad radiográfica**; no promedia logits, no aprende pesos ni ajusta un segundo modelo. La rama clínica corresponde al escenario principal. Su elección se verifica entre los tres algoritmos, y la rama visual exige la selección documentada en desarrollo.

Se exige coincidencia de participante, lado, pliegue, etiqueta y procedencia de entrenamiento. Se revisan también los candidatos utilizados para seleccionar las ramas. El orden de filas puede diferir, pero no las identidades. Las métricas por pliegue y las agrupadas quedan identificadas como resultados de desarrollo utilizados para selección; no constituyen evaluación final independiente. La calibración posterior, el umbral y la prueba final no se ejecutan aquí.

## Aumento de datos

La política `training_intensity_gamma_v1.0` aplica gamma entre 0,9 y 1,1 con probabilidad 0,5, antes de la normalización específica del codificador, únicamente sobre copias del subconjunto de entrenamiento. No desplaza píxeles, recorta, rota ni invierte la imagen. Conserva lateralidad y ceros del relleno. Validación e inferencia no reciben este aumento.

Los parámetros dependen de semilla, pliegue, época e identidad de rodilla, nunca de la etiqueta ni del orden del cargador. La repetición del mismo grupo ante un desbordamiento AMP utiliza exactamente las mismas imágenes. Los límites moderados son una decisión inicial de implementación; las comprobaciones de software acreditan su aplicación correcta, **no una mejora predictiva demostrada**. Su comportamiento en imágenes reales se debe revisar en desarrollo antes de autorizar la campaña correspondiente.

## Comparabilidad e historial

Las nuevas campañas radiográficas y de fusión intermedia de la versión 1.2 aplican la misma política de aumento. Los trabajos de la versión 1.1 mantienen su receta original sin aumentos. El resultado inicial de DenseNet con codificador congelado se conserva, pero no se mezcla silenciosamente con nuevos candidatos entrenados bajo otra receta para promover ganadores. Una comparación nueva requiere trabajos de la misma versión. El código rechaza esa mezcla; no declara inválidos los resultados históricos ni los reescribe.

No se han cambiado originales, recortes, resolución, particiones, desenlace, arquitecturas ni entradas del prototipo principal. No se ha entrenado con esta versión ni se ha abierto la prueba reservada. Las verificaciones usan paquetes, predicciones y modelos pequeños sintéticos; las llamadas de ajuste y optimizador del ejecutor se sustituyen por simulaciones en las pruebas de integración.

## Alcance de la comprobación

Las pruebas cubren contratos del plan, autorización, las nueve combinaciones clínicas, conservación del escenario principal, ausencia de ajuste del preprocesamiento en validación, variación real de píxeles solo en entrenamiento, normalización de los seis codificadores accesibles, procedencia de fusión, métricas y rechazo de alteraciones. No certifican rendimiento de la cohorte ni funcionamiento en Colab con la nueva entrada privada. Las libretas históricas no se actualizan silenciosamente: la siguiente comprobación operativa deberá usar la nueva versión fijada del repositorio.

El ejecutor 1.2 está separado en `src/knee/aligned_training/`. Los archivos del ejecutor 1.1 permanecen byte a byte sin cambios para conservar las verificaciones criptográficas de las libretas anteriores. No se debilitan sus comprobaciones para admitir una receta distinta.
