# Referencias metodológicas actuales

El documento académico aprobado por el investigador define los compromisos de la tesis. Este índice identifica las configuraciones que los ejecutan y conserva la historia sin reactivar decisiones sustituidas.

- `configs/modeling_release_v1_2.json` vincula el registro de modelos, los pesos y el plan vigente.
- `configs/training_plan_v1_2.json` define los tres escenarios clínicos, las fusiones, el aumento de intensidad exclusivo del entrenamiento y la selección mediante la media de AP de cinco pliegues.
- `configs/model_input_tensor_v1.json` y `configs/participant_split_v1.json` describen la preparación de entradas y la separación por participante ya utilizadas. Su publicación no modifica las imágenes ni vuelve a crear las particiones.
- `configs/governance/statistics_alignment_v1.json` registra la alineación del apartado 3.3.2: HE2–HE4 emplearán el límite inferior del IC del 95 % de ROC-AUC respecto de 0,5; HE5 exigirá un límite inferior de la diferencia pareada de AP mayor que cero frente a cada unimodal. El remuestreo se realizará por participante. La regla anterior que exigía Holm y un valor p adicional queda sustituida para estas decisiones.

El contrato público conserva bloques históricos del registro de candidatos, del recorte y de resoluciones exploratorias. Sus campos `current_operational_references` e `historical_sections_policy` identifican las fuentes vigentes; esos bloques antiguos no autorizan modelos, algoritmos ni cambios de resolución.

Las libretas históricas conservan sus revisiones para reproducibilidad. Las campañas nuevas deben usar el plan v1.2 y no presentar resultados de una receta anterior como si procedieran de la actual. Las libretas futuras requieren la confirmación expresa de su campaña antes de ajustar modelos.

Esta alineación prepara código y controles. No ejecuta entrenamiento, evaluación reservada, calibración, selección del umbral o reentrenamiento final; tampoco declara implementadas las etapas que siguen pendientes ni modifica el Word.
