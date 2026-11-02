# Campañas de desarrollo con la metodología v1.2

Las libretas 39 y 40 ejecutan el plan `configs/training_plan_v1_2.json` mediante
`knee.methodology_campaign_colab` y `knee.aligned_training.runner`. Publicarlas y
verificar sus pruebas sintéticas no equivale a ejecutar entrenamiento.

La libreta 38 debe haber terminado en Colab y guardado sus resultados completos.
El investigador revisa sus tres vistas de aumento antes de confirmar una campaña.
Un ensayo local o un conjunto parcial de archivos no satisfacen ese requisito.
Las carpetas privadas se introducen durante la sesión y no quedan fijadas en el
repositorio.

| Libreta | Recurso | Alcance de una confirmación futura |
| --- | --- | --- |
| 39 | CPU | Regresión logística, XGBoost y MLP; tres escenarios clínicos; cinco pliegues; semilla 2026: 45 trabajos. |
| 40 | GPU | DenseNet121, codificador congelado, cinco pliegues y semilla 2026, con el aumento de intensidad v1.2. |

Los escenarios clínicos son edad/sexo/IMC; edad/sexo/IMC/KL inicial; y
edad/sexo/IMC/cirugía previa/WOMAC. Se ajustan imputación y escalado dentro del
entrenamiento de cada pliegue. Las variables complementarias no se añaden al
multimodal principal. El aumento solo modifica copias de entrenamiento antes de
la normalización del codificador; las imágenes de validación son deterministas.

Cada libreta contiene una confirmación específica antes de iniciar los trabajos.
Las configuraciones publicadas no conceden autoridad global de entrenamiento.
Las campañas no abren la prueba reservada ni realizan calibración, elección de
umbral, reentrenamiento final o entrenamiento de otros candidatos.

Cada trabajo completo se guarda y verifica en Drive antes del siguiente. La
identidad de recuperación incluye receta, revisión, entradas y código. Los
trabajos interrumpidos se reinician; no se continúa un optimizador parcial.
Solo debe haber una sesión escribiendo una misma campaña.

Las libretas 36 y 37 conservan su receta y sus resultados históricos. La 36 no
incluía el aumento v1.2 y la 37 utilizaba solo el escenario clínico principal.
Sus trabajos no se incorporan silenciosamente a las campañas 39 y 40.

La selección utiliza la media no ponderada de AP de cinco pliegues. La AP conjunta
de las predicciones fuera de pliegue es descriptiva. Las métricas de desarrollo
no representan una evaluación final independiente.

El registro `configs/governance/statistics_alignment_v1.json` documenta el criterio
del Word para la futura evaluación confirmatoria. No implementa ni ejecuta esa
evaluación. La calibración, los intervalos por participante, el aplicativo y su
valoración continúan siendo etapas posteriores.
