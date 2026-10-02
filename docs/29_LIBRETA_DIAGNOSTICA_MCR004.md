# Libreta 15: diagnóstico de factibilidad MCR004

## Alcance autorizado

El investigador autorizó preparar la libreta diagnóstica y colocar su copia en el Drive privado para ejecutarla personalmente. Se desarrolla la sección 6.1 de [28_ANALISIS_FALLOS_MCR004_Y_PROPUESTA_ACOTADA.md](28_ANALISIS_FALLOS_MCR004_Y_PROPUESTA_ACOTADA.md): comprobar el comportamiento del candidato existente sobre las mismas diez adquisiciones y veinte rodillas históricas, conservando el cierre rechazado.

La fuente editable es `notebooks/15_diagnostico_factibilidad_mcr004.ipynb`; la copia de ejecución se denomina `15_diagnostico_factibilidad_mcr004_v1.0.ipynb`. El procesamiento está en `src/knee/roi_mcr004_diagnostic.py`; la libreta conecta, instala, comprueba, ejecuta y muestra los artefactos. No incorpora una regla geométrica nueva ni adjudica decisiones de revisión.

## Ejecución prevista en Colab

1. Seleccionar CPU, entorno Colab 2026.07 y Python 3.12.
2. Conservar el secreto privado `KNEE_DATA_ROOT` ya utilizado.
3. Ejecutar las celdas en orden o usar «Ejecutar todo», atendiendo la autorización de acceso a Drive.
4. Compartir únicamente el resumen público mostrado al terminar. Las vistas y los informes individuales permanecen privados.

No hay casillas de confirmación, formularios por rodilla ni solicitudes de diagnóstico clínico. No hace falta volver a ejecutar las libretas 13 o 14. La galería muestra las imágenes con su tamaño natural dentro de paneles desplazables y conserva el contexto necesario para inspeccionarlas.

Cada ejecución prepara una copia temporal nueva de la rama autorizada y registra su commit. El entorno aislado usa `virtualenv==21.7.11`, evita el mecanismo `venv/ensurepip` que falló en Colab y conserva las versiones fijadas en `requirements-mcr004.txt`. Antes del diagnóstico se ejecutan las pruebas sintéticas MCR004. Esas pruebas no acreditan aceptabilidad anatómica.

## Salidas y comportamiento ante errores

La carpeta nueva, relativa a la raíz privada autorizada, es `outputs/auditorias/fase_1/paso_4_localizacion_tibiofemoral/mcr004_diagnostico_v1`. Debe estar separada de la regresión histórica y no existir al iniciar una ejecución nueva.

Los productos previstos son `diagnostico_privado.json`, `galeria_diagnostica.html`, `resumen_diagnostico_publico.json` y `registro_diagnostico.json`. La libreta exige los cuatro archivos y comprueba que el resumen guardado coincida con el que devuelve el programa antes de mostrar la galería.

Si existe una salida previa, se detiene sin sobrescribirla ni borrarla. Si el proceso falla, muestra el código de salida, la salida estándar y el error interno completos. Una salida parcial se conserva para investigar su estado; la libreta no la presenta como diagnóstico completado. El investigador debe compartir el mensaje en lugar de cambiar nombres, eliminar carpetas o repetir cierres.

El análisis posterior de la evidencia determinará si hay una corrección técnica defendible. Este diagnóstico no produce un nuevo resultado de aceptación, no modifica el CSV cerrado, no selecciona participantes nuevos y no habilita procesamiento masivo, particiones, entrenamiento o prueba. Una corrección que cambie el método seguirá el control de cambios descrito en el documento 28.

## Verificación de la libreta

Ocho pruebas locales sobre datos sintéticos comprueban compilación de todas las celdas, ausencia de salidas e identificadores privados incrustados, versiones del entorno, procedencia del código, detención antes de ejecutar cuando faltan pruebas o existe salida, transmisión exacta de argumentos, conservación de salidas parciales o inconsistentes, exposición del error interno y visualización sin redimensionamiento añadido por la libreta. No equivalen a una ejecución real en Colab.

La publicación y la ejecución se acreditan mediante sus registros correspondientes. Este documento no afirma que el investigador haya ejecutado ya la libreta 15 ni que el diagnóstico permita superar las puertas actuales.

## Verificación del diagnóstico equivalente

La suite completa finalizó con **267/267 pruebas locales satisfactorias**: ocho nuevas de la libreta, trece del corredor diagnóstico y doce de instrumentación. Las demás pruebas existentes también se conservaron. Se ejecutó una reconstrucción completa sobre veinte matrices sintéticas de diez entradas controladas; no se procesaron radiografías OAI localmente.

El corredor lee el cierre ya existente, contrasta el CSV final y sus métricas, respuestas, huellas, propuestas completadas y ventanas guardadas. No llama a las funciones de revisión pendiente ni de cierre. Revalida originales y separación congelada, exige veinte igualdades exactas de cajas y píxeles, y comprueba la réplica instrumentada contra el núcleo y el adaptador sin modificar. Las entradas —incluida configuración, auditoría, separación, cierre anterior y paquetes fuente— se vuelven a cotejar antes de escribir un registro de diagnóstico satisfactorio.

Las pruebas incluyen alteraciones deliberadas de archivos, respuestas, ventanas, coordenadas y píxeles, así como pérdida de una ventana aun con una huella del manifiesto recalculada. Esas discrepancias detienen la entrega sin sobrescribir el cierre ni declarar un diagnóstico completo. La salida diagnóstica anterior tampoco se reemplaza.

Las tres rutas del detector oscuro se prueban con rectángulos artificiales: ausencia de detección, detección fuera de la caja final y detección con intersección en la caja final. La intersección se mide sobre el contorno seleccionado en la cuadrícula de trabajo; no identifica por sí sola una máscara de anonimización y no constituye un nuevo control de calidad de producción.

La galería conserva el denominador de veinte filas y prepara vistas espaciales 1:1 de las seis incidencias. La ROI visual no contiene superposiciones; el contexto lleva un borde en una copia separada. Se distingue claramente la ventana de intensidad de la resolución espacial. La relación marca/anatomía y la factibilidad de una corrección se mantienen pendientes de interpretación técnica, sin inventar un diagnóstico ni una nueva aceptación. El estado previsto es `diagnostic_evidence_ready_pending_interpretation`, no aprobación del método.
