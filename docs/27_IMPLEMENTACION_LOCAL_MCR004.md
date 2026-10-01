# MCR-2026-004 — implementación preparada y publicada, regresión pendiente

## Estado y autorización

El investigador aprobó el contenido técnico 1.0 de MCR-2026-004. La condición previa 8.2 se satisfizo al ejecutar la libreta 12 en Colab y contrastar los dos JSON del rechazo 8/20 de v0.4. La implementación se preparó y comprobó primero **solo localmente**. Posteriormente, el investigador autorizó de forma separada publicar el código, la configuración, las pruebas y la libreta 13 en la rama actual, y depositar la libreta en `03_notebooks`. Esa copia privada fue verificada en la carpeta autorizada. La publicación no ejecuta radiografías, no selecciona participantes nuevos y no cierra el paso 4.

## Código y procedencia

- Fuente externa: Emory-HITI/knee-crop, revisión `c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93`, MIT de HITI-LAB 2025. `src/knee/third_party/emory_hiti/` contiene su `config.py` íntegro y las funciones de `pipeline.py` líneas 112–436 necesarias para el camino unilateral. Se omitieron el CLI, la separación bilateral externa y las escrituras upstream.
- La comparación textual con la revisión fijada encontró igualdad de las 325 líneas del núcleo extraído después de deshacer únicamente el import adaptado y el retorno opcional de geometría. `config.py` también coincide íntegramente. El retorno por defecto de `process_knee_side` no cambia.
- `src/knee/roi_mcr004.py` adapta la mitad ya separada: p1/p99 a una copia de trabajo de 8 bits, preprocesamiento con traza de recortes/escalas, cálculo original de `left_min`, `right_min` y `notch`, rechazo de truncamientos, inversión al DICOM con límites semiabiertos y redondeo exterior, extracción sin transformación de los píxeles nativos y abstención por rodilla.
- `src/knee/roi_mcr004_pilot.py` verifica cierre de v0.4, auditoría y separación congelada; limita la ejecución a los diez estudios históricos / veinte rodillas y genera resultados privados, recortes nativos, vistas ciegas, una hoja de revisión vacía y un resumen sin desenlaces. No reescribe resultados previos.
- `configs/roi_mcr004_historical.example.json`, `requirements-mcr004.txt` y la libreta 13 fijan rutas relativas, dependencias y orden de ejecución. La libreta usa un entorno virtual separado en Colab y registra commit, versiones y huellas.

La adaptación no reproduce la validación publicada ni demuestra superioridad clínica. Se conserva el algoritmo y se cambia su interfaz/contrato de errores y de geometría. No hay pesos ni entrenamiento.

## Verificación local ejecutada

Se instaló en un entorno virtual separado NumPy 1.26.4, SciPy 1.14.1, OpenCV 4.10.0.84, pydicom 3.0.1 y Pillow 11.3.0, además del paquete local con sus dependencias declaradas. El entorno existente con NumPy 2.5.3 no se modificó. Once pruebas nuevas de núcleo y corredor pasaron: equivalencia de salida del retorno instrumentado y del preprocesamiento, ambas lateralidades sin espejo, offsets, espaciado anisótropo, píxeles nativos, redondeo/truncamiento, imágenes uniformes/pequeñas, repetibilidad, cierre previo obligatorio, denominador 20, rechazo de parámetros cambiados y fallo de escritura convertido en abstención. Tres pruebas documentales adicionales verificaron los límites de la libreta. La batería completa pasó **200/200** en el entorno aislado. En el entorno normal, 189 pasaron y once se omitieron porque allí no se instalaron OpenCV/SciPy. La libreta 13 se validó como JSON y sus celdas compilan; **no se ejecutó en Colab**.

Estas pruebas usan matrices sintéticas, no radiografías de OAI. Por tanto, `candidate.runtime_verified=true` en el registro significa solo ejecución técnica sintética; `candidate.local_performance_verified=false` permanece.

## Corrección operativa de la libreta 13 (v1.1)

La primera ejecución del investigador en Colab 2026.07/Python 3.12 se detuvo **antes del piloto**: `venv` terminó con código 1 durante el subproceso `ensurepip`. El motivo interno de `ensurepip` no quedó visible; el fallo no informa nada sobre la localización anatómica. La libreta 13 v1.1 fija `virtualenv==21.7.11`, desactiva sus actualizaciones periódicas de paquetes de arranque, comprueba Python 3.12 y muestra la salida de error si la creación vuelve a fallar. `virtualenv` usa su mecanismo de paquetes de arranque incluido, sin alterar `requirements-mcr004.txt`, el código ROI ni los umbrales. La corrección local se valida con pruebas, pero no se atribuye una ejecución Colab exitosa hasta que el investigador la confirme. La copia v1.0 ejecutada parcialmente se conserva en el historial privado.

## Revisión y puerta posterior

### Corrección operativa v1.3: serialización y diagnóstico

La ejecución con código `359dcb6` creó recortes y vistas parciales en Drive, pero terminó sin los CSV finales ni el resumen de regresión. La salida guardada de Colab solo contiene `CalledProcessError`; no permite afirmar por sí sola cuál fue la excepción interna.

Se reprodujo localmente un defecto del adaptador: las funciones de detección de líneas devuelven coordenadas NumPy y la traza podía contener `numpy.int64`. Al serializarla, el escritor JSON produce `TypeError: Object of type int64 is not JSON serializable`. Es compatible con una interrupción después de guardar una vista, aunque su atribución exacta a la ejecución real queda pendiente de confirmación en Colab.

Se convierten las cuatro coordenadas de líneas a enteros Python conservando exactamente sus valores. La prueba de regresión comprueba igualdad de los píxeles de preprocesamiento con el núcleo fijado y serialización de toda la traza. El corredor serializa la traza antes de escribir los archivos y registra una abstención ante un fallo de serialización por rodilla. La libreta v1.3 captura y muestra el error interno completo del subproceso. Las 201 pruebas locales pasan y todas las celdas compilan. El núcleo externo, la geometría y los umbrales permanecen iguales.

La salida incompleta se conserva como ejecución fallida en Drive antes de liberar la ruta de la repetición. La regresión histórica sigue sin completar y debe ejecutarla el investigador en Colab.

La hoja `revision_tecnica_ciega.csv` conserva veinte filas, incluso abstenciones. Para cada candidata se registran lateralidad, cobertura, encuadre, visualización, contaminación crítica, advertencia periférica y decisión: `aceptable`, `aceptable_con_advertencia_periferica`, `rechazado` o `no_evaluable`. Una abstención permanece en el denominador y no se reclasifica como aceptada. La revisión cotejará la mitad original con la ROI, sin desenlace ni predicción. Esta entrega no llena respuestas ni simula una segunda lectura.

La regresión histórica solo puede superar la puerta si `Q≥19/20`, `F=0`, `W≤2/20`, todas las candidatas tienen revisión técnica y las pruebas de integridad continúan pasando. Si falla, se documenta y se detiene; no se ajusta automáticamente. La selección nueva de veinte participantes de desarrollo se pospone hasta superar la regresión y fijar el código, la configuración y la guía. El procesamiento masivo, las particiones, el entrenamiento y la prueba reservada siguen bloqueados.

## Siguiente acción

Tras sincronizar esta revisión de la rama actual, el investigador podrá ejecutar la libreta 13 en Colab y devolver el resumen y las veinte vistas para revisión técnica. Esta publicación por sí sola no acredita el rendimiento del candidato ni autoriza una muestra nueva.
