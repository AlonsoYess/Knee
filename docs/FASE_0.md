# Fase 0  Base reproducible

## Estado y procedencia

La gobernanza y los entregables documentales de los pasos 1 a 4 fueron aprobados por el investigador el 27 de septiembre de 2026. La arquitectura reproducible GitHub–Colab–Drive del paso 5 fue aprobada como versión 1.0 el 28 de septiembre de 2026. El paso 6 se ejecutó posteriormente en Colab sobre el commit público `00255b76acb8f6151a839b0fe8d223c217957f68`, utilizando las fuentes privadas de Drive. Las evidencias fueron verificadas y la Fase 0 quedó cerrada técnicamente el 28 de septiembre de 2026.

La cohorte auditada procede de una extracción autorizada de OAI. La identificación del paquete, la consulta de extracción y los archivos tabulares se conservan en almacenamiento restringido; no se copian a este repositorio público. El capítulo III especifica el criterio de selección y las fuentes OAI. El código de esta fase verifica los archivos recibidos sin recalcular ni modificar la cohorte.

La ejecución local y la ejecución controlada en Colab confirmaron que los archivos privados coinciden en las verificaciones implementadas. Los recuentos exactos y las huellas permanecen en la auditoría privada. Esto comprueba la integridad tabular y del manifiesto; la lectura y calidad de todos los DICOM siguen pendientes para la Fase 1.

## Organización de Drive

La estructura privada separa administración y versiones, fuentes académicas, datos restringidos, DICOM, salidas experimentales y entregables. Las rutas exactas se mantienen en el inventario privado.

El nombre del CSV recibido se normalizó en Drive sin cambiar su contenido. El libro de documentación tenía dos copias locales con la misma huella SHA-256 y se almacenó una sola copia canónica. La variable de entorno `KNEE_DATA_ROOT` apunta a la raíz privada seleccionada por el investigador; no se edita el repositorio para incorporar una ruta personal. El resumen público se encuentra en [`02_ESTRUCTURA_DRIVE_Y_FUENTES.md`](02_ESTRUCTURA_DRIVE_Y_FUENTES.md) y la evidencia exacta permanece en el inventario privado.

## Ejecutar en Colab

Abrir [la libreta de arranque](../notebooks/00_arranque_colab.ipynb), conectar Drive y ejecutar las celdas en orden. La libreta es un orquestador delgado: clona desde GitHub la rama declarada, rechaza un directorio reutilizado, registra el commit resuelto, instala el paquete, ejecuta las pruebas y llama a los módulos de `src/knee`. No contiene lógica de preparación, entrenamiento ni inferencia.

La auditoría verifica el CSV contra la hoja `Manifiesto rodillas` y escribe en `outputs/auditorias/` dos JSON agregados: auditoría y registro de ejecución. La última celda exige que la bitácora corresponda al mismo commit clonado y a las mismas huellas de entrada. Una ejecución fallida se detiene antes de aceptar el cierre. Para otra revisión se debe restablecer el entorno de ejecución y volver a ejecutar la libreta completa; no se actualiza silenciosamente un clon anterior.

También se puede ejecutar localmente después de configurar `KNEE_DATA_ROOT` fuera del repositorio:

```bash
python -m pip install -e .
knee-audit --config configs/paths.example.json
knee-run-record --config configs/paths.example.json
python -m unittest discover -s tests -v
```

## Qué comprueba la auditoría

- Orden y presencia de las 21 columnas del CSV y campos esenciales del manifiesto.
- Clave única participante–rodilla, lateralidad 1/2 y un solo vínculo radiográfico basal por participante.
- KL basal 2/3, etiqueta coherente con incremento KL >= 1 y fecha V00 anterior a V06.
- Edad, sexo, IMC presente positivo, y conteos de IMC ausentes.
- Igualdad de las claves, barcode, ruta de imagen, KL, etiqueta y fechas entre CSV y manifiesto.
- SHA-256 de los dos archivos para identificar la versión exacta utilizada.

Los reportes contienen recuentos y huellas, no identificadores ni rutas de participantes. Deben permanecer en Drive hasta confirmar que sus condiciones de publicación permiten compartirlos. No se considera que una ruta S3 pruebe que un DICOM fue descargado o sea utilizable.

## Cierre técnico verificado

La ejecución en Colab completó las pruebas automáticas, la validación del control de cambios, la auditoría de la cohorte y la generación de la bitácora sin errores. La auditoría y la bitácora privadas coinciden en sus recuentos agregados y huellas de entrada, y la bitácora identifica el mismo commit clonado por la libreta. Los detalles privados permanecen en Drive y el resumen publicable se conserva en [`05_CIERRE_FASE_0.md`](05_CIERRE_FASE_0.md).

Con estas evidencias se cumple el criterio técnico de cierre de la Fase 0. Este cierre no autoriza entrenamiento, procesamiento masivo de DICOM ni apertura de la prueba reservada.
